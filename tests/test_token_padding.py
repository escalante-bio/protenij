"""Token/atom padding parity with learned CPU-sized inference modules."""

from dataclasses import fields
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from protenix.atom_padding import pad_atom_features
from protenix.token_padding import pad_token_features, TOKEN_FEATURE_AXES
from protenix.backend import _HAS_TORCH


def features(n):
    rng = np.random.default_rng(n)
    n_atoms = 2 * n
    f = {
        "atom_to_token_idx": np.arange(n_atoms, dtype=np.int32) // 2,
        "atom_to_tokatom_idx": np.arange(n_atoms, dtype=np.int32) % 2,
        "atom_rep_atom_idx": np.arange(n, dtype=np.int32) * 2,
        "residue_index": np.arange(n, dtype=np.int32),
        "token_index": np.arange(n, dtype=np.int32),
        "has_frame": np.ones(n, dtype=np.int32),
        "frame_atom_index": np.zeros((n, 3), dtype=np.int32),
        "asym_id": np.ones(n, np.int32), "entity_id": np.ones(n, np.int32),
        "sym_id": np.zeros(n, np.int32),
        "restype": np.eye(32, dtype=np.float32)[np.arange(n)],
        "profile": rng.random((n, 32), dtype=np.float32),
        "deletion_mean": rng.random(n, dtype=np.float32),
        "token_bonds": np.eye(n, dtype=np.float32),
        "constraint_feature": {name: rng.random((n,n,c), dtype=np.float32)
            for name,c in (("contact_atom",2),("contact",2),("pocket",1),("substructure",4))},
        "ref_pos": rng.normal(size=(n_atoms, 3)).astype(np.float32),
        "ref_charge": rng.normal(size=n_atoms).astype(np.float32),
        "ref_mask": np.ones(n_atoms, np.float32),
        "ref_space_uid": np.zeros(n_atoms, np.int32),
        "ref_element": rng.normal(size=(n_atoms, 128)).astype(np.float32),
        "ref_atom_name_chars": rng.normal(size=(n_atoms, 4, 64)).astype(np.float32),
        "msa": rng.integers(0, 32, (3, n), dtype=np.int32),
        "has_deletion": np.zeros((3, n), np.float32),
        "deletion_value": rng.random((3, n), dtype=np.float32),
        "template_aatype": rng.integers(0, 32, (1, n), dtype=np.int32),
        "template_distogram": rng.random((1, n, n, 39), dtype=np.float32),
        "template_pseudo_beta_mask": np.ones((1, n, n), np.float32),
        "template_unit_vector": rng.random((1, n, n, 3), dtype=np.float32),
        "template_backbone_frame_mask": np.ones((1, n, n), np.float32),
    }
    return f


def poison_tokens(f, n):
    result = dict(f)
    for name, axes in TOKEN_FEATURE_AXES.items():
        if name not in f or name == "token_pad_mask":
            continue
        value = f[name].copy()
        for axis in axes:
            index = [slice(None)] * value.ndim
            index[axis] = slice(n, None)
            value[tuple(index)] = np.nan if value.dtype.kind == "f" else 999999
        result[name] = value
    return result


class TokenSchemaTests(unittest.TestCase):
    def test_axes_composition_and_validation(self):
        f = features(3)
        p = pad_token_features(f, 8)
        self.assertEqual(p["msa"].shape, (3, 8))
        self.assertEqual(p["template_distogram"].shape, (1, 8, 8, 39))
        self.assertIs(p["ref_pos"], f["ref_pos"])
        assert_array_equal(p["token_pad_mask"], [1, 1, 1, 0, 0, 0, 0, 0])
        a = pad_atom_features(p, 16)
        b = pad_token_features(pad_atom_features(f, 16), 8)
        for va, vb in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            assert_array_equal(va, vb)
        assert_array_equal(p["constraint_feature"]["contact"][:3,:3], f["constraint_feature"]["contact"])
        self.assertEqual(p["frame_atom_index"].shape, (8,3))
        signatures = []
        for n in (3,5):
            padded = pad_token_features(pad_atom_features(features(n),16),8)
            signatures.append(jax.tree.map(lambda x: (x.shape,x.dtype), padded))
        self.assertEqual(signatures[0], signatures[1])
        for kwargs in ({"token_count": 2}, {"padding_multiple": 0}):
            with self.assertRaises(ValueError):
                pad_token_features(f, **kwargs)
        self.assertEqual(pad_token_features(f)["residue_index"].shape, (64,))
        self.assertNotIn("token_pad_mask", f)
        twice = pad_token_features(p, 10)
        assert_array_equal(twice["token_pad_mask"], np.arange(10) < 3)
        assert_array_equal(twice["restype"][:8], p["restype"])
        custom = f | {"custom": {"pairs": np.ones((3,3,2))}}
        padded_custom = pad_token_features(custom, 8, extra_token_axes={"custom.pairs": (0,1)})
        self.assertEqual(padded_custom["custom"]["pairs"].shape, (8,8,2))
        self.assertEqual(custom["custom"]["pairs"].shape, (3,3,2))

    def test_output_comparison_removes_token_axes(self):
        from protenix.protenij import Outputs, ConfidenceMetrics
        from scripts.atom_padding_diagnostics import comparison_arrays
        mask = jnp.array([True,True,False,False])
        logits = jnp.arange(2*4*4*3).reshape(2,4,4,3)
        output = Outputs(
            coordinates=jnp.zeros((2,6,3)),
            confidence_metrics=ConfidenceMetrics(jnp.zeros((2,6,50)), logits, logits,
                                                  jnp.zeros((2,6,2))),
            distogram_logits=logits[0], token_pad_mask=mask,
        )
        actual = comparison_arrays(output)
        assert_array_equal(actual[3], logits[:,:2,:2])
        assert_array_equal(actual[4], logits[:,:2,:2])
        assert_array_equal(actual[5], logits[0,:2,:2])


@unittest.skipUnless(_HAS_TORCH, "Random module construction requires torch")
class TokenModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from protenix.backend import from_torch
        from protenix.protenij import Protenix
        from protenix.model.modules.embedders import InputFeatureEmbedder, RelativePositionEncoding
        from protenix.model.modules.transformer import AtomAttentionEncoder
        from protenix.model.modules.pairformer import MSAModule, PairformerStack, TemplateEmbedder
        from protenix.model.modules.diffusion import DiffusionModule
        from protenix.model.modules.confidence import ConfidenceHead
        from protenix.model.modules.head import DistogramHead
        torch.manual_seed(72)

        def convert(module):
            module.eval()
            with torch.no_grad():
                for parameter in module.parameters():
                    if parameter.requires_grad:
                        parameter.normal_(0, 0.15)
            return from_torch(module)

        embed = InputFeatureEmbedder(c_atom=8, c_atompair=4, c_token=16)
        embed.atom_attention_encoder = AtomAttentionEncoder(
            has_coords=False, c_token=16, c_atom=8, c_atompair=4,
            n_blocks=1, n_heads=2, n_queries=4, n_keys=8,
        )
        args = {field.name: None for field in fields(Protenix)}
        args.update(
            input_embedder=convert(embed),
            relative_position_encoding=convert(RelativePositionEncoding(c_z=8)),
            template_embedder=convert(TemplateEmbedder(n_blocks=1, c=8, c_z=8)),
            msa_module=convert(MSAModule(n_blocks=2, c_m=8, c_z=8, c_s_inputs=81,
                msa_configs={"sample_cutoff": {"train": 2, "test": 2}})),
            pairformer_stack=convert(PairformerStack(n_blocks=2, n_heads=4, c_s=32, c_z=8)),
            linear_no_bias_sinit=convert(torch.nn.Linear(81, 32, bias=False)),
            linear_no_bias_zinit1=convert(torch.nn.Linear(32, 8, bias=False)),
            linear_no_bias_zinit2=convert(torch.nn.Linear(32, 8, bias=False)),
            linear_no_bias_token_bond=convert(torch.nn.Linear(1, 8, bias=False)),
            linear_no_bias_z_cycle=convert(torch.nn.Linear(8, 8, bias=False)),
            linear_no_bias_s=convert(torch.nn.Linear(32, 32, bias=False)),
            layernorm_z_cycle=convert(torch.nn.LayerNorm(8)),
            layernorm_s=convert(torch.nn.LayerNorm(32)),
            diffusion_module=convert(DiffusionModule(c_atom=8, c_atompair=4, c_token=16,
                c_s=32, c_z=8, c_s_inputs=81, atom_encoder={"n_blocks":1,"n_heads":2},
                transformer={"n_blocks":2,"n_heads":4}, atom_decoder={"n_blocks":1,"n_heads":2})),
            confidence_head=convert(ConfidenceHead(n_blocks=1, c_s=32, c_z=8, c_s_inputs=81)),
            distogram_head=convert(DistogramHead(c_z=8)),
        )
        cls.model = Protenix(**args)

    def test_trunk_recycles_nan_padding_and_compilation_reuse(self):
        model = self.model
        for passes in (1, 3):
            traces = []
            @eqx.filter_jit
            def run(f):
                traces.append(1)
                init = model.embed_inputs(input_feature_dict=f)
                state = model.recycle(initial_embedding=init, input_feature_dict=f,
                    recycling_steps=passes, key=jax.random.key(15))
                return init, state
            for n in (3, 5):
                f = features(n)
                native = run(f)
                traces.clear()
                p = poison_tokens(pad_token_features(pad_atom_features(f, 16), 8), n)
                padded = run(p)
                for a, b in zip(jax.tree.leaves(native), jax.tree.leaves(padded), strict=True):
                    index = (slice(n), slice(n)) if a.ndim == 3 else (slice(n),)
                    assert_allclose(b[index], a, atol=2e-5, rtol=2e-5)
                    assert_array_equal(b[n:], 0)
                    if b.ndim == 3:
                        assert_array_equal(b[:, n:], 0)
                # Native length change retraces, but both padded calls share one trace.
                self.assertEqual(len(traces), 1 if n == 3 else 0)

    def test_fully_masked_pairformer(self):
        single, pair = self.model.pairformer_stack(
            s=jnp.full((4,32), jnp.nan), z=jnp.full((4,4,8), jnp.nan),
            pair_mask=jnp.zeros((4,4), dtype=bool), key=jax.random.key(0),
        )
        assert_array_equal(single, 0)
        assert_array_equal(pair, 0)

    def test_token_only_padding(self):
        model = self.model
        native = features(3)
        padded = poison_tokens(pad_token_features(native, 8), 3)
        self.assertNotIn("atom_pad_mask", padded)
        outputs = []
        for f in (native, padded):
            init = model.embed_inputs(input_feature_dict=f)
            outputs.append(model.recycle(initial_embedding=init, input_feature_dict=f,
                recycling_steps=1, key=jax.random.key(15)))
        assert_allclose(outputs[1].s[:3], outputs[0].s, atol=2e-5, rtol=2e-5)
        assert_allclose(outputs[1].z[:3,:3], outputs[0].z, atol=2e-5, rtol=2e-5)
        assert_array_equal(outputs[1].s[3:], 0)
        assert_array_equal(outputs[1].z[3:], 0)
        assert_array_equal(outputs[1].z[:,3:], 0)

    def test_diffusion_and_confidence_native_vs_padded(self):
        model = self.model
        f = features(3)
        p = poison_tokens(pad_token_features(pad_atom_features(f, 16), 8), 3)
        outputs = []
        for features_ in (f, p):
            init = model.embed_inputs(input_feature_dict=features_)
            trunk = model.recycle(initial_embedding=init, input_feature_dict=features_,
                recycling_steps=2, key=jax.random.key(4))
            coords = np.arange(36, dtype=np.float32).reshape(2, 6, 3) / 10
            if features_ is p:
                coords = np.pad(coords, ((0,0),(0,10),(0,0)), constant_values=np.nan)
            denoised = eqx.filter_jit(model.diffusion_module)(
                x_noisy=jnp.asarray(coords), t_hat_noise_level=jnp.ones(2),
                input_feature_dict=features_, s_inputs=init.s_inputs,
                s_trunk=trunk.s, z_trunk=trunk.z)
            conf = model.confidence_metrics(initial_embedding=init, trunk_embedding=trunk,
                input_feature_dict=features_, coordinates=denoised, key=jax.random.key(5))
            outputs.append((denoised, conf))
        assert_allclose(outputs[1][0][:,:6], outputs[0][0], atol=2e-5, rtol=2e-5)
        assert_array_equal(outputs[1][0][:,6:], 0)
        for name in ("plddt_logits", "resolved_logits", "pae_logits", "pde_logits"):
            native = getattr(outputs[0][1], name)
            padded = getattr(outputs[1][1], name)
            if "pae" in name or "pde" in name:
                assert_allclose(padded[:,:3,:3], native, atol=2e-5, rtol=2e-5)
                assert_array_equal(padded[:,3:], 0)
                assert_array_equal(padded[:,:,3:], 0)
            else:
                assert_allclose(padded[:,:6], native, atol=2e-5, rtol=2e-5)
                assert_array_equal(padded[:,6:], 0)


if __name__ == "__main__":
    unittest.main()
