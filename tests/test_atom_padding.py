"""CPU-sized padding invariance tests; no checkpoints or network needed."""

import unittest
from io import StringIO

from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from protenix.atom_padding import center_atom_coordinates, pad_atom_features
from protenix.protenij import (
    ConfidenceMetrics,
    Outputs,
    ProtenixAttention,
    average_over_atoms,
    sample_diffusion,
)


class AtomPaddingTests(unittest.TestCase):
    def test_feature_schema_and_pitch(self):
        # Token count deliberately equals atom count: token/MSA axes must stay put.
        features = {
            "atom_to_token_idx": np.arange(5),
            "ref_pos": np.ones((5, 3)),
            "ref_mask": np.array([1, 0, 1, 1, 1]),
            "bond_mask": np.eye(5),
            "msa": np.ones((5, 5)),
            "atom_rep_atom_idx": np.arange(5),
        }
        padded = pad_atom_features(features, padding_multiple=4)
        self.assertEqual(padded["ref_pos"].shape, (8, 3))
        self.assertEqual(padded["bond_mask"].shape, (8, 8))
        self.assertIs(padded["msa"], features["msa"])
        self.assertIs(padded["atom_rep_atom_idx"], features["atom_rep_atom_idx"])
        assert_array_equal(padded["atom_pad_mask"], [1, 1, 1, 1, 1, 0, 0, 0])
        # A real atom with no reference conformer is still present.
        self.assertTrue(padded["atom_pad_mask"][1])
        assert_array_equal(features["ref_pos"], padded["ref_pos"][:5])
        self.assertNotIn("atom_pad_mask", features)
        self.assertEqual(pad_atom_features(features)["ref_pos"].shape[0], 256)
        with self.assertRaises(ValueError):
            pad_atom_features(features, 4)
        with self.assertRaises(ValueError):
            pad_atom_features(features, padding_multiple=0)

    def test_pooling_excludes_padding_from_sum_and_count(self):
        x = jnp.arange(30, dtype=jnp.float32).reshape(2, 5, 3)
        mapping = jnp.array([0, 0, 1, 1, 1])
        expected = average_over_atoms(x, mapping, 2)
        padded = jnp.pad(x, ((0, 0), (0, 7), (0, 0)), constant_values=jnp.nan)
        mask = jnp.arange(12) < 5
        # Invalid padding indices must not participate in gather/scatter.
        indices = jnp.concatenate([mapping, jnp.full(7, 999)])
        actual = jax.jit(average_over_atoms, static_argnums=2)(padded, indices, 2, mask)
        assert_allclose(actual, expected, atol=0, rtol=0)
        assert_array_equal(average_over_atoms(padded, indices, 3, mask)[:, 2], 0)

    def test_centering_ignores_adversarial_padding(self):
        x = jnp.arange(30, dtype=jnp.float32).reshape(2, 5, 3)
        mask = jnp.arange(12) < 5
        padded = jnp.pad(x, ((0, 0), (0, 7), (0, 0)), constant_values=jnp.nan)
        actual = center_atom_coordinates(padded, mask)
        assert_allclose(actual[:, :5], center_atom_coordinates(x), atol=0, rtol=0)
        assert_array_equal(actual[:, 5:], 0)
        assert_array_equal(center_atom_coordinates(padded, jnp.zeros(12, bool)), 0)

    def test_local_attention_crosses_partial_windows_and_masks_empty_windows(self):
        rng = np.random.default_rng(7)
        for n in (3, 4, 5, 9):
            for samples in (None, 2):
                shape = (2, n, 3) if samples is None else (samples, 2, n, 3)
                q, k, v = [
                    jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
                    for _ in range(3)
                ]

                def attend(q, k, v, mask=None):
                    return ProtenixAttention._local_attention(
                        q=q, k=k, v=v, n_queries=4, n_keys=8, atom_mask=mask
                    )

                expected = attend(q, k, v)
                widths = [(0, 0)] * len(shape)
                widths[-2] = (0, 20 - n)
                padded = [
                    jnp.pad(x, widths, constant_values=jnp.nan) for x in (q, k, v)
                ]
                actual = jax.jit(attend)(*padded, jnp.arange(20) < n)
                assert_allclose(actual[..., :n, :], expected, atol=2e-6, rtol=2e-6)
                assert_array_equal(actual[..., n:, :], 0)

    def test_diffusion_controlled_noise_parity(self):
        rng = np.random.default_rng(21)
        n, bucket, samples, steps = 5, 16, 2, 4
        initial = jnp.asarray(rng.normal(size=(samples, n, 3)), dtype=jnp.float32)
        noise = jnp.asarray(rng.normal(size=(steps, samples, n, 3)), dtype=jnp.float32)

        # A nonlinear deterministic denoiser sensitive to both input and center.
        def denoise(*, x_noisy, **kwargs):
            return 0.2 * jnp.tanh(x_noisy)

        common = {
            "denoise_net": denoise,
            "s_inputs": jnp.ones((2, 3)),
            "s_trunk": None,
            "z_trunk": None,
            "N_sample": samples,
            "noise_schedule": jnp.array([5.0, 3.0, 2.0, 1.0, 0.0]),
            "key": jax.random.key(0),
        }
        native = sample_diffusion(
            input_feature_dict={"atom_to_token_idx": jnp.zeros(n, int)},
            initial_noise=initial,
            step_noise=noise,
            **common,
        )
        padded = sample_diffusion(
            input_feature_dict={
                "atom_to_token_idx": jnp.zeros(bucket, int),
                "atom_pad_mask": jnp.arange(bucket) < n,
            },
            initial_noise=jnp.pad(
                initial, ((0, 0), (0, bucket - n), (0, 0)), constant_values=jnp.nan
            ),
            step_noise=jnp.pad(
                noise,
                ((0, 0), (0, 0), (0, bucket - n), (0, 0)),
                constant_values=jnp.nan,
            ),
            **common,
        )
        assert_allclose(padded[:, :n], native, atol=2e-6, rtol=2e-6)
        assert_array_equal(padded[:, n:], 0)

    def test_default_sampler_rng_matches_explicit_noise(self):
        samples, n, steps = 2, 5, 3
        key = jax.random.key(17)
        shape = (samples, n, 3)
        initial = jax.random.normal(key, shape)
        noise = []
        step_key = key
        for _ in range(steps):
            step_key = jax.random.fold_in(step_key, 1)
            noise.append(jax.random.normal(step_key, shape))

        def denoise(*, x_noisy, **kwargs):
            return 0.2 * jnp.tanh(x_noisy)

        common = {
            "denoise_net": denoise,
            "s_inputs": jnp.ones((2, 3)),
            "s_trunk": None,
            "z_trunk": None,
            "N_sample": samples,
            "input_feature_dict": {"atom_to_token_idx": jnp.zeros(n, int)},
            "noise_schedule": jnp.array([5.0, 3.0, 1.0, 0.0]),
            "key": key,
        }
        expected = sample_diffusion(**common)
        actual = sample_diffusion(
            initial_noise=initial, step_noise=jnp.stack(noise), **common
        )
        # Explicit tensors versus RNG inside scan can fuse slightly differently.
        assert_allclose(actual, expected, atol=1e-7, rtol=1e-5)

    def test_export_and_comparison_exclude_padding(self):
        from scripts.validate_atom_padding import comparison_arrays, compare_outputs

        atoms = AtomArray(3)
        atoms.coord[:] = 0
        atoms.chain_id[:] = "A"
        atoms.res_id[:] = 1
        atoms.res_name[:] = "ALA"
        atoms.atom_name[:] = ["N", "CA", "C"]
        atoms.element[:] = ["N", "C", "C"]
        coordinates = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        metrics = ConfidenceMetrics(
            jnp.ones((2, 3, 50)), jnp.ones((2, 3, 3, 64)),
            jnp.ones((2, 3, 3, 64)), jnp.ones((2, 3, 2)),
        )
        native = Outputs(jnp.asarray(coordinates), metrics, jnp.ones((3, 3, 64)))
        # Interspersed absent rows catch accidental prefix slicing; NaNs must
        # never enter exported structures or numerical comparisons.
        mask = np.array([True, False, True, False, True])
        def pad(values):
            result = np.full((*values.shape[:-2], 5, values.shape[-1]), np.nan)
            result[..., mask, :] = values
            return jnp.asarray(result)
        padded = Outputs(
            pad(native.coordinates),
            ConfidenceMetrics(pad(metrics.plddt_logits), metrics.pae_logits,
                              metrics.pde_logits, pad(metrics.resolved_logits)),
            native.distogram_logits, jnp.asarray(mask),
        )
        for output in (native, padded):
            structures = output.to_atom_arrays(atoms)
            self.assertEqual(len(structures), 2)
            for i, structure in enumerate(structures):
                assert_array_equal(structure.coord, coordinates[i])
                assert_array_equal(structure.atom_name, atoms.atom_name)
                pdb = PDBFile()
                pdb.set_structure(structure)
                buffer = StringIO()
                pdb.write(buffer)
                buffer.seek(0)
                restored = PDBFile.read(buffer).get_structure(model=1)
                assert_array_equal(restored.coord, coordinates[i])
            for actual, expected in zip(comparison_arrays(output), comparison_arrays(native)):
                assert_array_equal(actual, expected)
        assert_array_equal(atoms.coord, 0)  # Export does not mutate metadata input.
        self.assertEqual(padded.coordinates.shape, (2, 5, 3))
        report = compare_outputs(native, padded, {"atom_to_token_idx": np.arange(3)})
        self.assertTrue(all(item["all_atom_rmsd"] == 0 for item in report["geometry"]))
        with self.assertRaisesRegex(ValueError, "original atom array"):
            padded.to_atom_arrays(AtomArray(2))
        malformed = Outputs(padded.coordinates, padded.confidence_metrics,
                            padded.distogram_logits, jnp.ones(4, dtype=bool))
        with self.assertRaisesRegex(ValueError, "boolean mask for one sequence"):
            malformed.to_atom_arrays(atoms)
        batched = jax.tree.map(lambda x: jnp.stack([x, x]), padded)
        with self.assertRaisesRegex(ValueError, "each sequence separately"):
            batched.to_atom_arrays(atoms)

    def test_one_jit_trace_across_native_atom_counts(self):
        traces = []

        @jax.jit
        def run(features):
            traces.append(1)
            x = center_atom_coordinates(features["ref_pos"], features["atom_pad_mask"])
            return average_over_atoms(
                x, features["atom_to_token_idx"], 2, features["atom_pad_mask"]
            )

        for n in (5, 7, 6):
            features = {
                "ref_pos": np.ones((n, 3), np.float32),
                "atom_to_token_idx": np.arange(n, dtype=np.int32) % 2,
            }
            run(pad_atom_features(features, 8)).block_until_ready()
        self.assertEqual(len(traces), 1)


if __name__ == "__main__":
    unittest.main()
