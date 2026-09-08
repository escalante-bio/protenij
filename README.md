# Protenij: Protein + X + J

Translation of [Protenix](https://github.com/bytedance/Protenix) to JAX/Equinox.
This is pretty rough, we suggest using it through [mosaic](https://github.com/escalante-bio/mosaic).

## Installation

PyTorch is an **optional** dependency. The full inference pipeline (featurization, model loading, structure prediction) runs without it.

```bash
# Inference only (no PyTorch)
uv sync

# With PyTorch (needed for converting checkpoints from the original Protenix format)
uv sync --extra torch
```

## Serialized models

Pre-converted Equinox models skip the PyTorch dependency entirely and load in under a second.

| Model | Params | `.eqx` size |
|-------|--------|-------------|
| `protenix_tiny_default_v0.5.0` | 110M | 438 MB |
| `protenix_mini_default_v0.5.0` | 134M | 536 MB |
| `protenix_base_default_v1.0.0` | 368M | 1474 MB |
| `protenix_base_20250630_v1.0.0` | 368M | 1474 MB |

### Loading

Models are hosted on [HuggingFace](https://huggingface.co/nickrb/protenij) and downloaded automatically on first use.

```python
from protenix.backend import load_model

# Downloads from HuggingFace, caches to ~/.protenix/
model = load_model("protenix_base_default_v1.0.0")

# Or load from an explicit path
model = load_model("~/.protenix/protenix_base_default_v1.0.0")
```

### Translating from a PyTorch checkpoint

Requires the `torch` extra.

```bash
uv sync --extra torch
python translate_models.py
```

This downloads any missing checkpoints, converts each model to Equinox, saves `.eqx` + `.skeleton.pkl` to `~/.protenix/`, and verifies a bit-exact round-trip.

## Atom padding (JAX inference)

Atom padding is opt-in. It gives complexes with different native atom counts
identical atom dimensions for JIT reuse or sequence batching, provided their
other feature dimensions also match. Build native features first, then pad each
unbatched dictionary before transferring/batching it:

```python
from protenix.atom_padding import pad_atom_features

# Round up to a multiple of 256, or choose an explicit common atom_count.
padded = pad_atom_features(features, padding_multiple=256)
# padded = pad_atom_features(features, atom_count=4352)
output = model(input_feature_dict=padded, N_cycle=10, N_sample=2,
               N_steps=200, key=key)
output = output.unpad()  # On the host, before atom scoring or structure export.
```

`atom_pad_mask` is True for real atoms, False for padding. It is independent of
`ref_mask`, which describes reference-conformer availability: a real atom may
have `ref_mask=0`. Local atom attention excludes padded keys and queries,
atom-to-token means count only real atoms, and diffusion centers over real atoms
and zeros absent coordinates. Atom confidence logits at padding rows are zero;
**zero logits are not confidence scores**. Use the presence mask or `unpad()` to
exclude those rows before downstream scoring. Token-level PAE/PDE/distograms and
representative atom indices are unchanged.

Calls without `atom_pad_mask` retain the original behavior and random stream.
Padding changes the random tensor shape, so the same seed is not a guarantee of
identical valid-atom noise across buckets. For numerical parity checks,
`sample_diffusion` accepts optional `initial_noise` and `step_noise` arrays; use
identical standard-normal values on the real atoms in both runs. This does not
change the default sampler or introduce a different noise distribution.

The helper covers the native inference atom-feature schema (including both axes
of `bond_mask`). Extra atom-indexed fields need explicit `extra_atom_axes`;
training labels and custom nested atom features are not handled automatically.
Token, MSA and template dimensions are not padded. After vmapping sequences,
unpad each sequence's output separately. Consumers using the individual model
stages must carry `features["atom_pad_mask"]` to atom reductions/exports themselves.
The PyTorch implementation is unchanged.

CPU regression tests require no checkpoint:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -v
```

`scripts/validate_atom_padding.py` runs optional real-checkpoint parity checks on
trusted local feature pickles and reports lowering, compilation, and warm
execution separately. It uses a short sampling schedule by default; its output
is a validation report, not a production design evaluation.
