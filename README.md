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

Pad each unbatched feature dictionary to a common atom count for JIT reuse or
batching; other feature dimensions must also match.

```python
from protenix.atom_padding import pad_atom_features

padded = pad_atom_features(features, padding_multiple=256)  # Or atom_count=4352.
output = model(input_feature_dict=padded, N_cycle=10, N_sample=2,
               N_steps=200, key=key)
output = output.unpad()  # On the host, before scoring or export; per sequence if batched.
```

`atom_pad_mask` marks real atoms. Custom atom fields require `extra_atom_axes`;
token, MSA and template dimensions are not padded. Identical seeds do not imply
identical samples across padding sizes.
