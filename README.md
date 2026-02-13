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
