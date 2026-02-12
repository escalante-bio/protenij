# Protenij: Protein + X + J

Translation of [Protenix](https://github.com/bytedance/Protenix) to JAX/Equinox.
This is pretty rough, we suggest using it through [mosaic](https://github.com/escalante-bio/mosaic).

## Serialized models

Pre-converted Equinox models skip the PyTorch dependency entirely and load in under a second.

| Model | Params | `.eqx` size |
|-------|--------|-------------|
| `protenix_tiny_default_v0.5.0` | 110M | 438 MB |
| `protenix_mini_default_v0.5.0` | 134M | 536 MB |
| `protenix_base_default_v1.0.0` | 368M | 1474 MB |
| `protenix_base_20250630_v1.0.0` | 368M | 1474 MB |

### Saving and loading

```python
from protenix.backend import save_model, load_model

# Save (after converting from PyTorch)
save_model(jax_model, "~/.protenix/protenix_base_default_v1.0.0")
# produces .eqx (array data) + .skeleton.pkl (pytree structure)

# Load (no PyTorch needed)
model = load_model("~/.protenix/protenix_base_default_v1.0.0")
```

### Translating from a PyTorch checkpoint

```bash
python translate_models.py
```

This downloads any missing checkpoints, converts each model to Equinox, saves `.eqx` + `.skeleton.pkl` to `~/.protenix/`, and verifies a bit-exact round-trip.
