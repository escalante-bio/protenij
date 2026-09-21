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
```

`atom_pad_mask` marks real atoms; `output.to_atom_arrays(atom_array)` handles it
when exporting structures. Custom atom fields require `extra_atom_axes`;
token, MSA and template dimensions are not padded. Identical seeds do not imply
identical samples across padding sizes.

## Token padding (JAX inference)

Token padding composes with atom padding in either order. It pads token axes
of singles, pairs, MSAs, templates, frame metadata and known constraint
features, without cropping any native token or atom:

```python
from protenix.token_padding import pad_token_features

padded = pad_token_features(features, token_count=256)  # Or padding_multiple=64.
padded = pad_atom_features(padded, atom_count=2048)
initial = model.embed_inputs(input_feature_dict=padded)
trunk = model.recycle(initial_embedding=initial, input_feature_dict=padded,
                      recycling_steps=1, key=key)
# On the host, retain just the real token rows for representation caches.
mask = padded["token_pad_mask"]
single = trunk.s[mask]
pair = trunk.z[mask][:, mask]
```

`token_pad_mask` is independent of `atom_pad_mask` and `ref_mask`. The public
model API propagates it through input embedding, templates, MSA reductions,
Pairformer recycling, diffusion token attention and confidence prediction.
Padded single/pair embeddings and output logits are zero. `Outputs` retains
both presence masks. Mask or remove dummy rows **before** softmax-based
confidence aggregation; zero logits do not imply zero confidence.

Custom trunk/recycling loops must pass `token_pair_mask(padded)` (from
`protenix.token_padding`) to the template, MSA and Pairformer modules, rather
than `pair_mask=None`. Custom token fields require explicit axes through
`extra_token_axes`; dotted names address nested fields, for example
`{"constraint_feature.custom": (0, 1)}`. Unknown fields are preserved, not
inferred from shapes. This helper does not add support for constraints that
the JAX model itself does not consume.

For compilation reuse, **all** array shapes and static arguments must match.
These helpers do not pad the number of MSA rows or templates, or change MSA
sampling. Use matching row/template counts separately. Vmap a model over
individual padded feature dictionaries; the presence masks describe one
sequence each. Different recycle/sample/step counts can still recompile.

Deterministic trunk comparisons should use the same key and native MSA rows.
For diffusion comparisons across atom buckets, supply identical native noise
via `sample_diffusion(initial_noise=..., step_noise=...)`, padded with zeros;
shape-dependent random draws otherwise confound padding comparisons.

CPU tests use randomized nonzero weights, NaN/sentinel dummy metadata,
combined atom/token padding, templates and sampled MSAs, one and three trunk
passes, two diffusion samples, confidence heads and shared-bucket JIT reuse:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p 'test_*padding.py' -v
```

The checkpoint validator also accepts `--token-count` and compares native
single/pair states and logits after removing padding, with matched diffusion
noise. For example, supply trusted, locally generated feature pickle files:

```bash
python scripts/validate_atom_padding.py --features first.pkl second.pkl \
  --token-count 256 --padding-multiple 2048 --cycles 3 --samples 2
```

It reports JIT reuse only when the complete padded feature signatures match;
a shared token/atom bucket alone does not guarantee equal MSA/template shapes.

Protenix-v2 checkpoint trunk validation on an H100 covered two uncropped
protein complexes (190/235 tokens), single-sequence and original cached MSAs
(13,092/11,306 rows), and one/three passes. All eight unpadded runs were bitwise
identical to the pre-token-padding code. With 256-token/2,048-atom padding,
maximum relative RMS differences were 1.73e-5 for pairs and 5.76e-7 for singles;
dummy outputs were exactly zero. The two single-sequence inputs reused one
padded trace per pass count; different full-MSA depths still compiled separately.
The three-pass, 235-token pair comparisons (with and without MSAs) failed
strict elementwise `atol=rtol=1e-3` despite the small relative RMS differences.
This is representation validation, not full diffusion/structure parity on
those complexes.

Full-checkpoint H100 inference also passed the validator's strict `1e-3`
comparison on two complete 8/9-token peptides (62/71 atoms), padded to 16
tokens and 128 atoms, with one pass, two samples, five diffusion steps and
matched noise. All 11 compared arrays passed; dummy outputs were zero and
the second padded input reused the compiled trace. Maximum aligned all-atom
RMSD was 0.00105 Å. Longer diffusion trajectories remain unvalidated here.
