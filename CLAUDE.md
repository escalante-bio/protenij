# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Protenij is a JAX/Equinox translation of [Protenix](https://github.com/bytedance/Protenix) (ByteDance), which implements the AlphaFold 3 (AF3) protein structure prediction architecture. The project is intended to be used through [mosaic](https://github.com/escalante-bio/mosaic).

## Build & Development

**Requirements:** Python 3.12 (exact version enforced), `uv` package manager.

```bash
# Install dependencies
uv pip install -e .

# Sync environment
uv sync
```

There is no formal test suite or linting configuration. One test file exists at `protenix/openfold_local/utils/geometry/test_utils.py`.

## Architecture

### Two-Layer Design: PyTorch Model + JAX Translation

The codebase has two parallel representations of the same model:

1. **PyTorch layer** (`protenix/model/`, `protenix/openfold_local/`): The original Protenix model code (from ByteDance) using `torch.nn.Module`. This includes the training runner (`protenix/runner/train.py`) and inference runner (`protenix/runner/inference.py`, `protenix/inference.py`). Triangular operations (attention, multiplicative update, dropout, OPM) are consolidated under `protenix/model/triangular/` which re-exports from `protenix/openfold_local/model/`. Checkpointing and chunking utilities are re-exported via `protenix/model/utils.py`.

2. **JAX/Equinox layer** (`protenix/protenij.py`, `protenix/backend.py`): Translation of the PyTorch modules into Equinox modules. `backend.py` provides the `from_torch` conversion dispatch system and base classes (`AbstractFromTorch`) for converting PyTorch models to JAX. `protenij.py` registers JAX equivalents for each PyTorch module and re-implements operations using `jax.numpy`.

### PyTorch-to-JAX Conversion Pattern

The `from_torch` singledispatch in `backend.py` converts PyTorch `nn.Module` instances to Equinox modules. Each PyTorch module class gets a registered converter via `@register_from_torch`. The `AbstractFromTorch` base class automates conversion by matching dataclass fields between PyTorch and Equinox modules.

### Key Modules (under `protenix/model/modules/`)

Code comments reference algorithm numbers from the AF3 paper:
- `embedders.py` — Input, template, and constraint embedders
- `pairformer.py` — MSA module, PairformerStack (Alg 17), TemplateEmbedder (Alg 16)
- `transformer.py` — Attention mechanisms
- `diffusion.py` — Diffusion conditioning and structure generation
- `confidence.py` — Confidence head (pLDDT, pAE, ranking)
- `primitives.py` — Linear, LayerNorm, Transition building blocks

### Triangular Operations (`protenix/model/triangular/`)

Consolidated import path for triangular operations previously in `openfold_local/model/`:
- `layers.py` — Re-exports: `LayerNorm`, `Linear`, `Attention`, `Dropout`, `DropoutRowwise`, `OuterProductMean`, init functions
- `triangular.py` — Re-exports: `TriangleMultiplicativeUpdate`, `TriangleAttention` and their variants

### Data Pipeline (`protenix/data/`)

- `parser.py` — Parses input structures (PDB, mmCIF, JSON)
- `dataset.py` — Dataset class for training/inference
- `featurizer.py` — Protein feature extraction
- `msa_featurizer.py` — Multiple sequence alignment processing
- `esm_featurizer.py` — ESM2 language model embeddings
- `ccd.py` — Chemical Component Dictionary handling

### Configuration System

Uses `ml-collections` ConfigDict with a hierarchical structure:
- `configs/configs_base.py` — Training hyperparameters
- `configs/configs_data.py` — Data pipeline settings
- `configs/configs_inference.py` — Inference settings
- `configs/configs_model_type.py` — Model variants (base: 368M params, mini: 135M, tiny: 110M)
- `config/config.py` — ConfigManager with custom types (RequiredValue, GlobalConfigValue)

### Model Variants

Defined in `configs/configs_model_type.py`:
- `protenix_base_default_v0.5.0` (368M params) — default
- `protenix_base_constraint_v0.5.0` (368M params) — with constraint features
- `protenix_mini_esm_v0.5.0` (135M params) — uses ESM2 embeddings
- `protenix_tiny_default_v0.5.0` (110M params)
- `protenix_base_default_v1.0.0` (368M params) — v1.0.0 with TemplateEmbedder (n_blocks=2)
- `protenix_base_20250630_v1.0.0` (368M params) — v1.0.0 with 2025 training cutoff

### Key Dependencies

- **JAX + Equinox**: Target framework for the translation
- **PyTorch**: Source framework (model definitions, training, DDP)
- **BioPython / Biotite / Gemmi / RDKit**: Molecular structure parsing and manipulation
- **ml-collections**: Configuration management
- **safetensors**: Checkpoint serialization
