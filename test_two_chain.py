"""Test two-chain complex prediction with template for only one chain."""
import os
os.environ["PROTENIX_DATA_ROOT_DIR"] = os.path.expanduser("~/.protenix")

import copy
from pathlib import Path

import numpy as np
import torch
import jax
import jax.numpy as jnp
import gemmi
from ml_collections.config_dict import ConfigDict

from protenix.configs.configs_base import configs as configs_base
from protenix.configs.configs_data import data_configs
from protenix.configs.configs_inference import inference_configs
from protenix.configs.configs_model_type import model_configs
from protenix.config import parse_configs


# ── 1. Config + Model Loading ──────────────────────────────────────────────────

MODEL_NAME = "protenix_base_default_v1.0.0"
CACHE_DIR = os.path.expanduser("~/.protenix")
OUTPUT_DIR = "./output_two_chain"

configs_base["use_deepspeed_evo_attention"] = False
configs = {**configs_base, **{"data": data_configs}, **inference_configs}
configs = parse_configs(configs=configs, fill_required_with_null=True)
configs.model_name = MODEL_NAME
configs.load_checkpoint_dir = CACHE_DIR
configs.update(ConfigDict(model_configs[MODEL_NAME]))

# Vanilla ODE sampler
configs.sample_diffusion["gamma0"] = 0.0
configs.sample_diffusion["step_scale_eta"] = 1.0
configs.sample_diffusion["N_step"] = 30

print(f"Model: {MODEL_NAME}")

# Create PyTorch model + load checkpoint
from protenix.model.protenix import Protenix as TorchProtenix

torch_model = TorchProtenix(configs)
n_params = sum(p.numel() for p in torch_model.parameters())
print(f"PyTorch model created ({n_params / 1e6:.2f}M params)")

checkpoint_path = os.path.join(CACHE_DIR, f"{MODEL_NAME}.pt")
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Handle DDP prefix
sample_key = list(checkpoint["model"].keys())[0]
if sample_key.startswith("module."):
    checkpoint["model"] = {k[len("module."):]: v for k, v in checkpoint["model"].items()}

torch_model.load_state_dict(checkpoint["model"], strict=configs.load_strict)
torch_model.eval()
print("Checkpoint loaded")

# Convert to JAX/Equinox
import protenix.protenij
from protenix.backend import from_torch

print("Converting to JAX/Equinox...")
jax_model = from_torch(torch_model)
print("JAX model ready")


# ── 2. Define Two-Chain Complex ────────────────────────────────────────────────

# Chain A: Short peptide from insulin A chain (21 residues)
SEQUENCE_A = "GIVEQCCTSICSLYQLENYCN"

# Chain B: Another short peptide - glucagon (29 residues)
SEQUENCE_B = "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT"

print(f"\n=== Two-Chain Complex ===")
print(f"Chain A: {len(SEQUENCE_A)} residues (will have template)")
print(f"Chain B: {len(SEQUENCE_B)} residues (NO template)")
print(f"Total: {len(SEQUENCE_A) + len(SEQUENCE_B)} residues")


# ── 3. Featurize ────────────────────────────────────────────────────────────────

from protenix.data.template import ChainInput, featurize

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Download insulin structure for template
TEMPLATE_PDB = os.path.join(OUTPUT_DIR, "4ins.pdb")
if not os.path.exists(TEMPLATE_PDB):
    import urllib.request
    print("Downloading insulin structure (4INS) for template...")
    urllib.request.urlretrieve("https://files.rcsb.org/download/4INS.pdb", TEMPLATE_PDB)

# Load template structure
structure = gemmi.read_structure(TEMPLATE_PDB)
template_chain_a = structure[0]['A']  # Insulin A chain

# Define chains and featurize in one step
# - compute_msa=False: use dummy MSA (just query sequence)
# - compute_msa=True: run actual MSA search via MMSeqs2 service
print("\nFeaturizing...")
features_dict, atom_array, token_array = featurize([
    ChainInput(
        sequence=SEQUENCE_A,
        compute_msa=False,  # Use dummy MSA for faster testing
        template=template_chain_a,  # Has template
    ),
    ChainInput(
        sequence=SEQUENCE_B,
        compute_msa=True,  # Run MSA search
        template=None,  # No template
    ),
])

# Verify features
print(f"\nTemplate features:")
print(f"  template_aatype: {features_dict['template_aatype'].shape}")
print(f"  template_distogram: {features_dict['template_distogram'].shape}")

N_token = features_dict["token_index"].shape[0]
N_atom = features_dict["atom_to_token_idx"].shape[0]
N_msa = features_dict["msa"].shape[0]
print(f"\nFeaturized: {N_token} tokens, {N_atom} atoms, {N_msa} MSA sequences")


# ── 4. Convert to JAX ──────────────────────────────────────────────────────────

print("\nConverting features to JAX...")

def to_jax(v):
    """Recursively convert numpy arrays to JAX arrays."""
    if isinstance(v, dict):
        return {k: to_jax(v2) for k, v2 in v.items()}
    return jnp.array(v)

jax_features = {k: to_jax(v) for k, v in features_dict.items()}


# ── 5. Run Inference ───────────────────────────────────────────────────────────

N_cycle = 4
N_sample = 1

print(f"\nRunning inference (N_cycle={N_cycle}, N_sample={N_sample})...")
key = jax.random.PRNGKey(42)
outputs = jax_model(
    input_feature_dict=jax_features,
    N_cycle=N_cycle,
    N_sample=N_sample,
    key=key,
)

print(f"Output coordinates shape: {outputs.coordinates.shape}")
print(f"Coordinate range: [{float(outputs.coordinates.min()):.2f}, {float(outputs.coordinates.max()):.2f}]")


# ── 6. Save Output ─────────────────────────────────────────────────────────────

from biotite.structure.io.pdb import PDBFile

coords = np.array(outputs.coordinates)
for i in range(coords.shape[0]):
    pred_atom_array = copy.deepcopy(atom_array)
    pred_atom_array.coord = coords[i]

    pdb = PDBFile()
    pdb.set_structure(pred_atom_array)
    out_path = os.path.join(OUTPUT_DIR, f"two_chain_sample_{i}.pdb")
    pdb.write(out_path)
    print(f"Saved {out_path}")

    assert not np.any(np.isnan(coords[i])), "NaN coordinates!"
    assert not np.all(coords[i] == 0), "All-zero coordinates!"

print("\n=== Two-chain test complete! ===")
