"""Test two-chain complex prediction with template for only one chain."""
import os
os.environ["PROTENIX_DATA_ROOT_DIR"] = os.path.expanduser("~/.protenix")

import copy
import json
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

# JSON input for featurization
sample = {
    "name": "two_chain_test",
    "sequences": [
        {"proteinChain": {"sequence": SEQUENCE_A, "count": 1}},
        {"proteinChain": {"sequence": SEQUENCE_B, "count": 1}},
    ],
}


# ── 3. Featurize ───────────────────────────────────────────────────────────────

from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.utils import make_dummy_feature, data_type_transform

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Skip MSA search for faster testing - use dummy MSA
print("\nSkipping MSA search (using dummy MSA for faster test)...")

# Featurize
print("Featurizing...")
sample2feat = SampleDictToFeatures(sample)
features_dict, atom_array, token_array = sample2feat.get_feature_dict()
features_dict["distogram_rep_atom_mask"] = torch.Tensor(
    atom_array.distogram_rep_atom_mask
).long()

# Use dummy MSA features for faster testing
features_dict = make_dummy_feature(features_dict, dummy_feats=["msa"])
features_dict = data_type_transform(features_dict)


# ── 4. Build Template Features (Chain A only) ──────────────────────────────────

from protenix.data.template import ChainInput, build_templates_from_chains

# Download insulin structure for template
TEMPLATE_PDB = os.path.join(OUTPUT_DIR, "4ins.pdb")
if not os.path.exists(TEMPLATE_PDB):
    import urllib.request
    print("Downloading insulin structure (4INS) for template...")
    urllib.request.urlretrieve("https://files.rcsb.org/download/4INS.pdb", TEMPLATE_PDB)

# Load template structure
structure = gemmi.read_structure(TEMPLATE_PDB)
template_chain_a = structure[0]['A']  # Insulin A chain

print("\nBuilding template features...")
print(f"  Chain A: using template from 4INS chain A")
print(f"  Chain B: NO template")

# Build templates using the new interface
chains = [
    ChainInput(sequence=SEQUENCE_A, compute_msa=True, template=template_chain_a),
    ChainInput(sequence=SEQUENCE_B, compute_msa=True, template=None),  # No template!
]

templates = build_templates_from_chains(chains)
template_feats = templates.as_torch_dict()
features_dict.update(template_feats)

# Verify template structure
print(f"\nTemplate features:")
print(f"  template_aatype: {features_dict['template_aatype'].shape}")
print(f"  template_distogram: {features_dict['template_distogram'].shape}")

# Check masks
N_a, N_b = len(SEQUENCE_A), len(SEQUENCE_B)
mask = templates.atom_mask[0]
print(f"\nTemplate atom coverage:")
print(f"  Chain A atoms: {mask[:N_a].sum():.0f}")
print(f"  Chain B atoms: {mask[N_a:].sum():.0f} (should be 0)")

N_token = features_dict["token_index"].shape[0]
N_atom = features_dict["atom_to_token_idx"].shape[0]
N_msa = features_dict["msa"].shape[0]
print(f"\nFeaturized: {N_token} tokens, {N_atom} atoms, {N_msa} MSA sequences")


# ── 5. Convert to JAX ──────────────────────────────────────────────────────────

print("\nConverting features to JAX...")
jax_features = {}
for key, value in features_dict.items():
    if isinstance(value, torch.Tensor):
        jax_features[key] = jnp.array(value.numpy())
    else:
        jax_features[key] = value

jax_features["atom_rep_atom_idx"] = np.array(
    features_dict["distogram_rep_atom_mask"]
).nonzero()[0]


# ── 6. Run Inference ───────────────────────────────────────────────────────────

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


# ── 7. Save Output ─────────────────────────────────────────────────────────────

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
