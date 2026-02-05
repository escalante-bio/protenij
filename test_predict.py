"""Featurize a short protein, run JAX inference with the tiny model, and save PDB files."""
import os
os.environ["PROTENIX_DATA_ROOT_DIR"] = os.path.expanduser("~/.protenix")

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import jax
import jax.numpy as jnp
from ml_collections.config_dict import ConfigDict

from protenix.configs.configs_base import configs as configs_base
from protenix.configs.configs_data import data_configs
from protenix.configs.configs_inference import inference_configs
from protenix.configs.configs_model_type import model_configs
from protenix.config import parse_configs

# ── 1. Config + Model Loading ──────────────────────────────────────────────────

MODEL_NAME = "protenix_base_default_v1.0.0"
CACHE_DIR = os.path.expanduser("~/.protenix")
OUTPUT_DIR = "./output_test_predict"

configs_base["use_deepspeed_evo_attention"] = False
configs = {**configs_base, **{"data": data_configs}, **inference_configs}
configs = parse_configs(configs=configs, fill_required_with_null=True)
configs.model_name = MODEL_NAME
configs.load_checkpoint_dir = CACHE_DIR
configs.update(ConfigDict(model_configs[MODEL_NAME]))

# Vanilla ODE sampler (no corrector noise, no step overshoot)
configs.sample_diffusion["gamma0"] = 0.0
configs.sample_diffusion["step_scale_eta"] = 1.0
configs.sample_diffusion["noise_scale_lambda"] = 1.0
configs.sample_diffusion["N_step"] = 30

print(f"Model: {MODEL_NAME}")
print(f"Pairformer blocks: {configs.model.pairformer.n_blocks}")

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
import protenix.protenij  # registers all from_torch converters
from protenix.backend import from_torch

print("Converting to JAX/Equinox...")
jax_model = from_torch(torch_model)
print("JAX model ready")

# ── 2. Featurize with real MSA (following mosaic pattern) ──────────────────────

from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.data_pipeline import DataPipeline
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import make_dummy_feature, data_type_transform
from protenix.utils.torch_utils import dict_to_tensor
from protenix.runner import msa_search

sample = {
    "name": "test_b2m",
    "sequences": [
        {"proteinChain": {"sequence": "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA", "count": 1}}
    ],
}

# Run MSA search via mmseqs2 web service
os.makedirs(OUTPUT_DIR, exist_ok=True)
msa_dir = os.path.join(OUTPUT_DIR, "msa")
os.makedirs(msa_dir, exist_ok=True)

print("Running MSA search...")
p = Path(msa_dir) / (sample["name"] + ".json")
p.write_text(json.dumps([sample]))
msa_search.update_infer_json(str(p), msa_dir)
updated = Path(msa_dir) / f"{sample['name']}-add-msa.json"
if updated.exists():
    sample = json.loads(updated.read_text())[0]
    print("MSA search complete")
else:
    print("No MSA update needed (or search unavailable), continuing with dummy MSA")

# Featurize
print("Featurizing...")
sample2feat = SampleDictToFeatures(sample)
features_dict, atom_array, token_array = sample2feat.get_feature_dict()
features_dict["distogram_rep_atom_mask"] = torch.Tensor(
    atom_array.distogram_rep_atom_mask
).long()

# MSA features
entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
has_msa = any(
    "msa" in seq.get("proteinChain", {})
    for seq in sample["sequences"]
)
msa_features = (
    InferenceMSAFeaturizer.make_msa_feature(
        bioassembly=sample["sequences"],
        entity_to_asym_id=entity_to_asym_id,
        token_array=token_array,
        atom_array=atom_array,
    )
    if has_msa
    else {}
)

dummy_feats = ["template"]
if len(msa_features) == 0:
    dummy_feats.append("msa")
else:
    msa_features = dict_to_tensor(msa_features)
    features_dict.update(msa_features)
features_dict = make_dummy_feature(features_dict, dummy_feats=dummy_feats)

features_dict = data_type_transform(features_dict)

N_token = features_dict["token_index"].shape[0]
N_atom = features_dict["atom_to_token_idx"].shape[0]
N_msa = features_dict["msa"].shape[0]
print(f"Featurized: {N_token} tokens, {N_atom} atoms, {N_msa} MSA sequences")

# ── 3. Convert Torch Features to JAX Arrays ────────────────────────────────────

print("Converting features to JAX...")
jax_features = {}
for key, value in features_dict.items():
    if isinstance(value, torch.Tensor):
        jax_features[key] = jnp.array(value.numpy())
    else:
        jax_features[key] = value

# The JAX confidence head indexes by integer array, not boolean mask.
jax_features["atom_rep_atom_idx"] = np.array(
    features_dict["distogram_rep_atom_mask"]
).nonzero()[0]

print(f"atom_rep_atom_idx shape: {jax_features['atom_rep_atom_idx'].shape}")

# ── 4. Run JAX Inference ───────────────────────────────────────────────────────

N_cycle = 4
N_sample = 1

print(f"Running inference (N_cycle={N_cycle}, N_sample={N_sample}, N_steps=default)...")
print(f"  gamma0={jax_model.gamma0}, step_scale_eta={jax_model.step_scale_eta}, N_steps={jax_model.N_steps}")
key = jax.random.PRNGKey(42)
outputs = jax_model(
    input_feature_dict=jax_features,
    N_cycle=N_cycle,
    N_sample=N_sample,
    key=key,
)

print(f"Output coordinates shape: {outputs.coordinates.shape}")
print(f"Coordinate range: [{float(outputs.coordinates.min()):.2f}, {float(outputs.coordinates.max()):.2f}]")

# ── 5. Save PDB Files ──────────────────────────────────────────────────────────

from biotite.structure.io.pdb import PDBFile

os.makedirs(OUTPUT_DIR, exist_ok=True)

coords = np.array(outputs.coordinates)  # (N_sample, N_atom, 3)
for i in range(coords.shape[0]):
    pred_atom_array = copy.deepcopy(atom_array)
    pred_atom_array.coord = coords[i]

    pdb = PDBFile()
    pdb.set_structure(pred_atom_array)
    out_path = os.path.join(OUTPUT_DIR, f"test_b2m_sample_{i}.pdb")
    pdb.write(out_path)
    print(f"Saved {out_path}")

    # Sanity check
    assert not np.any(np.isnan(coords[i])), "NaN coordinates detected!"
    assert not np.all(coords[i] == 0), "All-zero coordinates detected!"

print("\nDone!")
