"""Test loading a Protenix checkpoint and converting to JAX/Equinox."""
import os
os.environ["PROTENIX_DATA_ROOT_DIR"] = os.path.expanduser("~/.protenix")

import torch
from ml_collections.config_dict import ConfigDict

from protenix.configs.configs_base import configs as configs_base
from protenix.configs.configs_data import data_configs
from protenix.configs.configs_inference import inference_configs
from protenix.configs.configs_model_type import model_configs
from protenix.config import parse_configs

MODEL_NAME = "protenix_base_default_v1.0.0"
CACHE_DIR = os.path.expanduser("~/.protenix")

# 1. Build configs
configs_base["use_deepspeed_evo_attention"] = False
configs = {**configs_base, **{"data": data_configs}, **inference_configs}
configs = parse_configs(configs=configs, fill_required_with_null=True)
configs.model_name = MODEL_NAME
configs.load_checkpoint_dir = CACHE_DIR

# Apply model-specific overrides
model_specific = ConfigDict(model_configs[MODEL_NAME])
configs.update(model_specific)

print(f"Model: {MODEL_NAME}")
print(f"Pairformer blocks: {configs.model.pairformer.n_blocks}")
print(f"Diffusion transformer blocks: {configs.model.diffusion_module.transformer.n_blocks}")

# 2. Create PyTorch model
from protenix.model.protenix import Protenix as TorchProtenix
torch_model = TorchProtenix(configs)
print(f"PyTorch model created")
n_params = sum(p.numel() for p in torch_model.parameters())
print(f"Parameters: {n_params / 1e6:.2f}M")

# 3. Load checkpoint
checkpoint_path = os.path.join(CACHE_DIR, f"{MODEL_NAME}.pt")
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Handle DDP prefix
sample_key = list(checkpoint["model"].keys())[0]
if sample_key.startswith("module."):
    checkpoint["model"] = {k[len("module."):]: v for k, v in checkpoint["model"].items()}

torch_model.load_state_dict(checkpoint["model"], strict=configs.load_strict)
torch_model.eval()
print("Checkpoint loaded into PyTorch model")

# 4. Convert to JAX/Equinox
import protenix.protenij  # registers all from_torch converters
from protenix.backend import from_torch

print("Converting to JAX/Equinox...")
jax_model = from_torch(torch_model)
print(f"JAX model type: {type(jax_model)}")

# 5. Count JAX parameters
import jax
leaves = jax.tree.leaves(jax_model)
n_jax_params = sum(x.size for x in leaves if hasattr(x, 'size'))
print(f"JAX parameters: {n_jax_params / 1e6:.2f}M")

print("\nSuccess! Model loaded and converted to JAX/Equinox.")
