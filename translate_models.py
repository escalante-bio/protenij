"""Translate PyTorch checkpoints to Equinox .eqx + .skeleton.pkl files."""
import os
os.environ["PROTENIX_DATA_ROOT_DIR"] = os.path.expanduser("~/.protenix")

import time
import torch
import jax
import numpy as np
from ml_collections.config_dict import ConfigDict

from protenix.configs.configs_base import configs as configs_base
from protenix.configs.configs_data import data_configs
from protenix.configs.configs_inference import inference_configs
from protenix.configs.configs_model_type import model_configs
from protenix.config import parse_configs

CACHE_DIR = os.path.expanduser("~/.protenix")

MODELS = [
    "protenix_mini_default_v0.5.0",
    "protenix_tiny_default_v0.5.0",
    "protenix_base_default_v1.0.0",
    "protenix_base_20250630_v1.0.0",
]


def translate(model_name):
    eqx_path = os.path.join(CACHE_DIR, model_name)
    if os.path.exists(f"{eqx_path}.eqx") and os.path.exists(f"{eqx_path}.skeleton.pkl"):
        print(f"  already exists, skipping")
        return

    checkpoint_path = os.path.join(CACHE_DIR, f"{model_name}.pt")
    if not os.path.exists(checkpoint_path):
        print(f"  checkpoint not found at {checkpoint_path}, skipping")
        return

    # Build configs (fresh copy each time since configs_base is mutable)
    from protenix.configs.configs_base import configs as cb
    cfg = {**cb, **{"data": data_configs}, **inference_configs}
    cfg["use_deepspeed_evo_attention"] = False
    cfg = parse_configs(configs=cfg, fill_required_with_null=True)
    cfg.model_name = model_name
    cfg.load_checkpoint_dir = CACHE_DIR
    cfg.update(ConfigDict(model_configs[model_name]))

    # PyTorch model
    from protenix.model.protenix import Protenix as TorchProtenix
    t0 = time.perf_counter()
    torch_model = TorchProtenix(cfg)
    t1 = time.perf_counter()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    sample_key = list(checkpoint["model"].keys())[0]
    if sample_key.startswith("module."):
        checkpoint["model"] = {k[len("module."):]: v for k, v in checkpoint["model"].items()}
    torch_model.load_state_dict(checkpoint["model"], strict=cfg.load_strict)
    torch_model.eval()
    t2 = time.perf_counter()

    # Convert to JAX
    import protenix.protenij
    from protenix.backend import from_torch, save_model
    jax_model = from_torch(torch_model)
    t3 = time.perf_counter()

    # Save
    save_model(jax_model, eqx_path)
    t4 = time.perf_counter()

    # Verify round-trip
    from protenix.backend import load_model
    jax_model2 = load_model(eqx_path)
    leaves_a = jax.tree.leaves(jax_model)
    leaves_b = jax.tree.leaves(jax_model2)
    max_err = max(
        (float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
         for a, b in zip(leaves_a, leaves_b) if hasattr(a, "shape")),
        default=0.0,
    )

    eqx_mb = os.path.getsize(f"{eqx_path}.eqx") / 1e6
    print(f"  construct={t1-t0:.1f}s  load={t2-t1:.1f}s  convert={t3-t2:.1f}s  save={t4-t3:.1f}s")
    print(f"  {eqx_mb:.0f} MB  max_err={max_err}")


def main():
    for name in MODELS:
        print(f"\n{name}")
        translate(name)
    print("\ndone")


if __name__ == "__main__":
    main()
