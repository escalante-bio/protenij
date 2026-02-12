"""Benchmark: PyTorch-path vs Equinox-path model loading."""
import os
os.environ["PROTENIX_DATA_ROOT_DIR"] = os.path.expanduser("~/.protenix")

import time
import jax
import jax.numpy as jnp
import numpy as np
import torch
from ml_collections.config_dict import ConfigDict

from protenix.configs.configs_base import configs as configs_base
from protenix.configs.configs_data import data_configs
from protenix.configs.configs_inference import inference_configs
from protenix.configs.configs_model_type import model_configs
from protenix.config import parse_configs

MODEL_NAME = "protenix_base_default_v1.0.0"
CACHE_DIR = os.path.expanduser("~/.protenix")
EQX_PATH = os.path.join(CACHE_DIR, MODEL_NAME)  # will produce .eqx + .skeleton.pkl


def build_configs():
    cfg = {**configs_base, **{"data": data_configs}, **inference_configs}
    cfg["use_deepspeed_evo_attention"] = False
    cfg = parse_configs(configs=cfg, fill_required_with_null=True)
    cfg.model_name = MODEL_NAME
    cfg.load_checkpoint_dir = CACHE_DIR
    cfg.update(ConfigDict(model_configs[MODEL_NAME]))
    return cfg


def load_torch_path(configs):
    """Load model via PyTorch, return (jax_model, timings_dict)."""
    from protenix.model.protenix import Protenix as TorchProtenix
    import protenix.protenij
    from protenix.backend import from_torch

    # Step 1: torch.load + DDP strip
    checkpoint_path = os.path.join(CACHE_DIR, f"{MODEL_NAME}.pt")
    t0 = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    sample_key = list(checkpoint["model"].keys())[0]
    if sample_key.startswith("module."):
        checkpoint["model"] = {
            k[len("module."):]: v for k, v in checkpoint["model"].items()
        }
    t1 = time.perf_counter()

    # Step 2: construct PyTorch model
    torch_model = TorchProtenix(configs)
    t2 = time.perf_counter()

    # Step 3: load_state_dict
    torch_model.load_state_dict(checkpoint["model"], strict=configs.load_strict)
    torch_model.eval()
    t3 = time.perf_counter()

    # Step 4: from_torch conversion
    jax_model = from_torch(torch_model)
    t4 = time.perf_counter()

    timings = {
        "torch.load + DDP strip": t1 - t0,
        "TorchProtenix(configs)": t2 - t1,
        "load_state_dict": t3 - t2,
        "from_torch()": t4 - t3,
        "total": t4 - t0,
    }
    return jax_model, timings


def load_eqx_path():
    """Load model via Equinox serialization, return (jax_model, timings_dict)."""
    from protenix.backend import load_model

    # Step 1: pickle.load skeleton
    t0 = time.perf_counter()
    import protenix.protenij  # ensure pytree node types are registered
    import pickle
    with open(f"{EQX_PATH}.skeleton.pkl", "rb") as f:
        skeleton = pickle.load(f)
    t1 = time.perf_counter()

    # Step 2: tree_deserialise_leaves
    import equinox as eqx
    jax_model = eqx.tree_deserialise_leaves(f"{EQX_PATH}.eqx", skeleton)
    t2 = time.perf_counter()

    timings = {
        "pickle.load skeleton": t1 - t0,
        "tree_deserialise_leaves": t2 - t1,
        "total": t2 - t0,
    }
    return jax_model, timings


def verify_match(model_a, model_b):
    """Check that two models are bit-exact."""
    leaves_a = jax.tree.leaves(model_a)
    leaves_b = jax.tree.leaves(model_b)
    assert len(leaves_a) == len(leaves_b), f"Leaf count mismatch: {len(leaves_a)} vs {len(leaves_b)}"
    max_err = 0.0
    for a, b in zip(leaves_a, leaves_b):
        if hasattr(a, "shape"):
            err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            max_err = max(max_err, err)
    return max_err


def main():
    configs = build_configs()

    eqx_exists = os.path.exists(f"{EQX_PATH}.eqx") and os.path.exists(
        f"{EQX_PATH}.skeleton.pkl"
    )

    # -- PyTorch path --
    print("=" * 60)
    print("PyTorch path")
    print("=" * 60)
    torch_model, torch_timings = load_torch_path(configs)
    for step, t in torch_timings.items():
        print(f"  {step:30s}  {t:6.2f}s")

    # -- Save if needed --
    if not eqx_exists:
        from protenix.backend import save_model

        print(f"\nSaving Equinox model to {EQX_PATH}.eqx ...")
        t0 = time.perf_counter()
        save_model(torch_model, EQX_PATH)
        t1 = time.perf_counter()
        print(f"  save_model: {t1 - t0:.2f}s")
        eqx_size = os.path.getsize(f"{EQX_PATH}.eqx") / 1e6
        skel_size = os.path.getsize(f"{EQX_PATH}.skeleton.pkl") / 1e6
        print(f"  {EQX_PATH}.eqx: {eqx_size:.1f} MB")
        print(f"  {EQX_PATH}.skeleton.pkl: {skel_size:.3f} MB")

    # -- Equinox path --
    print()
    print("=" * 60)
    print("Equinox path")
    print("=" * 60)
    eqx_model, eqx_timings = load_eqx_path()
    for step, t in eqx_timings.items():
        print(f"  {step:30s}  {t:6.2f}s")

    # -- Verify --
    print()
    max_err = verify_match(torch_model, eqx_model)
    print(f"Max absolute error: {max_err}")

    # -- Summary --
    speedup = torch_timings["total"] / eqx_timings["total"]
    print()
    print("=" * 60)
    print(f"PyTorch total:  {torch_timings['total']:.2f}s")
    print(f"Equinox total:  {eqx_timings['total']:.2f}s")
    print(f"Speedup:        {speedup:.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
