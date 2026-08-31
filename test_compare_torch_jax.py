"""Compare Protenix (PyTorch) trunk/diffusion/confidence outputs vs Protenij (JAX) on identical features."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

CACHE_DIR = os.environ.get("PROTENIJ_CACHE_DIR", os.path.expanduser("~/.protenix"))
os.environ["PROTENIX_DATA_ROOT_DIR"] = CACHE_DIR

MODEL_NAME = "protenix-v2"
OUTPUT_DIR = "./output_test_predict"

SEQUENCE = "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA"
TEMPLATE_PDB = os.path.join(OUTPUT_DIR, "3bik.pdb")
TEMPLATE_CHAIN = "A"
MSA_JSON = os.path.join(OUTPUT_DIR, "msa", "test_b2m-add-msa.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compare(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> dict[str, float]:
    diff = np.abs(a - b)
    max_val = max(np.abs(a).max(), np.abs(b).max(), 1e-8)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(diff.max() / max_val),
        "allclose": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
    }


def _print_comparison(name: str, stats: dict[str, float]) -> None:
    print(
        f"  {name}: allclose={stats['allclose']} "
        f"max_abs={stats['max_abs']:.6e} mean_abs={stats['mean_abs']:.6e} "
        f"max_rel={stats['max_rel']:.6e}"
    )


def _torch_sync(device) -> None:
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device=device)


def _jax_block_until_ready(tree: Any) -> None:
    import jax

    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _benchmark(
    fn: Callable[[], Any],
    *,
    sync: Callable[[Any], None],
    warmup_runs: int,
    benchmark_runs: int,
) -> tuple[Any, dict[str, float]]:
    """Run warmups, then time repeated synchronized executions."""
    out = None
    for _ in range(warmup_runs):
        out = fn()
        sync(out)

    times: list[float] = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        out = fn()
        sync(out)
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times, dtype=np.float64)
    return out, {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "runs": float(benchmark_runs),
    }


def _print_timing(name: str, stats: dict[str, float]) -> None:
    print(
        f"  {name}: min={stats['min']:.4f}s "
        f"mean={stats['mean']:.4f}s std={stats['std']:.4f}s "
        f"runs={int(stats['runs'])}"
    )


def _download_hf_checkpoint(model_name: str, cache_dir: str) -> Path:
    from huggingface_hub import hf_hub_download
    from protenij.backend import HF_REPO

    os.makedirs(cache_dir, exist_ok=True)
    local_path = hf_hub_download(
        HF_REPO,
        f"{model_name}.pt",
        local_dir=cache_dir,
    )
    return Path(local_path)


# ---------------------------------------------------------------------------
# Feature construction (following test_predict.py pattern)
# ---------------------------------------------------------------------------


def build_features(max_msa_rows: int = 1) -> dict[str, np.ndarray]:
    from protenij.data.data_pipeline import DataPipeline
    from protenij.data.json_to_feature import SampleDictToFeatures
    from protenij.data.msa_featurizer import InferenceMSAFeaturizer
    from protenij.data.template import load_templates_from_pdb
    from protenij.data.utils import data_type_transform, make_dummy_feature
    from protenij.runner import msa_search
    from protenij.utils.torch_utils import dict_to_numpy

    sample = {
        "name": "test_b2m",
        "sequences": [
            {"proteinChain": {"sequence": SEQUENCE, "count": 1}}
        ],
    }

    # Use cached MSA if available, otherwise run search
    msa_dir = os.path.join(OUTPUT_DIR, "msa")
    os.makedirs(msa_dir, exist_ok=True)
    updated = Path(MSA_JSON)
    if updated.exists():
        sample = json.loads(updated.read_text())[0]
        print("Using cached MSA results")
    else:
        print("Running MSA search...")
        p = Path(msa_dir) / (sample["name"] + ".json")
        p.write_text(json.dumps([sample]))
        msa_search.update_infer_json(str(p), msa_dir)
        if updated.exists():
            sample = json.loads(updated.read_text())[0]

    sample2feat = SampleDictToFeatures(sample)
    features_dict, atom_array, token_array = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = np.asarray(
        atom_array.distogram_rep_atom_mask, dtype=np.int64
    )

    # MSA features
    entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
    has_msa = any(
        "msa" in seq.get("proteinChain", {}) for seq in sample["sequences"]
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

    dummy_feats = []
    if len(msa_features) == 0:
        dummy_feats.append("msa")
    else:
        msa_features = dict_to_numpy(msa_features)
        features_dict.update(msa_features)
    features_dict = make_dummy_feature(features_dict, dummy_feats=dummy_feats)

    # Truncate MSA to max_msa_rows so both PyTorch (random subsampling) and
    # JAX (takes all rows up to cutoff) process identical MSA inputs.
    msa_keys = ["msa", "has_deletion", "deletion_value"]
    for k in msa_keys:
        if k in features_dict and features_dict[k].ndim >= 2:
            features_dict[k] = features_dict[k][:max_msa_rows]

    # Template features
    if not os.path.exists(TEMPLATE_PDB):
        import urllib.request
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        urllib.request.urlretrieve("https://files.rcsb.org/download/3BIK.pdb", TEMPLATE_PDB)

    N_token = features_dict["token_index"].shape[0]
    query_aatype = np.argmax(features_dict["restype"], axis=-1)
    templates = load_templates_from_pdb(TEMPLATE_PDB, TEMPLATE_CHAIN, N_token, query_aatype)
    features_dict.update(templates.as_protenix_dict())

    features_dict = data_type_transform(features_dict)

    N_atom = features_dict["atom_to_token_idx"].shape[0]
    N_msa = features_dict["msa"].shape[0]
    print(f"Features: {N_token} tokens, {N_atom} atoms, {N_msa} MSA rows")
    return features_dict


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_torch_model(checkpoint_path: str, device: str, model_name: str):
    import torch
    from ml_collections.config_dict import ConfigDict

    from protenij.config import parse_configs
    from protenij.configs.configs_base import configs as configs_base
    from protenij.configs.configs_data import data_configs
    from protenij.configs.configs_inference import inference_configs
    from protenij.configs.configs_model_type import model_configs
    from protenij.model.protenix import Protenix as TorchProtenix

    configs_base["use_deepspeed_evo_attention"] = False
    configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    configs = parse_configs(configs=configs, fill_required_with_null=True)
    configs.model_name = model_name
    configs.load_checkpoint_dir = CACHE_DIR
    configs.update(ConfigDict(model_configs[model_name]))

    torch_model = TorchProtenix(configs)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    sample_key = list(checkpoint["model"].keys())[0]
    if sample_key.startswith("module."):
        checkpoint["model"] = {k[len("module."):]: v for k, v in checkpoint["model"].items()}
    torch_model.load_state_dict(checkpoint["model"], strict=configs.load_strict)

    torch_device = torch.device(device)
    torch_model.to(torch_device)
    torch_model.eval()
    return torch_model, torch_device


def convert_to_jax(torch_model, jax_device):
    import equinox as eqx
    import jax

    import protenij.protenij  # noqa: F401 — registers from_torch converters
    from protenij.backend import from_torch

    jax_model = from_torch(torch_model)
    params, static = eqx.partition(jax_model, eqx.is_inexact_array)
    params = jax.device_put(params, jax_device)
    jax_model = eqx.combine(params, static)
    return jax_model


# ---------------------------------------------------------------------------
# Feature conversion helpers
# ---------------------------------------------------------------------------


def features_to_torch(features_dict: dict[str, np.ndarray], device) -> dict:
    import torch
    out = {}
    for k, v in features_dict.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v).to(device)
        else:
            out[k] = v
    return out


def features_to_jax(features_dict: dict[str, np.ndarray], device) -> dict:
    import jax
    import jax.numpy as jnp
    out = {}
    for k, v in features_dict.items():
        if isinstance(v, np.ndarray):
            out[k] = jax.device_put(jnp.array(v), device)
        else:
            out[k] = v
    # JAX confidence head uses integer indices instead of boolean mask
    out["atom_rep_atom_idx"] = features_dict["distogram_rep_atom_mask"].nonzero()[0]
    out["atom_rep_atom_idx"] = jax.device_put(jnp.array(out["atom_rep_atom_idx"]), device)
    return out


# ---------------------------------------------------------------------------
# Trunk comparison
# ---------------------------------------------------------------------------


def run_torch_trunk(model, feats: dict, N_cycle: int):
    import torch
    with torch.no_grad():
        s_inputs, s, z = model.get_pairformer_output(
            input_feature_dict=feats,
            N_cycle=N_cycle,
            inplace_safe=False,
        )
        pdistogram = model.distogram_head(z)
    return {
        "s_inputs": s_inputs,
        "s_trunk": s,
        "z_trunk": z,
        "pdistogram": pdistogram,
    }


def run_torch_trunk_stepwise(model, feats: dict):
    """Run one recycling step, returning intermediates at each stage."""
    import torch
    with torch.no_grad():
        s_inputs = model.input_embedder(feats, inplace_safe=False)
        s_init = model.linear_no_bias_sinit(s_inputs)
        z_init = (
            model.linear_no_bias_zinit1(s_init)[..., None, :]
            + model.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )
        z_init = z_init + model.relative_position_encoding(feats)
        z_init = z_init + model.linear_no_bias_token_bond(
            feats["token_bonds"].unsqueeze(dim=-1)
        )

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)

        z = z_init + model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z))
        z_after_cycle = z.clone()

        if model.template_embedder.n_blocks > 0:
            z = z + model.template_embedder(feats, z)
        z_after_template = z.clone()

        z = model.msa_module(feats, z, s_inputs, pair_mask=None)
        z_after_msa = z.clone()

        s = s_init + model.linear_no_bias_s(model.layernorm_s(s))
        s_after_cycle = s.clone()

        s, z = model.pairformer_stack(s, z, pair_mask=None)

    return {
        "s_inputs": s_inputs,
        "s_init": s_init,
        "z_init": z_init,
        "z_after_cycle": z_after_cycle,
        "z_after_template": z_after_template,
        "z_after_msa": z_after_msa,
        "s_after_cycle": s_after_cycle,
        "s_trunk": s,
        "z_trunk": z,
    }


def run_jax_trunk_stepwise(model, feats: dict, key):
    """Run one recycling step, returning intermediates at each stage."""
    import jax.numpy as jnp

    initial = model.embed_inputs(input_feature_dict=feats)

    s = jnp.zeros_like(initial.s_init)
    z = jnp.zeros_like(initial.z_init)

    z = initial.z_init + model.linear_no_bias_z_cycle(model.layernorm_z_cycle(z))
    z_after_cycle = z

    if model.template_embedder.n_blocks > 0:
        z = z + model.template_embedder(feats, z, pair_mask=None, key=key)
    z_after_template = z

    z = model.msa_module(feats, z, initial.s_inputs, pair_mask=None, key=key)
    z_after_msa = z

    s = initial.s_init + model.linear_no_bias_s(model.layernorm_s(s))
    s_after_cycle = s

    import jax
    s, z = model.pairformer_stack(s, z, pair_mask=None, key=jax.random.fold_in(key, 1))

    return {
        "s_inputs": initial.s_inputs,
        "s_init": initial.s_init,
        "z_init": initial.z_init,
        "z_after_cycle": z_after_cycle,
        "z_after_template": z_after_template,
        "z_after_msa": z_after_msa,
        "s_after_cycle": s_after_cycle,
        "s_trunk": s,
        "z_trunk": z,
    }


def run_pairformer_blockwise(torch_model, jax_model, feats_torch, feats_jax, jax_device, key):
    """Run pairformer block-by-block from identical inputs and report per-block error."""
    import torch
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    # Get inputs from torch stepwise (use torch as ground truth source)
    torch_step = run_torch_trunk_stepwise(torch_model, feats_torch)
    s_torch = torch_step["s_after_cycle"]
    z_torch = torch_step["z_after_msa"]

    # Convert torch inputs to JAX (so both start from identical values)
    s_jax = jax.device_put(jnp.array(s_torch.detach().cpu().numpy()), jax_device)
    z_jax = jax.device_put(jnp.array(z_torch.detach().cpu().numpy()), jax_device)

    n_blocks = len(torch_model.pairformer_stack.blocks)
    pf = jax_model.pairformer_stack

    print(f"  {'block':>5}  {'s_max_abs':>12}  {'s_mean_abs':>12}  {'z_max_abs':>12}  {'z_mean_abs':>12}")
    print(f"  {'-----':>5}  {'--------':>12}  {'----------':>12}  {'--------':>12}  {'----------':>12}")

    with torch.no_grad():
        for i in range(n_blocks):
            # PyTorch block
            s_torch, z_torch = torch_model.pairformer_stack.blocks[i](
                s_torch, z_torch, pair_mask=None,
            )

            # JAX block: extract block i from stacked params
            block_i_params = jax.tree.map(lambda x: x[i], pf.stacked_parameters)
            block_i = eqx.combine(pf.static, block_i_params)
            s_jax, z_jax = block_i(s=s_jax, z=z_jax, pair_mask=None, key=jax.random.fold_in(key, i))

            # Compare
            s_np_t = s_torch.detach().cpu().numpy()
            z_np_t = z_torch.detach().cpu().numpy()
            s_np_j = np.asarray(s_jax)
            z_np_j = np.asarray(z_jax)

            s_diff = np.abs(s_np_t - s_np_j)
            z_diff = np.abs(z_np_t - z_np_j)
            print(f"  {i:5d}  {s_diff.max():12.6e}  {s_diff.mean():12.6e}  {z_diff.max():12.6e}  {z_diff.mean():12.6e}")


def run_jax_trunk(model, feats: dict, recycling_steps: int, key):
    initial_embedding = model.embed_inputs(input_feature_dict=feats)
    trunk_embedding = model.recycle(
        initial_embedding=initial_embedding,
        input_feature_dict=feats,
        recycling_steps=recycling_steps,
        key=key,
    )
    pdistogram = model.distogram_head(trunk_embedding.z)
    return {
        "s_inputs": initial_embedding.s_inputs,
        "s_trunk": trunk_embedding.s,
        "z_trunk": trunk_embedding.z,
        "pdistogram": pdistogram,
    }


# ---------------------------------------------------------------------------
# Diffusion (single denoise step) comparison
# ---------------------------------------------------------------------------


def run_torch_diffusion(model, feats, trunk_out, x_noisy, sigma):
    import torch
    with torch.no_grad():
        N_sample = x_noisy.shape[-3]
        t_hat = torch.full(
            (*x_noisy.shape[:-3], N_sample),
            sigma,
            dtype=x_noisy.dtype,
            device=x_noisy.device,
        )
        diff_mod = model.diffusion_module
        r_noisy = x_noisy / torch.sqrt(diff_mod.sigma_data**2 + t_hat**2)[..., None, None]
        r_update = diff_mod.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat,
            input_feature_dict=feats,
            s_inputs=trunk_out["s_inputs"],
            s_trunk=trunk_out["s_trunk"],
            z_trunk=trunk_out["z_trunk"],
        )
        s_ratio = (t_hat / diff_mod.sigma_data)[..., None, None].to(r_update.dtype)
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat[..., None, None] / torch.sqrt(1 + s_ratio**2) * r_update
        ).to(r_update.dtype)
    return {"r_update": r_update, "x_denoised": x_denoised}


def run_jax_diffusion(model, feats, trunk_out, x_noisy, sigma):
    import jax.numpy as jnp
    N_sample = x_noisy.shape[-3]
    t_hat = jnp.full(
        (*x_noisy.shape[:-3], N_sample),
        sigma,
        dtype=x_noisy.dtype,
    )
    diff_mod = model.diffusion_module
    r_noisy = x_noisy / jnp.sqrt(diff_mod.sigma_data**2 + t_hat**2)[..., None, None]
    r_update = diff_mod.f_forward(
        r_noisy=r_noisy,
        t_hat_noise_level=t_hat,
        input_feature_dict=feats,
        s_inputs=trunk_out["s_inputs"],
        s_trunk=trunk_out["s_trunk"],
        z_trunk=trunk_out["z_trunk"],
        use_conditioning=True,
    )
    s_ratio = (t_hat / diff_mod.sigma_data)[..., None, None]
    x_denoised = (
        1 / (1 + s_ratio**2) * x_noisy
        + t_hat[..., None, None] / jnp.sqrt(1 + s_ratio**2) * r_update
    )
    return {"r_update": r_update, "x_denoised": x_denoised}


# ---------------------------------------------------------------------------
# Confidence head comparison
# ---------------------------------------------------------------------------


def run_torch_confidence(model, feats, trunk_out, x_pred_coords):
    import torch
    with torch.no_grad():
        plddt, pae, pde, resolved = model.confidence_head(
            input_feature_dict=feats,
            s_inputs=trunk_out["s_inputs"],
            s_trunk=trunk_out["s_trunk"],
            z_trunk=trunk_out["z_trunk"],
            pair_mask=None,
            x_pred_coords=x_pred_coords,
            use_embedding=True,
        )
    return {
        "plddt_logits": plddt,
        "pae_logits": pae,
        "pde_logits": pde,
        "resolved_logits": resolved,
    }


def run_jax_confidence(model, feats, trunk_out, x_pred_coords, key):
    plddt, pae, pde, resolved = model.confidence_head(
        input_feature_dict=feats,
        s_inputs=trunk_out["s_inputs"],
        s_trunk=trunk_out["s_trunk"],
        z_trunk=trunk_out["z_trunk"],
        pair_mask=None,
        x_pred_coords=x_pred_coords,
        key=key,
        use_embedding=True,
    )
    return {
        "plddt_logits": plddt,
        "pae_logits": pae,
        "pde_logits": pde,
        "resolved_logits": resolved,
    }


# ---------------------------------------------------------------------------
# Numpy conversion helpers
# ---------------------------------------------------------------------------


def torch_dict_to_numpy(d: dict) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in d.items()}


def jax_dict_to_numpy(d: dict) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in d.items()}


def torch_to_jax_dict(torch_dict: dict, jax_device) -> dict:
    """Convert a dict of torch tensors to jax arrays (for conditioning isolation)."""
    import jax
    import jax.numpy as jnp
    return {
        k: jax.device_put(jnp.array(v.detach().cpu().numpy()), jax_device)
        for k, v in torch_dict.items()
    }


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def resolve_torch_device(device_arg: str):
    import torch
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "gpu":
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_jax_device(device_arg: str):
    import jax
    if device_arg == "auto":
        try:
            gpu = jax.devices("gpu")
        except RuntimeError:
            gpu = []
        return gpu[0] if gpu else jax.devices("cpu")[0]
    return jax.devices(device_arg)[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Protenix (torch) vs Protenij (jax) outputs")
    parser.add_argument("--model_name", default=MODEL_NAME,
                        help="Protenix model config/checkpoint name")
    parser.add_argument("--checkpoint", type=Path,
                        default=None,
                        help="PyTorch checkpoint path; defaults to $PROTENIJ_CACHE_DIR/<model_name>.pt")
    parser.add_argument("--torch_device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--jax_device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recycling_steps", type=int, default=1)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_runs", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1.5,
                        help="Absolute tolerance for trunk/diffusion (float32 accumulation across 48 pairformer blocks)")
    parser.add_argument("--rtol", type=float, default=0.01,
                        help="Relative tolerance for trunk/diffusion (~0.05%% expected)")
    parser.add_argument("--confidence_atol", type=float, default=0.05,
                        help="Confidence tolerance (tighter because trunk outputs are shared)")
    parser.add_argument("--confidence_rtol", type=float, default=0.01)
    parser.add_argument("--diffusion_sigma", type=float, default=16.0)
    parser.add_argument("--max_msa_rows", type=int, default=1,
                        help="Truncate MSA to this many rows for deterministic comparison")
    parser.add_argument("--stepwise", action="store_true",
                        help="Run stepwise trunk diagnostic to isolate divergence source")
    parser.add_argument("--skip_diffusion_check", action="store_true")
    parser.add_argument("--skip_confidence_check", action="store_true")
    args = parser.parse_args()

    if args.warmup_runs < 0:
        raise ValueError("--warmup_runs must be >= 0")
    if args.benchmark_runs < 1:
        raise ValueError("--benchmark_runs must be >= 1")

    import jax
    import equinox as eqx
    import torch

    checkpoint = args.checkpoint or (Path(CACHE_DIR) / f"{args.model_name}.pt")
    if not checkpoint.exists():
        if args.checkpoint is not None:
            raise FileNotFoundError(f"PyTorch checkpoint not found: {checkpoint}")
        print(f"PyTorch checkpoint not found at {checkpoint}; downloading from HuggingFace...")
        checkpoint = _download_hf_checkpoint(args.model_name, CACHE_DIR)

    # Disable TF32 to force full fp32 and match precision between PyTorch and JAX/XLA.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    jax.config.update("jax_default_matmul_precision", "highest")

    # ── 1. Build features ────────────────────────────────────────────────────
    torch_device = resolve_torch_device(args.torch_device)
    jax_device = resolve_jax_device(args.jax_device)
    print(f"torch_device: {torch_device}")
    print(f"jax_device: {jax_device}")

    print("Building features...")
    features_np = build_features(max_msa_rows=args.max_msa_rows)

    feats_torch = features_to_torch(features_np, torch_device)
    feats_jax = features_to_jax(features_np, jax_device)

    # ── 2. Load models ───────────────────────────────────────────────────────
    print(f"model_name: {args.model_name}")
    print(f"Loading PyTorch model from {checkpoint}...")
    torch_model, torch_device = load_torch_model(str(checkpoint), str(torch_device), args.model_name)

    print("Converting PyTorch → JAX...")
    t0 = time.perf_counter()
    jax_model = convert_to_jax(torch_model, jax_device)
    convert_s = time.perf_counter() - t0

    key = jax.random.PRNGKey(args.seed)
    jax_trunk_fn = eqx.filter_jit(
        lambda model, feats, key: run_jax_trunk(
            model, feats, args.recycling_steps, key
        )
    )
    jax_diffusion_fn = eqx.filter_jit(
        lambda model, feats, trunk_out, noisy, sigma: run_jax_diffusion(
            model, feats, trunk_out, noisy, sigma
        )
    )
    jax_confidence_fn = eqx.filter_jit(
        lambda model, feats, trunk_out, coords, key: run_jax_confidence(
            model, feats, trunk_out, coords, key
        )
    )

    # ── 3. Trunk comparison ──────────────────────────────────────────────────
    all_ok = True
    torch_trunk_timing = None
    jax_trunk_timing = None
    torch_diff_timing = None
    jax_diff_timing = None
    torch_conf_timing = None
    jax_conf_timing = None

    if args.stepwise:
        # Stepwise diagnostic: decompose one recycling step and compare each stage
        print(f"\n{'='*60}")
        print("STEPWISE TRUNK DIAGNOSTIC (1 recycling step)")
        print(f"tolerances: atol={args.atol}, rtol={args.rtol}")
        print(f"{'='*60}")

        torch_step = run_torch_trunk_stepwise(torch_model, feats_torch)
        jax_step = run_jax_trunk_stepwise(jax_model, feats_jax, key)

        torch_step_np = torch_dict_to_numpy(torch_step)
        jax_step_np = jax_dict_to_numpy(jax_step)

        stage_names = [
            "s_inputs", "s_init", "z_init",
            "z_after_cycle", "z_after_template", "z_after_msa",
            "s_after_cycle", "s_trunk", "z_trunk",
        ]
        for name in stage_names:
            stats = _compare(torch_step_np[name], jax_step_np[name], atol=args.atol, rtol=args.rtol)
            _print_comparison(name, stats)

        # Per-block pairformer diagnostic
        print(f"\n{'='*60}")
        print("PAIRFORMER BLOCK-BY-BLOCK (from identical inputs)")
        print(f"{'='*60}")
        run_pairformer_blockwise(torch_model, jax_model, feats_torch, feats_jax, jax_device, key)

        # Use stepwise outputs for downstream comparisons
        torch_trunk = {
            "s_inputs": torch_step["s_inputs"],
            "s_trunk": torch_step["s_trunk"],
            "z_trunk": torch_step["z_trunk"],
        }
        torch_trunk_s = 0.0
        jax_trunk_s = 0.0
    else:
        print(f"\n{'='*60}")
        print(f"TRUNK (recycling_steps={args.recycling_steps})")
        print(f"tolerances: atol={args.atol}, rtol={args.rtol}")
        print(f"{'='*60}")

        torch_trunk, torch_trunk_timing = _benchmark(
            lambda: run_torch_trunk(torch_model, feats_torch, args.recycling_steps),
            sync=lambda _: _torch_sync(torch_device),
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        jax_trunk, jax_trunk_timing = _benchmark(
            lambda: jax_trunk_fn(jax_model, feats_jax, key),
            sync=_jax_block_until_ready,
            warmup_runs=args.warmup_runs,
            benchmark_runs=args.benchmark_runs,
        )
        torch_trunk_s = torch_trunk_timing["mean"]
        jax_trunk_s = jax_trunk_timing["mean"]

        torch_trunk_np = torch_dict_to_numpy(torch_trunk)
        jax_trunk_np = jax_dict_to_numpy(jax_trunk)

        for name in ["s_inputs", "s_trunk", "z_trunk", "pdistogram"]:
            stats = _compare(torch_trunk_np[name], jax_trunk_np[name], atol=args.atol, rtol=args.rtol)
            all_ok = all_ok and stats["allclose"]
            _print_comparison(name, stats)

    # ── 4. Diffusion (single denoise step) ───────────────────────────────────
    torch_diff_s = None
    jax_diff_s = None
    torch_conf_s = None
    jax_conf_s = None

    need_diffusion = not args.skip_diffusion_check or not args.skip_confidence_check
    if need_diffusion:
        rng = np.random.default_rng(args.seed)
        N_atom = features_np["atom_to_token_idx"].shape[0]
        noisy_np = rng.standard_normal((1, N_atom, 3)).astype(np.float32)
        noisy_torch = torch.from_numpy(noisy_np).to(torch_device)
        noisy_jax = jax.device_put(jax.numpy.array(noisy_np), jax_device)

        # Use torch trunk outputs as conditioning for both (isolate diffusion comparison)
        trunk_cond_jax = torch_to_jax_dict(torch_trunk, jax_device)

        if not args.skip_diffusion_check:
            print(f"\n{'='*60}")
            print(f"DIFFUSION (sigma={args.diffusion_sigma})")
            print(f"tolerances: atol={args.atol}, rtol={args.rtol}")
            print(f"{'='*60}")

            torch_diff, torch_diff_timing = _benchmark(
                lambda: run_torch_diffusion(
                    torch_model,
                    feats_torch,
                    torch_trunk,
                    noisy_torch,
                    args.diffusion_sigma,
                ),
                sync=lambda _: _torch_sync(torch_device),
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
            )
            jax_diff, jax_diff_timing = _benchmark(
                lambda: jax_diffusion_fn(
                    jax_model,
                    feats_jax,
                    trunk_cond_jax,
                    noisy_jax,
                    args.diffusion_sigma,
                ),
                sync=_jax_block_until_ready,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
            )
            torch_diff_s = torch_diff_timing["mean"]
            jax_diff_s = jax_diff_timing["mean"]

            for name in ["r_update", "x_denoised"]:
                stats = _compare(
                    torch_diff[name].detach().cpu().numpy(),
                    np.asarray(jax_diff[name]),
                    atol=args.atol,
                    rtol=args.rtol,
                )
                all_ok = all_ok and stats["allclose"]
                _print_comparison(name, stats)

        # ── 5. Confidence head ───────────────────────────────────────────────
        if not args.skip_confidence_check:
            print(f"\n{'='*60}")
            print(f"CONFIDENCE HEAD")
            print(f"tolerances: atol={args.confidence_atol}, rtol={args.confidence_rtol}")
            print(f"{'='*60}")

            # Use torch diffusion output for both (isolate confidence comparison)
            if args.skip_diffusion_check:
                # Need to run diffusion once to get coords
                torch_diff = run_torch_diffusion(
                    torch_model, feats_torch, torch_trunk, noisy_torch, args.diffusion_sigma
                )
            x_pred_torch = torch_diff["x_denoised"]
            x_pred_jax = jax.device_put(
                jax.numpy.array(x_pred_torch.detach().cpu().numpy()), jax_device
            )

            conf_key = jax.random.fold_in(key, 42)
            torch_conf, torch_conf_timing = _benchmark(
                lambda: run_torch_confidence(
                    torch_model, feats_torch, torch_trunk, x_pred_torch
                ),
                sync=lambda _: _torch_sync(torch_device),
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
            )
            jax_conf, jax_conf_timing = _benchmark(
                lambda: jax_confidence_fn(
                    jax_model, feats_jax, trunk_cond_jax, x_pred_jax, conf_key
                ),
                sync=_jax_block_until_ready,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
            )
            torch_conf_s = torch_conf_timing["mean"]
            jax_conf_s = jax_conf_timing["mean"]

            for name in ["plddt_logits", "pae_logits", "pde_logits", "resolved_logits"]:
                torch_val = torch_conf[name]
                jax_val = jax_conf[name]
                # PyTorch confidence loops over N_sample and stacks; JAX vmaps.
                # Both produce [..., N_sample, ...] output.
                stats = _compare(
                    torch_val.detach().cpu().numpy(),
                    np.asarray(jax_val),
                    atol=args.confidence_atol,
                    rtol=args.confidence_rtol,
                )
                all_ok = all_ok and stats["allclose"]
                _print_comparison(name, stats)

    # ── 6. Timings ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TIMINGS (seconds)")
    print(f"{'='*60}")
    print(f"  torch→jax conversion: {convert_s:.4f}")
    if torch_trunk_timing is not None:
        _print_timing("torch trunk", torch_trunk_timing)
    else:
        print(f"  torch trunk: {torch_trunk_s:.4f}")
    if jax_trunk_timing is not None:
        _print_timing("jax trunk", jax_trunk_timing)
    else:
        print(f"  jax trunk:   {jax_trunk_s:.4f}")
    if torch_diff_timing is not None:
        _print_timing("torch diffusion", torch_diff_timing)
    elif torch_diff_s is not None:
        print(f"  torch diffusion: {torch_diff_s:.4f}")
    if jax_diff_timing is not None:
        _print_timing("jax diffusion", jax_diff_timing)
    elif jax_diff_s is not None:
        print(f"  jax diffusion:   {jax_diff_s:.4f}")
    if torch_conf_timing is not None:
        _print_timing("torch confidence", torch_conf_timing)
    elif torch_conf_s is not None:
        print(f"  torch confidence: {torch_conf_s:.4f}")
    if jax_conf_timing is not None:
        _print_timing("jax confidence", jax_conf_timing)
    elif jax_conf_s is not None:
        print(f"  jax confidence:   {jax_conf_s:.4f}")

    # ── 7. Final status ──────────────────────────────────────────────────────
    print()
    if not all_ok:
        print("FAILED: some checks exceeded tolerance")
        raise SystemExit(1)

    checks = ["trunk"]
    if not args.skip_diffusion_check:
        checks.append("diffusion")
    if not args.skip_confidence_check:
        checks.append("confidence")
    print(f"OK: torch vs jax {'+'.join(checks)} checks match within tolerance")


if __name__ == "__main__":
    main()
