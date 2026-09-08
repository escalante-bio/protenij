"""Opt-in checkpoint validation on trusted local feature pickles.

Run from the repository with its dependencies available, for example:
  PYTHONPATH=. python scripts/validate_atom_padding.py --features native1.pkl native2.pkl

This is a validation harness, not a replacement prediction pipeline. It uses
real weights and the public model stages with controlled noise on valid atoms.
Keep GPU preallocation enabled and set an explicit memory budget if needed.
Never load feature pickles from an untrusted source.
"""

import argparse
import json
import pickle
import time

import equinox as eqx
import jax
import numpy as np

from protenix.atom_padding import pad_atom_features
from protenix.backend import load_model
from protenix.protenij import Outputs, sample_diffusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--model", default="protenix-v2")
    parser.add_argument("--padding-multiple", type=int, default=256)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", default="atom-padding-validation.json")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    model = jax.tree.map(
        lambda x: jax.device_put(x) if eqx.is_array(x) else x, load_model(args.model)
    )
    features = []
    for path in args.features:
        with open(path, "rb") as handle:
            features.append(pickle.load(handle))
    traces = []

    @eqx.filter_jit
    def predict(model, features, initial_noise, step_noise):
        traces.append(1)
        initial = model.embed_inputs(input_feature_dict=features)
        trunk = model.recycle(
            initial_embedding=initial,
            input_feature_dict=features,
            recycling_steps=args.cycles,
            key=jax.random.key(0),
        )
        coordinates = sample_diffusion(
            denoise_net=model.diffusion_module,
            input_feature_dict=features,
            s_inputs=initial.s_inputs,
            s_trunk=trunk.s,
            z_trunk=trunk.z,
            noise_schedule=model.inference_noise_scheduler(args.steps),
            N_sample=1,
            gamma0=model.gamma0,
            gamma_min=model.gamma_min,
            noise_scale_lambda=model.noise_scale_lambda,
            step_scale_eta=model.step_scale_eta,
            key=jax.random.key(1),
            initial_noise=initial_noise,
            step_noise=step_noise,
        )
        confidence = model.confidence_metrics(
            initial_embedding=initial,
            trunk_embedding=trunk,
            input_feature_dict=features,
            coordinates=coordinates,
            key=jax.random.key(2),
        )
        return (
            initial,
            trunk,
            Outputs(
                coordinates,
                confidence,
                model.distogram_head(trunk.z),
                features.get("atom_pad_mask"),
            ),
        )

    results = []
    for index, native in enumerate(features):
        padded = pad_atom_features(native, padding_multiple=args.padding_multiple)
        n = native["atom_to_token_idx"].shape[0]
        bucket = padded["atom_to_token_idx"].shape[0]
        rng = np.random.default_rng(100 + index)
        initial = rng.normal(size=(1, bucket, 3)).astype(np.float32)
        step = rng.normal(size=(args.steps, 1, bucket, 3)).astype(np.float32)
        # Poison padding so a missed mask cannot quietly pass on zero rows.
        initial[:, n:] = np.nan
        step[:, :, n:] = np.nan
        outputs = []
        for label, f, ni, ns in (
            ("native", native, initial[:, :n], step[:, :, :n]),
            ("padded", padded, initial, step),
        ):
            f, ni, ns = jax.device_put((f, ni, ns))
            trace_count = len(traces)
            start = time.perf_counter()
            lowered = predict.lower(model, f, ni, ns)
            lowered_s = time.perf_counter() - start
            start = time.perf_counter()
            compiled = lowered.compile()
            compile_s = time.perf_counter() - start
            start = time.perf_counter()
            value = compiled(model, f, ni, ns)
            jax.block_until_ready(value)
            execution_s = time.perf_counter() - start
            start = time.perf_counter()
            jax.block_until_ready(compiled(model, f, ni, ns))
            warm_s = time.perf_counter() - start
            timing = {
                "fixture": index,
                "variant": label,
                "atoms": f["atom_to_token_idx"].shape[0],
                "tokens": f["residue_index"].shape[0],
                "lower_s": lowered_s,
                "compile_s": compile_s,
                "first_execution_s": execution_s,
                "warm_execution_s": warm_s,
                "new_traces": len(traces) - trace_count,
            }
            print(json.dumps(timing), flush=True)
            results.append(timing)
            initial_out, trunk_out, output = value
            outputs.append(jax.device_get((initial_out, trunk_out, output.unpad())))
        native_leaves = jax.tree.leaves(outputs[0])
        padded_leaves = jax.tree.leaves(outputs[1])
        differences = []
        parity_passed = True
        for a, b in zip(native_leaves, padded_leaves, strict=True):
            if a.shape != b.shape or not np.isfinite(b).all():
                raise AssertionError("Padded output shape or finite-value check failed")
            close = bool(np.allclose(a, b, atol=args.atol, rtol=args.rtol))
            parity_passed = parity_passed and close
            differences.append(
                {
                    "shape": a.shape,
                    "within_tolerance": close,
                    "max_abs": float(np.max(np.abs(a - b))),
                    "rms": float(np.sqrt(np.mean((a - b) ** 2))),
                }
            )
        report = {
            "fixture": index,
            "native_atoms": n,
            "padded_atoms": bucket,
            "output_differences": differences,
            "parity_passed": parity_passed,
        }
        print(json.dumps(report), flush=True)
        results.append(report)
    with open(args.output, "w") as handle:
        json.dump(
            {
                "model": args.model,
                "cycles": args.cycles,
                "steps": args.steps,
                "devices": [str(d) for d in jax.devices()],
                "atol": args.atol,
                "rtol": args.rtol,
                "matmul_precision": jax.config.jax_default_matmul_precision,
                "results": results,
            },
            handle,
            indent=2,
        )

    if any(not result.get("parity_passed", True) for result in results):
        raise SystemExit(
            "Numerical parity tolerance exceeded; inspect the saved report"
        )


if __name__ == "__main__":
    main()
