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
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from protenix.atom_padding import (
    ATOM_FEATURE_AXES,
    center_atom_coordinates,
    pad_atom_features,
)
from protenix.backend import load_model
from protenix.protenij import Outputs, sample_diffusion


def trunk_embeddings(model, features, cycles, key):
    """Public model stages, factored so downstream validation can use its adapter."""
    initial = model.embed_inputs(input_feature_dict=features)
    trunk = model.recycle(
        initial_embedding=initial,
        input_feature_dict=features,
        recycling_steps=cycles,
        key=key,
    )
    return initial, trunk


def comparison_arrays(output):
    """Select real atoms inside the host-side comparison/export boundary."""
    confidence = output.confidence_metrics
    atom_values = [
        np.asarray(output.coordinates),
        np.asarray(confidence.plddt_logits),
        np.asarray(confidence.resolved_logits),
    ]
    if output.atom_pad_mask is not None:
        mask = np.asarray(output.atom_pad_mask)
        if mask.dtype != np.bool_ or mask.shape != (atom_values[0].shape[-2],):
            raise ValueError("Compare each sequence separately with its atom mask")
        atom_values = [value[..., mask, :] for value in atom_values]
    return (*atom_values, confidence.pae_logits, confidence.pde_logits,
            output.distogram_logits)


def compare_outputs(native, padded, features):
    """Geometry after removing absent atom rows; one result per independent key."""
    n = features["atom_to_token_idx"].shape[0]
    a = comparison_arrays(native)[0].reshape(-1, n, 3)
    b = comparison_arrays(padded)[0].reshape(-1, n, 3)
    geometry = []
    for ref, test in zip(a, b, strict=True):
        ref_centered = ref - ref.mean(0)
        test_centered = test - test.mean(0)
        u, _, vh = np.linalg.svd(test_centered.T @ ref_centered)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vh)
        fitted = test_centered @ (u @ correction @ vh)
        delta = ref - test
        geometry.append(
            {
                "all_atom_rmsd": float(np.sqrt(np.mean(np.sum(delta**2, axis=-1)))),
                "aligned_all_atom_rmsd": float(
                    np.sqrt(np.mean(np.sum((ref_centered - fitted) ** 2, axis=-1)))
                ),
                "max_atom_distance": float(np.sqrt(np.sum(delta**2, axis=-1)).max()),
            }
        )
    return {"geometry": geometry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--model", default="protenix-v2")
    parser.add_argument("--padding-multiple", type=int, default=256)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warm-runs", type=int, default=1)
    parser.add_argument("--coordinates-dir", type=Path)
    parser.add_argument("--output", default="atom-padding-validation.json")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--diagnose-centering", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.cycles < 1 or args.steps < 1 or args.warm_runs < 0:
        parser.error(
            "samples/cycles/steps must be positive; warm-runs must be nonnegative"
        )
    if args.coordinates_dir is not None:
        args.coordinates_dir.mkdir(parents=True, exist_ok=True)
    model = jax.tree.map(
        lambda x: jax.device_put(x) if eqx.is_array(x) else x, load_model(args.model)
    )
    features = []
    for path in args.features:
        with open(path, "rb") as handle:
            features.append(pickle.load(handle))
    traces = []

    def predict_one(model, features, initial_noise, step_noise, key):
        initial, trunk = trunk_embeddings(model, features, args.cycles, key)
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
            key=key,
            initial_noise=initial_noise,
            step_noise=step_noise,
        )
        confidence = model.confidence_metrics(
            initial_embedding=initial,
            trunk_embedding=trunk,
            input_feature_dict=features,
            coordinates=coordinates,
            key=key,
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

    @eqx.filter_jit
    def predict(model, features, initial_noise, step_noise, keys):
        traces.append(1)
        initial, trunk, out = jax.vmap(
            lambda noise, steps, key: predict_one(model, features, noise, steps, key)
        )(initial_noise, step_noise, keys)
        # All independent keys here belong to one sequence, with one presence mask.
        out = Outputs(
            out.coordinates,
            out.confidence_metrics,
            out.distogram_logits,
            features.get("atom_pad_mask"),
        )
        return initial, trunk, out

    @eqx.filter_jit
    def diagnose(model, f, initial_embedding, trunk_embedding, noise):
        def one(initial, trunk, noise):
            centered = center_atom_coordinates(
                noise * model.inference_noise_scheduler(args.steps)[0],
                f.get("atom_pad_mask"),
            )
            denoised = model.diffusion_module(
                x_noisy=noise,
                t_hat_noise_level=jnp.array([5.0]),
                input_feature_dict=f,
                s_inputs=initial.s_inputs,
                s_trunk=trunk.s,
                z_trunk=trunk.z,
            )
            return centered, denoised

        return jax.vmap(one)(initial_embedding, trunk_embedding, noise)

    results = []
    for index, native in enumerate(features):
        padded = pad_atom_features(native, padding_multiple=args.padding_multiple)
        n = native["atom_to_token_idx"].shape[0]
        bucket = padded["atom_to_token_idx"].shape[0]
        # Invalid metadata/indices must be harmless, not merely zero by convention.
        for name, axes in ATOM_FEATURE_AXES.items():
            if name == "atom_pad_mask" or name not in padded:
                continue
            value = padded[name]
            poison = np.nan if np.issubdtype(value.dtype, np.floating) else 999999
            for axis in axes:
                selection = [slice(None)] * value.ndim
                selection[axis] = slice(n, None)
                value[tuple(selection)] = poison
        keys = jnp.stack(
            [
                jax.random.fold_in(
                    jax.random.fold_in(jax.random.key(args.seed), index), sample
                )
                for sample in range(args.samples)
            ]
        )

        # Native-shaped draws reproduce each chosen unpadded key's RNG stream.
        # Copy those same valid values into the padded bucket for parity.
        def noise_for_key(key, n=n):
            initial = jax.random.normal(key, (1, n, 3))

            def add_noise(key, _):
                key = jax.random.fold_in(key, 1)
                return key, jax.random.normal(key, (1, n, 3))

            return initial, jax.lax.scan(add_noise, key, None, length=args.steps)[1]

        initial, step = jax.vmap(noise_for_key)(keys)
        initial = jnp.pad(
            initial, ((0, 0), (0, 0), (0, bucket - n), (0, 0)), constant_values=jnp.nan
        )
        step = jnp.pad(
            step,
            ((0, 0), (0, 0), (0, 0), (0, bucket - n), (0, 0)),
            constant_values=jnp.nan,
        )
        outputs = []
        diagnostics = []
        for label, f, ni, ns in (
            ("native", native, initial[:, :, :n], step[:, :, :, :n]),
            ("padded", padded, initial, step),
        ):
            f, ni, ns = jax.device_put((f, ni, ns))
            jax.block_until_ready((f, ni, ns, keys))
            trace_count = len(traces)
            start = time.perf_counter()
            lowered = predict.lower(model, f, ni, ns, keys)
            lowered_s = time.perf_counter() - start
            start = time.perf_counter()
            compiled = lowered.compile()
            compile_s = time.perf_counter() - start
            start = time.perf_counter()
            value = compiled(model, f, ni, ns, keys)
            jax.block_until_ready(value)
            execution_s = time.perf_counter() - start
            warm_s = []
            for _ in range(args.warm_runs):
                start = time.perf_counter()
                jax.block_until_ready(compiled(model, f, ni, ns, keys))
                warm_s.append(time.perf_counter() - start)
            timing = {
                "fixture": index,
                "variant": label,
                "atoms": f["atom_to_token_idx"].shape[0],
                "tokens": f["residue_index"].shape[0],
                "lower_s": lowered_s,
                "compile_s": compile_s,
                "first_execution_s": execution_s,
                "warm_execution_s": warm_s,
                "sample_keys": np.asarray(jax.random.key_data(keys)).tolist(),
                "new_traces": len(traces) - trace_count,
            }
            print(json.dumps(timing), flush=True)
            results.append(timing)
            initial_out, trunk_out, output = value
            if args.diagnose_centering:
                diagnosis = diagnose(model, f, initial_out, trunk_out, ni)
                diagnostics.append(tuple(np.asarray(x)[..., :n, :] for x in diagnosis))
            if label == "padded":
                for name, values in (
                    ("coordinates", output.coordinates),
                    ("plddt_logits", output.confidence_metrics.plddt_logits),
                    ("resolved_logits", output.confidence_metrics.resolved_logits),
                ):
                    absent = np.asarray(values)[..., n:, :]
                    if not np.isfinite(absent).all() or np.any(absent != 0):
                        raise AssertionError(f"Padding {name} are not exactly zero")
            if args.coordinates_dir is not None:
                np.savez_compressed(
                    args.coordinates_dir / f"fixture{index}-{label}.npz",
                    coordinates=comparison_arrays(output)[0],
                    sample_keys=np.asarray(jax.random.key_data(keys)),
                )
            outputs.append(jax.device_get((initial_out, trunk_out, output)))
        native_leaves, padded_leaves = [
            jax.tree.leaves((initial, trunk, comparison_arrays(output)))
            for initial, trunk, output in outputs
        ]
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
            "padding_excluded": True,
            **compare_outputs(outputs[0][2], outputs[1][2], native),
        }
        if diagnostics:
            report["diagnostics"] = {}
            for name, a, b in zip(
                ("initial_center", "fixed_input_denoiser"),
                diagnostics[0],
                diagnostics[1],
                strict=True,
            ):
                difference = a - b
                report["diagnostics"][name] = {
                    "max_abs": float(np.max(np.abs(difference))),
                    "rms": float(np.sqrt(np.mean(difference**2))),
                }
        print(json.dumps(report), flush=True)
        results.append(report)
    with open(args.output, "w") as handle:
        json.dump(
            {
                "model": args.model,
                "cycles": args.cycles,
                "steps": args.steps,
                "samples": args.samples,
                "seed": args.seed,
                "sampler": {
                    "gamma0": model.gamma0,
                    "step_scale_eta": model.step_scale_eta,
                    "noise_scale_lambda": model.noise_scale_lambda,
                },
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
