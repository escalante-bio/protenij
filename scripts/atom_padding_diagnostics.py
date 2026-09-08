"""Host-side comparison, geometry, and optional checkpoint diagnostics."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from protenix.atom_padding import center_atom_coordinates


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
    """Geometry after removing absent atom rows; one result per diffusion sample."""
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


@eqx.filter_jit
def diagnose(model, features, initial, trunk, noise, steps):
    centered = center_atom_coordinates(
        noise * model.inference_noise_scheduler(steps)[0],
        features.get("atom_pad_mask"),
    )
    denoised = model.diffusion_module(
        x_noisy=noise,
        t_hat_noise_level=jnp.full((noise.shape[0],), 5.0),
        input_feature_dict=features,
        s_inputs=initial.s_inputs,
        s_trunk=trunk.s,
        z_trunk=trunk.z,
    )
    return centered, denoised


def diagnostic_differences(native, padded):
    result = {}
    for name, a, b in zip(
        ("initial_center", "fixed_input_denoiser"), native, padded, strict=True
    ):
        difference = a - b
        result[name] = {
            "max_abs": float(np.max(np.abs(difference))),
            "rms": float(np.sqrt(np.mean(difference**2))),
        }
    return result
