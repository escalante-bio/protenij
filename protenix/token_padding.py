"""Explicit token-presence padding for one unbatched JAX inference input."""

import jax.numpy as jnp
import numpy as np

from .atom_padding import mask_atom_values

# Axes are schema-defined, never inferred from coincidentally equal dimensions.
TOKEN_FEATURE_AXES = {
    **{name: (0,) for name in (
        "token_index", "residue_index", "asym_id", "entity_id", "sym_id",
        "restype", "profile", "deletion_mean", "atom_rep_atom_idx",
        "has_frame", "frame_atom_index",
        "token_pad_mask",
    )},
    "token_bonds": (0, 1),
    "msa": (1,),
    "has_deletion": (1,),
    "deletion_value": (1,),
    "template_aatype": (1,),
    "template_distogram": (1, 2),
    "template_pseudo_beta_mask": (1, 2),
    "template_unit_vector": (1, 2),
    "template_backbone_frame_mask": (1, 2),
    **{f"constraint_feature.{name}": (0, 1) for name in (
        "contact_atom", "contact", "pocket", "substructure",
    )},
}


def _feature_leaf(features, name):
    """Locate an explicitly named nested feature without mutating its parents."""
    result = features
    parts = name.split(".")
    for part in parts[:-1]:
        if part not in result:
            return None, parts[-1]
        result[part] = dict(result[part])
        result = result[part]
    return result, parts[-1]


def pad_token_features(features, token_count=None, *, padding_multiple=64,
                       extra_token_axes=None):
    """Pad explicit token axes, retaining native token/atom index values.

    Compose with ``pad_atom_features`` in either order. MSA row and template
    counts are unchanged; callers must separately match these for JIT reuse.
    Additional feature axes must be declared in ``extra_token_axes``; dotted
    names address nested features (e.g. ``constraint_feature.contact``).
    """
    if features["residue_index"].ndim != 1:
        raise ValueError("Pad each feature dictionary before batching")
    native_count = features["residue_index"].shape[0]
    if not isinstance(padding_multiple, (int, np.integer)) or padding_multiple < 1:
        raise ValueError("padding_multiple must be a positive integer")
    if token_count is None:
        token_count = ((native_count + padding_multiple - 1) // padding_multiple) * padding_multiple
    if not isinstance(token_count, (int, np.integer)) or token_count < native_count:
        raise ValueError("token_count must be an integer at least the existing token count")
    result = dict(features)
    result["token_pad_mask"] = np.asarray(
        features.get("token_pad_mask", np.ones(native_count, dtype=bool)), dtype=bool
    )
    axes = TOKEN_FEATURE_AXES | (extra_token_axes or {})
    for name, token_axes in axes.items():
        parent, leaf = _feature_leaf(result, name)
        if parent is None or leaf not in parent:
            continue
        value = np.asarray(parent[leaf])
        widths = [(0, 0)] * value.ndim
        for axis in token_axes:
            if value.shape[axis] != native_count:
                raise ValueError(f"{name} axis {axis} does not match the token count")
            widths[axis] = (0, token_count - native_count)
        parent[leaf] = np.pad(value, widths)
    return result


def token_pair_mask(features):
    mask = features.get("token_pad_mask")
    return None if mask is None else mask[:, None] & mask[None, :]


def mask_pair_values(values, pair_mask):
    return values if pair_mask is None else jnp.where(pair_mask[..., None], values, 0)


def mask_token_features(features):
    """Sanitize dummy metadata, including NaNs/invalid indices, before use."""
    mask = features.get("token_pad_mask")
    if mask is None:
        return features
    result = dict(features)
    for name, axes in TOKEN_FEATURE_AXES.items():
        parent, leaf = _feature_leaf(result, name)
        if name == "token_pad_mask" or parent is None or leaf not in parent:
            continue
        value = parent[leaf]
        for axis in axes:
            value = mask_atom_values(value, mask, axis=axis)
        parent[leaf] = value
    return result
