"""Explicit atom-presence padding for the JAX inference API.

``ref_mask`` describes availability of reference-conformer coordinates; it does
not describe whether an atom exists. ``atom_pad_mask`` is a separate boolean
array: True for a real atom, False for a padding row. Tokens are not padded.
"""

import jax.numpy as jnp
import numpy as np

# Explicit schema: never infer atom axes merely from a dimension's size (a
# token, MSA or template dimension can coincidentally equal the atom count).
ATOM_FEATURE_AXES = {
    "ref_pos": (0,),
    "ref_mask": (0,),
    "ref_element": (0,),
    "ref_charge": (0,),
    "ref_atom_name_chars": (0,),
    "ref_space_uid": (0,),
    "atom_to_token_idx": (0,),
    "atom_to_tokatom_idx": (0,),
    "is_protein": (0,),
    "is_ligand": (0,),
    "is_dna": (0,),
    "is_rna": (0,),
    "mol_id": (0,),
    "mol_atom_index": (0,),
    "entity_mol_id": (0,),
    "pae_rep_atom_mask": (0,),
    "plddt_m_rep_atom_mask": (0,),
    "distogram_rep_atom_mask": (0,),
    "modified_res_mask": (0,),
    "bond_mask": (0, 1),
    "atom_pad_mask": (0,),
}


def pad_atom_features(
    features, atom_count=None, *, padding_multiple=256, extra_atom_axes=None
):
    """Return NumPy features padded to a common atom bucket before JIT/vmap.

    If atom_count is omitted, round up to padding_multiple (default 256).
    Accepts one unbatched, inference feature dictionary. Native rows and token
    features (including atom indices into native rows) are unchanged. Callers
    with additional atom-indexed features must declare them in
    ``extra_atom_axes``. Training labels and nested custom atom features are not
    automatically padded. No MSA, token or residue padding is performed.
    """
    native_count = features["atom_to_token_idx"].shape[0]
    if not isinstance(padding_multiple, (int, np.integer)) or padding_multiple < 1:
        raise ValueError("padding_multiple must be a positive integer")
    if atom_count is None:
        atom_count = (
            (native_count + padding_multiple - 1) // padding_multiple
        ) * padding_multiple
    if features["atom_to_token_idx"].ndim != 1:
        raise ValueError("Pad each feature dictionary before batching")
    if not isinstance(atom_count, (int, np.integer)) or atom_count < native_count:
        raise ValueError(
            "atom_count must be an integer at least the existing atom count"
        )
    axes = dict(ATOM_FEATURE_AXES)
    axes.update(extra_atom_axes or {})
    result = dict(features)
    result["atom_pad_mask"] = np.asarray(
        features.get("atom_pad_mask", np.ones(native_count, dtype=bool)), dtype=bool
    )
    for name, atom_axes in axes.items():
        if name not in result:
            continue
        value = np.asarray(result[name])
        widths = [(0, 0)] * value.ndim
        for axis in atom_axes:
            if value.shape[axis] != native_count:
                raise ValueError(f"{name} axis {axis} does not match the atom count")
            widths[axis] = (0, atom_count - native_count)
        result[name] = np.pad(value, widths)
    return result


def mask_atom_values(values, atom_mask):
    """Zero absent rows of [..., atoms, channels], including NaN padding."""
    if atom_mask is None:
        return values
    return jnp.where(jnp.asarray(atom_mask, dtype=bool)[..., None], values, 0)


def center_atom_coordinates(coordinates, atom_mask=None):
    """Center using real atoms only, independently for each sample."""
    if atom_mask is None:
        return coordinates - jnp.mean(coordinates, axis=-2, keepdims=True)
    coordinates = mask_atom_values(coordinates, atom_mask)
    count = jnp.maximum(jnp.sum(atom_mask), 1)
    center = jnp.sum(coordinates, axis=-2, keepdims=True) / count
    return mask_atom_values(coordinates - center, atom_mask)
