# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Template featurizer for loading templates from structure files.

This module provides functionality to load template structures from PDB/mmCIF files
and compute the derived features (distogram, unit vectors, masks) expected by
the TemplateEmbedder module.

Translated from upstream Protenix, adapted to use openfold utilities.
"""

import dataclasses
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray
from biotite.structure.io.pdb import PDBFile as BiotitePDBFile

import protenix.openfold_local.np.residue_constants as rc
from protenix.openfold_local.utils.rigid_utils import Rigid


@dataclasses.dataclass(frozen=True)
class Templates:
    """Dataclass containing template features.

    Stores raw template data (aatype, atom positions, masks) and provides
    methods to compute derived features for the model.

    Attributes:
        aatype: Residue types per template. Shape: (num_templates, num_res)
        atom_positions: Atom coordinates in atom37 format.
            Shape: (num_templates, num_res, 37, 3)
        atom_mask: Atom presence mask in atom37 format.
            Shape: (num_templates, num_res, 37)
    """

    aatype: np.ndarray  # (num_templates, num_res)
    atom_positions: np.ndarray  # (num_templates, num_res, 37, 3)
    atom_mask: np.ndarray  # (num_templates, num_res, 37)

    def as_protenix_dict(self) -> Dict[str, np.ndarray]:
        """Compute derived template features for the TemplateEmbedder.

        Returns:
            Dictionary with:
                template_aatype: (num_templates, num_res)
                template_distogram: (num_templates, num_res, num_res, 39)
                template_pseudo_beta_mask: (num_templates, num_res, num_res)
                template_unit_vector: (num_templates, num_res, num_res, 3)
                template_backbone_frame_mask: (num_templates, num_res, num_res)
        """
        dgrams, pb_masks = [], []
        unit_vectors, bb_masks = [], []

        num_templates = self.aatype.shape[0]
        for i in range(num_templates):
            aatype = self.aatype[i]
            mask = self.atom_mask[i]
            pos = self.atom_positions[i] * mask[..., None]

            # Compute pseudo-beta positions and mask
            pb_pos, pb_mask = _pseudo_beta_fn(aatype, pos, mask)
            pb_mask_2d = pb_mask[:, None] * pb_mask[None, :]

            # Compute distogram (39 bins, 3.25-50.75 Angstroms)
            dgram = _dgram_from_positions(pb_pos)
            dgrams.append(dgram * pb_mask_2d[..., None])
            pb_masks.append(pb_mask_2d)

            # Compute unit vectors from backbone frames
            uv, bb_mask_2d = _compute_template_unit_vector(aatype, pos, mask)
            unit_vectors.append(uv * bb_mask_2d[..., None])
            bb_masks.append(bb_mask_2d)

        return {
            "template_aatype": self.aatype,
            "template_distogram": np.stack(dgrams),
            "template_pseudo_beta_mask": np.stack(pb_masks),
            "template_unit_vector": np.stack(unit_vectors),
            "template_backbone_frame_mask": np.stack(bb_masks),
        }

    def as_torch_dict(self) -> Dict[str, torch.Tensor]:
        """Convert features to torch tensors."""
        np_dict = self.as_protenix_dict()
        return {k: torch.from_numpy(v) for k, v in np_dict.items()}


def _pseudo_beta_fn(
    aatype: np.ndarray,
    atom_positions: np.ndarray,
    atom_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pseudo-beta (CB or CA for Gly) positions and mask.

    Args:
        aatype: Residue types. Shape: (num_res,)
        atom_positions: Atom coordinates in atom37 format. Shape: (num_res, 37, 3)
        atom_mask: Atom presence mask. Shape: (num_res, 37)

    Returns:
        pseudo_beta: CB/CA positions. Shape: (num_res, 3)
        pseudo_beta_mask: Validity mask. Shape: (num_res,)
    """
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]

    # Glycine uses CA, others use CB
    is_gly = aatype == rc.restype_order["G"]

    pseudo_beta = np.where(
        is_gly[..., None],
        atom_positions[:, ca_idx, :],
        atom_positions[:, cb_idx, :],
    )

    pseudo_beta_mask = np.where(
        is_gly,
        atom_mask[:, ca_idx],
        atom_mask[:, cb_idx],
    )

    return pseudo_beta, pseudo_beta_mask


def _dgram_from_positions(
    positions: np.ndarray,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    num_bins: int = 39,
) -> np.ndarray:
    """Compute distogram from pseudo-beta positions.

    Args:
        positions: Pseudo-beta coordinates. Shape: (num_res, 3)
        min_bin: Left edge of first distance bin (Angstroms).
        max_bin: Left edge of last distance bin (Angstroms).
        num_bins: Number of distance bins.

    Returns:
        distogram: One-hot encoded distance bins. Shape: (num_res, num_res, num_bins)
    """
    # Squared distance bins
    lower_breaks = np.linspace(min_bin, max_bin, num_bins) ** 2
    upper_breaks = np.concatenate([lower_breaks[1:], np.array([1e8])])

    # Pairwise squared distances
    diff = positions[:, None, :] - positions[None, :, :]
    dist2 = np.sum(diff ** 2, axis=-1, keepdims=True)

    # One-hot encode into bins
    dgram = (dist2 > lower_breaks) & (dist2 < upper_breaks)
    return dgram.astype(np.float32)


def _compute_template_unit_vector(
    aatype: np.ndarray,
    atom_positions: np.ndarray,
    atom_mask: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized unit vectors between residue pairs in local backbone frames.

    For each residue, constructs a local coordinate frame from backbone atoms (N, CA, C),
    then expresses all other CA positions in that local frame.

    Args:
        aatype: Residue types. Shape: (num_res,)
        atom_positions: Atom coordinates in atom37 format. Shape: (num_res, 37, 3)
        atom_mask: Atom presence mask. Shape: (num_res, 37)
        epsilon: Small value for numerical stability.

    Returns:
        unit_vector: Normalized direction vectors. Shape: (num_res, num_res, 3)
        mask_2d: Validity mask for residue pairs. Shape: (num_res, num_res)
    """
    n_idx = rc.atom_order["N"]
    ca_idx = rc.atom_order["CA"]
    c_idx = rc.atom_order["C"]

    n_pos = atom_positions[:, n_idx]
    ca_pos = atom_positions[:, ca_idx]
    c_pos = atom_positions[:, c_idx]

    n_mask = atom_mask[:, n_idx]
    ca_mask = atom_mask[:, ca_idx]
    c_mask = atom_mask[:, c_idx]

    # Backbone valid where N, CA, C all present
    backbone_mask = (n_mask * ca_mask * c_mask).astype(np.float32)

    # Build local frame: origin at CA, x-axis along C-CA
    v1 = c_pos - ca_pos  # C-CA direction
    v2 = n_pos - ca_pos  # N-CA direction

    # Gram-Schmidt orthogonalization
    e1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + epsilon)
    e2 = v2 - np.sum(v2 * e1, axis=-1, keepdims=True) * e1
    e2 = e2 / (np.linalg.norm(e2, axis=-1, keepdims=True) + epsilon)
    e3 = np.cross(e1, e2)

    # Relative positions: diff[i, j] = CA[j] - CA[i]
    diff = ca_pos[None, :, :] - ca_pos[:, None, :]

    # Transform to local frame: project onto basis vectors
    ux = np.sum(e1[:, None, :] * diff, axis=-1)
    uy = np.sum(e2[:, None, :] * diff, axis=-1)
    uz = np.sum(e3[:, None, :] * diff, axis=-1)

    unit_vector = np.stack([ux, uy, uz], axis=-1)

    # Normalize to unit vector
    uv_norm = np.linalg.norm(unit_vector, axis=-1, keepdims=True)
    unit_vector = unit_vector / (uv_norm + epsilon)

    # 2D mask: valid only if both residues have backbone
    mask_2d = backbone_mask[:, None] * backbone_mask[None, :]

    return unit_vector, mask_2d


def load_templates_from_pdb(
    pdb_path: str,
    chain_id: str,
    num_residues: int,
    query_aatype: Optional[np.ndarray] = None,
) -> Templates:
    """Load template features from a PDB file.

    Parses a PDB structure, extracts atom coordinates for the specified chain,
    and constructs a Templates object ready for feature computation.

    Args:
        pdb_path: Path to PDB file.
        chain_id: Chain identifier to extract.
        num_residues: Number of residues to extract (from the start of the chain).
        query_aatype: Optional query residue types to use instead of template's.
            Shape: (num_residues,). If None, uses the template's residue types.

    Returns:
        Templates object containing the extracted structure data.
    """
    # Parse PDB
    pdb_file = BiotitePDBFile.read(pdb_path)
    atoms = pdb_file.get_structure(model=1)
    chain = atoms[atoms.chain_id == chain_id]

    # Get ordered residue IDs from CA atoms
    ca_atoms = chain[chain.atom_name == "CA"]
    res_ids = ca_atoms.res_id[:num_residues]

    # Build atom37 positions and masks
    atom_positions = np.zeros((num_residues, 37, 3), dtype=np.float32)
    atom_mask = np.zeros((num_residues, 37), dtype=np.float32)

    for i, rid in enumerate(res_ids):
        res_atoms = chain[chain.res_id == rid]
        for atom in res_atoms:
            aname = atom.atom_name
            if aname in rc.atom_order:
                aidx = rc.atom_order[aname]
                atom_positions[i, aidx] = atom.coord
                atom_mask[i, aidx] = 1.0

    # Get aatype from template or use provided query aatype
    if query_aatype is not None:
        aatype = query_aatype
    else:
        # Extract from template structure
        pdb_seq = "".join(
            ProteinSequence.convert_letter_3to1(r)
            for r in ca_atoms.res_name[:num_residues]
        )
        aatype = np.array(
            [rc.restype_order.get(aa, rc.unk_restype_index) for aa in pdb_seq],
            dtype=np.int64,
        )

    # Add template dimension
    return Templates(
        aatype=aatype[None],  # (1, num_res)
        atom_positions=atom_positions[None],  # (1, num_res, 37, 3)
        atom_mask=atom_mask[None],  # (1, num_res, 37)
    )
