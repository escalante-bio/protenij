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
"""

import dataclasses
from typing import Dict, Optional, Sequence

import gemmi
import numpy as np
import torch

import protenix.openfold_local.np.residue_constants as rc


@dataclasses.dataclass
class ChainInput:
    """Input specification for a single chain in a prediction.

    Attributes:
        sequence: Amino acid sequence (one-letter codes).
        compute_msa: Whether to search for MSA for this chain. If False,
            uses a dummy MSA (just the query sequence).
        precomputed_msa_dir: Optional path to directory containing pre-computed
            MSA files (pairing.a3m, non_pairing.a3m). If provided, used instead
            of running MSA search.
        template: Optional gemmi.Chain containing template coordinates,
            or None if no template is available for this chain.
    """

    sequence: str
    compute_msa: bool = True
    precomputed_msa_dir: Optional[str] = None
    template: Optional[gemmi.Chain] = None


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


def load_templates_from_structure(
    path: str,
    chain_id: str,
    num_residues: Optional[int] = None,
    query_aatype: Optional[np.ndarray] = None,
) -> Templates:
    """Load template features from a PDB or mmCIF file.

    Parses a structure file using gemmi, extracts atom coordinates for the
    specified chain, and constructs a Templates object ready for feature computation.

    Args:
        path: Path to PDB or mmCIF file.
        chain_id: Chain identifier to extract.
        num_residues: Number of residues to extract. If None, uses all residues.
        query_aatype: Optional query residue types to use instead of template's.
            Shape: (num_residues,). If None, uses the template's residue types.

    Returns:
        Templates object containing the extracted structure data.
    """
    # Parse structure (gemmi auto-detects PDB vs mmCIF)
    structure = gemmi.read_structure(path)
    model = structure[0]  # First model
    chain = model[chain_id]

    # Get polymer residues only (skip ligands, water, etc.)
    residues = [
        res for res in chain
        if gemmi.find_tabulated_residue(res.name).is_amino_acid()
    ]
    if num_residues is not None:
        residues = residues[:num_residues]
    else:
        num_residues = len(residues)

    # Build atom37 positions and masks
    atom_positions = np.zeros((num_residues, 37, 3), dtype=np.float32)
    atom_mask = np.zeros((num_residues, 37), dtype=np.float32)
    res_names = []

    for i, res in enumerate(residues):
        res_names.append(res.name)
        for atom in res:
            aname = atom.name
            if aname in rc.atom_order:
                aidx = rc.atom_order[aname]
                atom_positions[i, aidx] = [atom.pos.x, atom.pos.y, atom.pos.z]
                atom_mask[i, aidx] = 1.0

    # Get aatype from template or use provided query aatype
    if query_aatype is not None:
        aatype = query_aatype
    else:
        # Extract from template structure using gemmi's 3-to-1 conversion
        aatype = np.array(
            [
                rc.restype_order.get(
                    gemmi.find_tabulated_residue(name).one_letter_code,
                    rc.unk_restype_index,
                )
                for name in res_names
            ],
            dtype=np.int64,
        )

    # Add template dimension
    return Templates(
        aatype=aatype[None],  # (1, num_res)
        atom_positions=atom_positions[None],  # (1, num_res, 37, 3)
        atom_mask=atom_mask[None],  # (1, num_res, 37)
    )


# Backwards compatibility alias
load_templates_from_pdb = load_templates_from_structure


def _extract_chain_template(
    chain: gemmi.Chain,
    expected_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract template features from a gemmi.Chain.

    Args:
        chain: gemmi.Chain containing template coordinates.
        expected_length: Expected number of residues (for validation/padding).

    Returns:
        Tuple of (aatype, atom_positions, atom_mask) for this chain.
        Shapes: (num_res,), (num_res, 37, 3), (num_res, 37)
    """
    # Get polymer residues only
    residues = [
        res for res in chain
        if gemmi.find_tabulated_residue(res.name).is_amino_acid()
    ]
    num_res = len(residues)

    if num_res != expected_length:
        raise ValueError(
            f"Template chain has {num_res} residues but expected {expected_length}"
        )

    # Build atom37 positions and masks
    atom_positions = np.zeros((num_res, 37, 3), dtype=np.float32)
    atom_mask = np.zeros((num_res, 37), dtype=np.float32)
    aatype = np.zeros(num_res, dtype=np.int64)

    for i, res in enumerate(residues):
        # Get aatype from residue name
        one_letter = gemmi.find_tabulated_residue(res.name).one_letter_code
        aatype[i] = rc.restype_order.get(one_letter, rc.unk_restype_index)

        # Extract atom coordinates
        for atom in res:
            if atom.name in rc.atom_order:
                aidx = rc.atom_order[atom.name]
                atom_positions[i, aidx] = [atom.pos.x, atom.pos.y, atom.pos.z]
                atom_mask[i, aidx] = 1.0

    return aatype, atom_positions, atom_mask


def _empty_chain_template(num_res: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create empty template features for a chain without a template.

    Args:
        num_res: Number of residues in the chain.

    Returns:
        Tuple of (aatype, atom_positions, atom_mask) filled with gaps/zeros.
        Shapes: (num_res,), (num_res, 37, 3), (num_res, 37)
    """
    # Use gap token (index 20 in standard ordering, but we use unk which is also fine)
    # The mask being zero means these positions won't contribute
    aatype = np.full(num_res, rc.unk_restype_index, dtype=np.int64)
    atom_positions = np.zeros((num_res, 37, 3), dtype=np.float32)
    atom_mask = np.zeros((num_res, 37), dtype=np.float32)
    return aatype, atom_positions, atom_mask


def build_templates_from_chains(
    chains: Sequence[ChainInput],
    num_templates: int = 1,
) -> Templates:
    """Build combined template features from multiple chain inputs.

    Creates a Templates object with features for all chains concatenated.
    Chains with templates get their template coordinates; chains without
    templates get zero-filled features (which the model ignores via masks).

    Args:
        chains: Sequence of ChainInput specifications.
        num_templates: Number of template slots to create (default 1).

    Returns:
        Templates object with concatenated features for all chains.
        Shapes will be (num_templates, total_residues, ...).
    """
    total_residues = sum(len(c.sequence) for c in chains)

    # Initialize arrays for all templates
    all_aatype = np.zeros((num_templates, total_residues), dtype=np.int64)
    all_positions = np.zeros((num_templates, total_residues, 37, 3), dtype=np.float32)
    all_mask = np.zeros((num_templates, total_residues, 37), dtype=np.float32)

    # Fill in each chain's contribution
    offset = 0
    for chain_input in chains:
        chain_len = len(chain_input.sequence)

        if chain_input.template is not None:
            # Extract template features from the provided chain
            aatype, positions, mask = _extract_chain_template(
                chain_input.template, chain_len
            )
        else:
            # No template - use empty features
            aatype, positions, mask = _empty_chain_template(chain_len)

        # Copy into the first template slot (others remain zero)
        all_aatype[0, offset : offset + chain_len] = aatype
        all_positions[0, offset : offset + chain_len] = positions
        all_mask[0, offset : offset + chain_len] = mask

        offset += chain_len

    return Templates(
        aatype=all_aatype,
        atom_positions=all_positions,
        atom_mask=all_mask,
    )


# ============================================================================
# Featurization
# ============================================================================


def _make_dummy_msa(sequence: str, msa_dir: str) -> None:
    """Create dummy MSA files (just the query sequence)."""
    import os
    os.makedirs(msa_dir, exist_ok=True)
    for fname in ["pairing.a3m", "non_pairing.a3m"]:
        with open(os.path.join(msa_dir, fname), "w") as f:
            f.write(f">query\n{sequence}\n")


def _run_msa_search(sequences: Sequence[str], msa_dir: str, email: str = "") -> list:
    """Run MMSeqs2 search for sequences, return list of result directories."""
    import os
    from protenix.web_service.colab_request_parser import RequestParser

    os.makedirs(msa_dir, exist_ok=True)
    seqs_sorted = sorted(set(sequences))
    tmp_fasta = os.path.join(msa_dir, "msa_input.fasta")

    RequestParser.msa_search(
        seqs_pending_msa=seqs_sorted,
        tmp_fasta_fpath=tmp_fasta,
        msa_res_dir=msa_dir,
        email=email,
    )
    return RequestParser.msa_postprocess(
        seqs_pending_msa=seqs_sorted,
        msa_res_dir=msa_dir,
    )


def featurize(
    chains: Sequence[ChainInput],
    name: str = "prediction",
    email: str = "",
) -> tuple[Dict[str, np.ndarray], "AtomArray", "TokenArray"]:
    """Featurize a multi-chain complex for structure prediction.

    Takes a sequence of ChainInputs and produces complete features ready
    for the model. Handles everything:
    - MSA: runs search if compute_msa=True, uses precomputed if provided,
      otherwise uses dummy MSA (just the query sequence)
    - Templates: extracts features from template chains if provided
    - All standard Protenix featurization

    Args:
        chains: Sequence of ChainInput specifications. Each chain has:
            - sequence: amino acid sequence
            - compute_msa: whether to run MSA search (default True)
            - precomputed_msa_dir: path to existing MSA (optional)
            - template: gemmi.Chain with template coordinates (optional)
        name: Name for the prediction.
        email: Optional email for MSA service.

    Returns:
        Tuple of (features_dict, atom_array, token_array).
        features_dict contains numpy arrays ready for JAX.

    Example:
        >>> import gemmi
        >>> structure = gemmi.read_structure("template.pdb")
        >>> chains = [
        ...     ChainInput("ACDEFGHIKLMNPQRSTVWY", compute_msa=False, template=structure[0]['A']),
        ...     ChainInput("GGGGGGGGGG", compute_msa=False),
        ... ]
        >>> features, atom_array, token_array = featurize(chains)
    """
    import tempfile
    from collections import defaultdict
    import torch
    from protenix.data.json_to_feature import SampleDictToFeatures
    from protenix.data.msa_featurizer import InferenceMSAFeaturizer
    from protenix.data.utils import data_type_transform

    # --- Build sample dict with MSA configuration ---

    # Collect sequences needing MSA search
    seqs_needing_search = []
    for chain in chains:
        if chain.compute_msa and chain.precomputed_msa_dir is None:
            seqs_needing_search.append(chain.sequence)

    # Run MSA search if needed
    seq_to_msa_dir: Dict[str, str] = {}
    if seqs_needing_search:
        msa_tmpdir = tempfile.mkdtemp(prefix="protenix_msa_")
        unique_seqs = sorted(set(seqs_needing_search))
        msa_subdirs = _run_msa_search(unique_seqs, msa_tmpdir, email)
        seq_to_msa_dir = dict(zip(unique_seqs, msa_subdirs))

    # Build sequences list for sample dict
    sequences = []
    for i, chain in enumerate(chains):
        protein_chain = {"sequence": chain.sequence, "count": 1}

        if chain.precomputed_msa_dir is not None:
            msa_dir = chain.precomputed_msa_dir
        elif chain.compute_msa:
            msa_dir = seq_to_msa_dir[chain.sequence]
        else:
            # Create dummy MSA
            msa_dir = tempfile.mkdtemp(prefix=f"protenix_msa_dummy_{i}_")
            _make_dummy_msa(chain.sequence, msa_dir)

        protein_chain["msa"] = {
            "precomputed_msa_dir": msa_dir,
            "pairing_db": "uniref100",
        }
        sequences.append({"proteinChain": protein_chain})

    sample = {"name": name, "sequences": sequences}

    # --- Base featurization ---

    sample2feat = SampleDictToFeatures(sample)
    features_dict, atom_array, token_array = sample2feat.get_feature_dict()
    features_dict["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()

    # --- MSA featurization ---

    # Build entity_to_asym_id mapping from atom_array
    entity_to_asym_id: Dict[str, set] = defaultdict(set)
    for entity_id, asym_id_int in zip(
        atom_array.label_entity_id, atom_array.asym_id_int
    ):
        entity_to_asym_id[entity_id].add(asym_id_int)
    entity_to_asym_id = dict(entity_to_asym_id)

    # Load and process MSA features
    msa_feats = InferenceMSAFeaturizer.make_msa_feature(
        bioassembly=sequences,
        entity_to_asym_id=entity_to_asym_id,
        token_array=token_array,
        atom_array=atom_array,
    )
    if msa_feats:
        for k, v in msa_feats.items():
            features_dict[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v

    # Apply data type transforms (still uses torch internally)
    features_dict = data_type_transform(features_dict)

    # --- Template features ---

    templates = build_templates_from_chains(chains)
    features_dict.update(templates.as_torch_dict())

    # --- Convert to numpy ---

    def to_numpy(v):
        if isinstance(v, torch.Tensor):
            return v.numpy()
        elif isinstance(v, dict):
            return {k2: to_numpy(v2) for k2, v2 in v.items()}
        elif isinstance(v, np.ndarray):
            return v
        else:
            return v

    result = {k: to_numpy(v) for k, v in features_dict.items()}

    # Add atom_rep_atom_idx (needed by JAX model)
    result["atom_rep_atom_idx"] = result["distogram_rep_atom_mask"].nonzero()[0]

    return result, atom_array, token_array
