"""
pipeline/build_molecular_graphs.py
===================================
Convert drug SMILES strings to PyTorch Geometric molecular graphs.

Atom features (74-dim):
  - Atomic number one-hot (44 common elements)
  - Degree one-hot (0-10)
  - Formal charge (5 bins: -2,-1,0,1,2)
  - Hybridization one-hot (SP, SP2, SP3, SP3D, SP3D2, OTHER)
  - Aromaticity (binary)
  - Hydrogen count (0,1,2,3,4+)
  - Ring membership (binary)
  - Implicit valence (0-5+)

Bond features (12-dim):
  - Bond type one-hot (SINGLE, DOUBLE, TRIPLE, AROMATIC)
  - Ring membership (binary)
  - Stereo one-hot (NONE, ANY, Z, E, CIS, TRANS)
  - Conjugated (binary)

Output:
  - data/graphs/molecular/{drugbank_id}.pt  (one file per drug)
  - data/graphs/drug_graph_index.parquet    (index: drug_id → file path)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch_geometric.data import Data
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.loader import load_config  # noqa: E402

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')   # suppress RDKit deprecation noise
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdchem import BondStereo, BondType, HybridizationType
    RDKIT_AVAILABLE = True
except ImportError:
    logger.error("RDKit not installed. Run: pip install rdkit")
    RDKIT_AVAILABLE = False

# ─── Feature Constants ────────────────────────────────────────────────────────

# Most common elements in drug molecules (44)
COMMON_ATOMS = [
    1, 6, 7, 8, 9, 15, 16, 17, 35, 53,  # H,C,N,O,F,P,S,Cl,Br,I
    5, 14, 32, 33, 34, 50, 51, 52, 56,   # B,Si,Ge,As,Se,Sn,Sb,Te,Ba
    11, 12, 19, 20, 26, 27, 28, 29, 30,  # Na,Mg,K,Ca,Fe,Co,Ni,Cu,Zn
    40, 42, 44, 45, 46, 47, 48,          # Zr,Mo,Ru,Rh,Pd,Ag,Cd
    78, 79, 80, 82,                       # Pt,Au,Hg,Pb
]
ATOM_DEGREE_VALUES = list(range(11))     # 0–10
FORMAL_CHARGE_VALUES = [-2, -1, 0, 1, 2]
HYBRIDIZATION_VALUES = [
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    HybridizationType.OTHER,
] if RDKIT_AVAILABLE else []
H_COUNT_VALUES = [0, 1, 2, 3, 4]
IMPLICIT_VALENCE_VALUES = list(range(6))  # 0–5

BOND_TYPE_VALUES = [
    BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC
] if RDKIT_AVAILABLE else []
BOND_STEREO_VALUES = [
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
] if RDKIT_AVAILABLE else []


def _one_hot(value: Any, choices: list) -> list[float]:
    """One-hot encode value from choices list. Unknown → zero vector."""
    encoding = [0.0] * (len(choices) + 1)  # +1 for "other"
    try:
        idx = choices.index(value)
        encoding[idx] = 1.0
    except ValueError:
        encoding[-1] = 1.0  # "other" bucket
    return encoding


def atom_features(atom: Any) -> torch.Tensor:
    """
    Compute 74-dimensional atom feature vector.

    Feature breakdown:
      atomic_num: 45 (44 common + other)
      degree:     11 (0-10 + other? no, just 11)
      charge:      5
      hybrid:      6
      aromatic:    1
      h_count:     5 (0-4+)
      in_ring:     1
      impl_val:    6
    Total: 45+11+5+6+1+5+1+6 = 80... let's recount for actual dim
    """
    feats = []

    # Atomic number one-hot (44 choices + 1 other = 45 dims)
    feats.extend(_one_hot(atom.GetAtomicNum(), COMMON_ATOMS))

    # Degree one-hot (11 values + 1 other = 12, but we use 11)
    feats.extend(_one_hot(atom.GetDegree(), ATOM_DEGREE_VALUES))

    # Formal charge one-hot (5 values)
    fc = atom.GetFormalCharge()
    fc = max(-2, min(2, fc))  # clamp to [-2, 2]
    feats.extend(_one_hot(fc, FORMAL_CHARGE_VALUES))

    # Hybridization one-hot (6 values)
    feats.extend(_one_hot(atom.GetHybridization(), HYBRIDIZATION_VALUES))

    # Aromaticity (1 dim)
    feats.append(float(atom.GetIsAromatic()))

    # Total hydrogen count (5 values: 0,1,2,3,4+)
    h = min(atom.GetTotalNumHs(), 4)
    feats.extend(_one_hot(h, H_COUNT_VALUES))

    # Ring membership (1 dim)
    feats.append(float(atom.IsInRing()))

    # Implicit valence (6 values: 0-5+)
    iv = min(atom.GetImplicitValence(), 5)
    feats.extend(_one_hot(iv, IMPLICIT_VALENCE_VALUES))

    return torch.tensor(feats, dtype=torch.float)


def bond_features(bond: Any) -> torch.Tensor:
    """
    Compute 12-dimensional bond feature vector.

    Feature breakdown:
      bond_type:  4 (SINGLE, DOUBLE, TRIPLE, AROMATIC) + 1 other = 5
      in_ring:    1
      stereo:     6 types + 1 other = 7 — but we use 6 flat
      conjugated: 1
    Total: 5+1+6+1 = 13 → actual: 4+1+6+1=12 (no "other" for bond type/stereo)
    """
    feats = []

    # Bond type one-hot (4 dims + other = 5)
    feats.extend(_one_hot(bond.GetBondType(), BOND_TYPE_VALUES))

    # Ring membership (1 dim)
    feats.append(float(bond.IsInRing()))

    # Stereo (6 dims + other = 7, but flatten to 6)
    stereo = bond.GetStereo()
    stereo_enc = [0.0] * len(BOND_STEREO_VALUES)
    if stereo in BOND_STEREO_VALUES:
        stereo_enc[BOND_STEREO_VALUES.index(stereo)] = 1.0
    feats.extend(stereo_enc)

    # Conjugated (1 dim)
    feats.append(float(bond.GetIsConjugated()))

    return torch.tensor(feats, dtype=torch.float)


# Actual dimensions (bond_type uses _one_hot → 4+1=5 dims, not 4)
ATOM_FEATURE_DIM = 45 + 11 + 5 + 6 + 1 + 5 + 1 + 6  # = 80
BOND_FEATURE_DIM = 5 + 1 + 6 + 1  # bond_type(4+1other) + ring + stereo(6) + conjugated = 13


def smiles_to_graph(
    smiles: str,
    drug_id: str = "",
    max_atoms: int = 120,
) -> Data | None:
    """
    Convert SMILES string to a PyTorch Geometric Data object.

    Args:
        smiles:    SMILES string
        drug_id:   DrugBank ID for the drug (stored as metadata)
        max_atoms: Truncate molecules with more atoms than this

    Returns:
        PyG Data object or None if SMILES is invalid
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required. Install with: pip install rdkit")

    if not smiles or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add hydrogens for accurate H count, then remove for graph (explicit H = noise)
    # Kekulize for consistent bond representation
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return None
    if num_atoms > max_atoms:
        logger.debug(f"  Drug {drug_id}: {num_atoms} atoms > {max_atoms}, truncating")
        # We don't truncate — just skip very large molecules
        return None

    # ── Node features ─────────────────────────────────────────────────────────
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_feats, dim=0)  # [num_atoms, atom_feature_dim]

    # ── Edge index + features (undirected: each bond → 2 directed edges) ──────
    edge_indices = []
    edge_feats = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        # Both directions
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_feats.append(bf)
        edge_feats.append(bf)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()  # [2, num_edges]
        edge_attr = torch.stack(edge_feats, dim=0)  # [num_edges, bond_feature_dim]
    else:
        # Isolated atom (no bonds) — add self-loop with zero bond features
        edge_index = torch.zeros((2, 1), dtype=torch.long)
        edge_attr = torch.zeros((1, BOND_FEATURE_DIM))  # 13-dim, matches bond_features()

    # ── Molecular-level descriptors (stored as graph-level features) ──────────
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        mol_descriptors = torch.tensor(
            [mw / 1000.0, logp / 10.0, tpsa / 200.0, hbd / 10.0, hba / 10.0, rot_bonds / 20.0],
            dtype=torch.float,
        )
    except Exception:
        mol_descriptors = torch.zeros(6, dtype=torch.float)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        mol_descriptors=mol_descriptors,
        num_atoms=torch.tensor([num_atoms], dtype=torch.long),
        drug_id=drug_id,
        smiles=smiles,
    )

    return graph


def build_molecular_graphs(
    drugs_parquet: Path,
    output_dir: Path,
    max_atoms: int = 120,
) -> dict[str, Path]:
    """
    Build molecular graphs for all drugs in the DrugBank Parquet.

    Args:
        drugs_parquet: Path to drugs.parquet
        output_dir:    Directory to save per-drug .pt files
        max_atoms:     Max atoms per molecule

    Returns:
        dict mapping drug_id → Path to .pt file
    """
    import pandas as pd

    if not drugs_parquet.exists():
        raise FileNotFoundError(f"drugs.parquet not found: {drugs_parquet}")

    output_dir.mkdir(parents=True, exist_ok=True)

    drugs_df = pd.read_parquet(drugs_parquet, columns=["drugbank_id", "name", "smiles"])
    # Filter to drugs with SMILES
    drugs_with_smiles = drugs_df[drugs_df["smiles"].notna() & (drugs_df["smiles"] != "")]
    logger.info(f"Building molecular graphs for {len(drugs_with_smiles):,} drugs with SMILES...")

    index: dict[str, Path] = {}
    failed = 0
    too_large = 0

    for _, row in tqdm(drugs_with_smiles.iterrows(), total=len(drugs_with_smiles), desc="Mol graphs"):
        drug_id = row["drugbank_id"]
        smiles = row["smiles"]

        out_path = output_dir / f"{drug_id}.pt"
        if out_path.exists():
            index[drug_id] = out_path
            continue

        graph = smiles_to_graph(smiles, drug_id=drug_id, max_atoms=max_atoms)

        if graph is None:
            logger.debug(f"  Failed to parse SMILES for {drug_id} ({row['name']})")
            failed += 1
            continue

        if graph.num_atoms.item() > max_atoms:
            too_large += 1
            continue

        torch.save(graph, out_path)
        index[drug_id] = out_path

    # Save index
    index_df = pd.DataFrame(
        [{"drug_id": k, "graph_path": str(v)} for k, v in index.items()]
    )
    index_path = output_dir.parent / "drug_graph_index.parquet"
    index_df.to_parquet(index_path, index=False)

    logger.info(
        f"  Molecular graphs built: {len(index):,} | "
        f"Failed (bad SMILES): {failed} | "
        f"Too large (>{max_atoms} atoms): {too_large}"
    )
    return index


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Build molecular graphs from SMILES")
    parser.add_argument("--max-atoms", type=int, default=120)
    args = parser.parse_args()

    cfg = load_config()
    drugs_parquet = ROOT / cfg.paths.data_processed / "drugs.parquet"
    output_dir = ROOT / cfg.paths.data_graphs / "molecular"

    index = build_molecular_graphs(
        drugs_parquet=drugs_parquet,
        output_dir=output_dir,
        max_atoms=args.max_atoms,
    )
    logger.success(f"Built {len(index):,} molecular graphs")
