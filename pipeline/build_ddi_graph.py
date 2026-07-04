"""
pipeline/build_ddi_graph.py
============================
Construct the heterogeneous DDI knowledge graph for R-GCN training.

Graph structure:
  Node types:
    - drug:    One node per unique drug across all datasets
    - target:  One node per known drug target (protein)
    - pathway: One node per known biological pathway (from DrugBank)

  Edge types:
    - (drug, interacts_with, drug):   DDI edges (primary prediction target)
    - (drug, has_target, target):     Drug→protein binding
    - (target, targeted_by, drug):    Reverse
    - (drug, shares_cyp_enzyme, drug): Shared CYP450 metabolic pathway edges
    - (drug, involved_in, pathway):   Drug→pathway

Node features — Drug nodes:
    - Molecular embedding (64-dim, initially zeros; filled after GIN pretraining)
    - ATC top-level one-hot (26 letters, A–Z)
    - Molecular descriptors: MW, LogP, TPSA, HBD, HBA, rotatable bonds (6-dim)
    - CYP450 flags: 5 enzymes × 3 roles = 15-dim binary
    - log(num_interactions + 1) (1-dim)
    - QT prolongation flag (1-dim)
    - CNS depressant flag (1-dim)
    - NSAID flag (1-dim)
    - Anticoagulant flag (1-dim)
    Total: 64 + 26 + 6 + 15 + 1 + 1 + 1 + 1 + 1 = 116-dim (embedding filled later)

Output:
  - data/graphs/ddi_hetero_graph.pt       (PyG HeteroData)
  - data/graphs/drug_id_map.parquet       (drug_id → node index)
  - data/graphs/target_id_map.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import HeteroData
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.loader import load_config  # noqa: E402

# ATC top-level code letters (A–Z, 26 classes)
ATC_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# CYP enzyme × role feature ordering
CYP_ENZYMES = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
CYP_ROLES = ["substrate", "inhibitor", "inducer"]

# Embedding dim placeholder (filled by GIN encoder during training)
EMBEDDING_DIM = 64  # Must match gin.embedding_dim in config


def _atc_one_hot(atc_level1: str, atc_codes_fallback: str = "") -> list[float]:
    """One-hot encode ATC top-level code from atc_level1 (single letter) or atc_codes text."""
    enc = [0.0] * len(ATC_LETTERS)
    # Prefer atc_level1 (pre-parsed single letter)
    letter = str(atc_level1 or "").strip().upper()
    if not letter or letter == "NAN":
        # Fall back to first character of atc_codes text
        codes_text = str(atc_codes_fallback or "")
        letter = codes_text[0].upper() if codes_text else ""
    if letter and letter in ATC_LETTERS:
        enc[ATC_LETTERS.index(letter)] = 1.0
    return enc


def _cyp_feature_vector(row: dict | pd.Series) -> list[float]:
    """Extract 15-dim CYP450 flag vector from a drug row."""
    feats = []
    is_dict = isinstance(row, dict)
    for cyp in CYP_ENZYMES:
        for role in CYP_ROLES:
            col = f"{cyp}_{role}"
            if is_dict:
                has_col = col in row
                val = row.get(col, 0.0)
            else:
                has_col = col in row.index
                val = row[col]
            feats.append(float(val) if (has_col and pd.notna(val)) else 0.0)
    return feats


def _is_qt_drug(drug_name: str, qt_drugs: set[str]) -> bool:
    return drug_name.lower().strip() in qt_drugs


def _is_cns_drug(categories: str, cns_categories: set[str]) -> bool:
    if not categories or pd.isna(categories):
        return False
    cats = set(categories.lower().split("|"))
    return bool(cats & cns_categories)


def _is_nsaid(drug_name: str, nsaid_drugs: set[str]) -> bool:
    return drug_name.lower().strip() in nsaid_drugs


def _is_anticoagulant(drug_name: str, anticoag_drugs: set[str]) -> bool:
    return drug_name.lower().strip() in anticoag_drugs


def build_drug_node_features(
    drugs_df: pd.DataFrame,
    cfg: Any,
) -> tuple[torch.Tensor, dict[str, int]]:
    """
    Build drug node feature matrix and drug_id → node_index mapping.

    Returns:
        x:          [num_drugs, feature_dim] tensor
        drug_to_idx: dict mapping drugbank_id → node index
    """
    import yaml
    from typing import Any

    # Load special drug lists from config
    qt_drugs = set(d.lower() for d in cfg.qt_prolonging_drugs)
    cns_cats = set(d.lower() for d in cfg.cns_depressant_categories)
    nsaid_drugs = set(d.lower() for d in cfg.nsaid_drugs)
    anticoag_drugs = set(d.lower() for d in cfg.anticoagulant_drugs)

    drug_to_idx: dict[str, int] = {}
    feature_rows = []

    # Count interactions per drug (log-scaled)
    # Will be filled below after building the interaction graph

    for idx, row in enumerate(drugs_df.itertuples(index=False)):
        drug_to_idx[row.drugbank_id] = idx

        # 1. Molecular embedding placeholder (64-dim zeros — filled by GIN later)
        embed = [0.0] * EMBEDDING_DIM

        # 2. ATC one-hot (26-dim) — use atc_level1 first, fall back to atc_codes text
        atc_level1 = getattr(row, "atc_level1", "") or ""
        atc_codes_fb = getattr(row, "atc_codes", "") or ""
        atc_vec = _atc_one_hot(atc_level1, atc_codes_fb)

        # 3. Molecular descriptors (6-dim, normalized)
        mw = float(getattr(row, "molecular_weight", 0) or 0) / 1000.0
        desc_vec = [
            min(mw, 2.0),  # MW (capped at 2kDa = 2000 Da)
            0.0,            # LogP placeholder (from PubChem)
            0.0,            # TPSA placeholder
            0.0,            # HBD placeholder
            0.0,            # HBA placeholder
            0.0,            # Rotatable bonds placeholder
        ]

        # 4. CYP450 flags (15-dim)
        cyp_vec = _cyp_feature_vector(row._asdict())

        # 5. Special flags (4-dim)
        name = getattr(row, "name", "")
        cats = getattr(row, "categories", "")
        qt_flag = float(_is_qt_drug(name, qt_drugs))
        cns_flag = float(_is_cns_drug(cats, cns_cats))
        nsaid_flag = float(_is_nsaid(name, nsaid_drugs))
        anticoag_flag = float(_is_anticoagulant(name, anticoag_drugs))

        # 6. Interaction count placeholder (1-dim)
        interaction_count_log = 0.0

        full_feat = (
            embed + atc_vec + desc_vec + cyp_vec
            + [qt_flag, cns_flag, nsaid_flag, anticoag_flag, interaction_count_log]
        )
        feature_rows.append(full_feat)

    x = torch.tensor(feature_rows, dtype=torch.float)
    logger.info(f"  Drug node features: {x.shape[0]:,} nodes × {x.shape[1]} dims")
    return x, drug_to_idx


def build_ddi_knowledge_graph(
    processed_dir: Path,
    graphs_dir: Path,
    cfg: Any,
) -> HeteroData:
    """
    Build the full heterogeneous DDI knowledge graph.

    Args:
        processed_dir: Path to data/processed/
        graphs_dir:    Path to data/graphs/
        cfg:           Loaded config object

    Returns:
        PyG HeteroData object
    """
    from typing import Any as _Any

    graphs_dir.mkdir(parents=True, exist_ok=True)

    # ── Load base data ─────────────────────────────────────────────────────────
    logger.info("Loading processed data files...")

    drugs_path = processed_dir / "drugs.parquet"
    if not drugs_path.exists():
        raise FileNotFoundError(
            f"drugs.parquet not found. Run pipeline/parse_drugbank.py first."
        )
    drugs_df = pd.read_parquet(drugs_path)
    logger.info(f"  Drugs: {len(drugs_df):,}")

    # Load interactions (DrugBank primary source)
    interactions_path = processed_dir / "interactions_drugbank.parquet"
    interactions_df = pd.DataFrame()
    if interactions_path.exists():
        interactions_df = pd.read_parquet(interactions_path)
        logger.info(f"  DrugBank interactions: {len(interactions_df):,}")

    # Load TWOSIDES (secondary, for severity augmentation)
    twosides_path = processed_dir / "twosides.parquet"
    twosides_df = pd.DataFrame()
    if twosides_path.exists():
        twosides_df = pd.read_parquet(twosides_path)
        logger.info(f"  TWOSIDES pairs: {len(twosides_df):,}")

    # Load FAERS harm signals
    faers_path = processed_dir / "faers_harm_signals.parquet"
    faers_df = pd.DataFrame()
    if faers_path.exists():
        faers_df = pd.read_parquet(faers_path)
        logger.info(f"  FAERS harm signals: {len(faers_df):,}")

    # Load drug targets
    targets_path = processed_dir / "drug_targets.parquet"
    targets_df = pd.DataFrame()
    if targets_path.exists():
        targets_df = pd.read_parquet(targets_path)
        logger.info(f"  Drug-target relations: {len(targets_df):,}")

    # ── Build drug node features + index ──────────────────────────────────────
    logger.info("Building drug node features...")
    drug_x, drug_to_idx = build_drug_node_features(drugs_df, cfg)
    num_drugs = len(drug_to_idx)

    # Update log(interaction_count) in features
    if len(interactions_df) > 0:
        interaction_counts = pd.concat([
            interactions_df["drug1_id"].value_counts(),
            interactions_df["drug2_id"].value_counts(),
        ]).groupby(level=0).sum()
        for drug_id, count in interaction_counts.items():
            if drug_id in drug_to_idx:
                idx = drug_to_idx[drug_id]
                drug_x[idx, -1] = float(np.log1p(count))

    # ── Build interaction edges (drug → drug) ─────────────────────────────────
    logger.info("Building drug-drug interaction edges...")
    ddi_src, ddi_dst = [], []
    ddi_severity, ddi_type_id, ddi_faers = [], [], []
    ddi_support_count = []  # number of sources supporting this edge

    # Create a merged interaction dict: (id1, id2) → {severity, faers_score, support_count}
    pair_data: dict[tuple[str, str], dict] = {}

    if len(interactions_df) > 0:
        for _, row in interactions_df.iterrows():
            id1, id2 = row["drug1_id"], row["drug2_id"]
            if id1 not in drug_to_idx or id2 not in drug_to_idx:
                continue
            key = (id1, id2)
            pair_data[key] = {
                "severity": int(row.get("severity", 1)),
                "type_id": 0,  # placeholder — filled from mechanism type
                "faers_score": 0.0,
                "support_count": 1,
                "description": str(row.get("description", "")),
            }

    # Augment with FAERS
    if len(faers_df) > 0:
        # Map FAERS names to DrugBank IDs
        name_to_id = {
            row["name"].lower(): row["drugbank_id"]
            for _, row in drugs_df.iterrows()
        }
        for _, row in faers_df.iterrows():
            id1 = name_to_id.get(str(row.get("drug1_name", "")).lower())
            id2 = name_to_id.get(str(row.get("drug2_name", "")).lower())
            if not id1 or not id2:
                continue
            if id1 > id2:
                id1, id2 = id2, id1
            key = (id1, id2)
            if key in pair_data:
                pair_data[key]["faers_score"] = float(row.get("faers_harm_score", 0.0))
                pair_data[key]["support_count"] += 1
                # Take max severity
                faers_sev = int(min(round(float(row.get("faers_harm_score", 0))), 3))
                pair_data[key]["severity"] = max(pair_data[key]["severity"], faers_sev)
            else:
                if id1 in drug_to_idx and id2 in drug_to_idx:
                    pair_data[key] = {
                        "severity": int(min(round(float(row.get("faers_harm_score", 0))), 3)),
                        "type_id": 0,
                        "faers_score": float(row.get("faers_harm_score", 0.0)),
                        "support_count": 1,
                        "description": "",
                    }

    # Build mechanism_type → int mapping from interactions_df
    if len(interactions_df) > 0 and "mechanism_type" in interactions_df.columns:
        mech_types = sorted(interactions_df["mechanism_type"].dropna().unique())
        mech_to_id = {m: i for i, m in enumerate(mech_types)}
        for _, row in interactions_df.iterrows():
            id1, id2 = row["drug1_id"], row["drug2_id"]
            key = (id1, id2)
            if key in pair_data and "mechanism_type" in row:
                pair_data[key]["type_id"] = mech_to_id.get(str(row.get("mechanism_type", "")), 0)
    else:
        mech_to_id = {}

    # Convert to tensors
    for (id1, id2), data in pair_data.items():
        i = drug_to_idx[id1]
        j = drug_to_idx[id2]
        # Add both directions
        ddi_src.extend([i, j])
        ddi_dst.extend([j, i])
        for _ in range(2):
            ddi_severity.append(data["severity"])
            ddi_type_id.append(data["type_id"])
            ddi_faers.append(data["faers_score"])
            ddi_support_count.append(data["support_count"])

    ddi_edge_index = torch.tensor([ddi_src, ddi_dst], dtype=torch.long)
    ddi_severity_t = torch.tensor(ddi_severity, dtype=torch.long)
    ddi_type_t = torch.tensor(ddi_type_id, dtype=torch.long)
    ddi_faers_t = torch.tensor(ddi_faers, dtype=torch.float)
    ddi_support_t = torch.tensor(ddi_support_count, dtype=torch.long)
    logger.info(f"  DDI edges: {ddi_edge_index.shape[1]:,} (bidirectional)")

    # ── Build CYP450 shared-enzyme edges ──────────────────────────────────────
    logger.info("Building CYP450 shared-enzyme edges...")
    cyp_src, cyp_dst, cyp_enzyme_id = [], [], []
    cyp_names = cfg.cyp450_enzymes

    # For each CYP enzyme, find all drugs that are substrates or inhibitors
    for cyp_idx, cyp in enumerate(cyp_names):
        for role in ["substrate", "inhibitor"]:  # Two drugs sharing both → strong metabolic DDI signal
            col = f"{cyp}_{role}"
            if col not in drugs_df.columns:
                continue
            cyp_drugs = drugs_df[drugs_df[col] == True]["drugbank_id"].tolist()
            cyp_drug_indices = [drug_to_idx[d] for d in cyp_drugs if d in drug_to_idx]

            # Connect all pairs that share this CYP role
            for ii in range(len(cyp_drug_indices)):
                for jj in range(ii + 1, len(cyp_drug_indices)):
                    i, j = cyp_drug_indices[ii], cyp_drug_indices[jj]
                    cyp_src.extend([i, j])
                    cyp_dst.extend([j, i])
                    cyp_enzyme_id.extend([cyp_idx, cyp_idx])

    if cyp_src:
        cyp_edge_index = torch.tensor([cyp_src, cyp_dst], dtype=torch.long)
        cyp_enzyme_t = torch.tensor(cyp_enzyme_id, dtype=torch.long)
    else:
        cyp_edge_index = torch.zeros((2, 0), dtype=torch.long)
        cyp_enzyme_t = torch.zeros(0, dtype=torch.long)
    logger.info(f"  CYP450 shared-enzyme edges: {cyp_edge_index.shape[1]:,}")

    # ── Build drug-target edges ────────────────────────────────────────────────
    target_to_idx: dict[str, int] = {}
    target_x_list = []
    tgt_src, tgt_dst = [], []

    if len(targets_df) > 0:
        logger.info("Building drug-target edges...")
        for _, row in targets_df.iterrows():
            drug_id = row["drug_id"]
            target_id = row["target_id"]
            if drug_id not in drug_to_idx:
                continue
            if target_id not in target_to_idx:
                target_to_idx[target_id] = len(target_to_idx)
                target_x_list.append([0.0])  # Placeholder target features

            tgt_src.append(drug_to_idx[drug_id])
            tgt_dst.append(target_to_idx[target_id])

    num_targets = max(len(target_to_idx), 1)
    target_x = torch.zeros((num_targets, 16), dtype=torch.float)  # 16-dim target features
    if tgt_src:
        target_edge_index = torch.tensor([tgt_src, tgt_dst], dtype=torch.long)
        target_edge_index_rev = torch.stack([target_edge_index[1], target_edge_index[0]])
    else:
        target_edge_index = torch.zeros((2, 0), dtype=torch.long)
        target_edge_index_rev = torch.zeros((2, 0), dtype=torch.long)
    logger.info(f"  Drug-target edges: {target_edge_index.shape[1]:,}")

    # ── Assemble HeteroData ────────────────────────────────────────────────────
    logger.info("Assembling HeteroData graph...")
    data = HeteroData()

    # Node features
    data["drug"].x = drug_x
    data["drug"].num_nodes = num_drugs
    data["target"].x = target_x
    data["target"].num_nodes = num_targets

    # DDI edges
    data["drug", "interacts_with", "drug"].edge_index = ddi_edge_index
    data["drug", "interacts_with", "drug"].severity = ddi_severity_t
    data["drug", "interacts_with", "drug"].interaction_type = ddi_type_t
    data["drug", "interacts_with", "drug"].faers_score = ddi_faers_t
    data["drug", "interacts_with", "drug"].support_count = ddi_support_t

    # CYP edges
    data["drug", "shares_cyp_enzyme", "drug"].edge_index = cyp_edge_index
    data["drug", "shares_cyp_enzyme", "drug"].enzyme_id = cyp_enzyme_t

    # Target edges
    data["drug", "has_target", "target"].edge_index = target_edge_index
    data["target", "targeted_by", "drug"].edge_index = target_edge_index_rev

    # Metadata
    data.drug_to_idx = drug_to_idx
    data.idx_to_drug = {v: k for k, v in drug_to_idx.items()}
    data.target_to_idx = target_to_idx
    data.mech_to_id = mech_to_id

    # Save
    graph_path = graphs_dir / "ddi_hetero_graph.pt"
    torch.save(data, graph_path)
    logger.info(f"  Saved: {graph_path}")

    # Save ID maps
    pd.DataFrame(
        [{"drug_id": k, "node_idx": v} for k, v in drug_to_idx.items()]
    ).to_parquet(graphs_dir / "drug_id_map.parquet", index=False)

    pd.DataFrame(
        [{"target_id": k, "node_idx": v} for k, v in target_to_idx.items()]
    ).to_parquet(graphs_dir / "target_id_map.parquet", index=False)

    logger.info(
        f"\nDDI Knowledge Graph Summary:\n"
        f"  Drug nodes:          {num_drugs:>10,}\n"
        f"  Target nodes:        {num_targets:>10,}\n"
        f"  DDI edges (bidir):   {ddi_edge_index.shape[1]:>10,}\n"
        f"  CYP edges (bidir):   {cyp_edge_index.shape[1]:>10,}\n"
        f"  Drug→Target edges:   {target_edge_index.shape[1]:>10,}\n"
        f"  Drug feature dim:    {drug_x.shape[1]:>10}"
    )

    return data


if __name__ == "__main__":
    cfg = load_config()
    processed_dir = ROOT / cfg.paths.data_processed
    graphs_dir = ROOT / cfg.paths.data_graphs

    data = build_ddi_knowledge_graph(
        processed_dir=processed_dir,
        graphs_dir=graphs_dir,
        cfg=cfg,
    )
    logger.success("DDI knowledge graph built successfully")
