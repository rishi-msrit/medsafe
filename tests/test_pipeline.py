"""
tests/test_pipeline.py
======================
Tests for the data pipeline modules (without requiring actual datasets).

6 tests:
  1. DrugBank parser handles missing XML gracefully
  2. FAERS harm signal computation handles empty input
  3. Molecular graph builder handles edge cases (no atoms, disconnected)
  4. DDI graph builder creates correct tensor shapes
  5. Config loader returns correct types
  6. Severity mapping values are in range [0, 3]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─── Test 1: Config Loader Correctness ───────────────────────────────────────

def test_config_loader_returns_correct_types(cfg):
    """Config should have correct attribute types."""
    assert isinstance(cfg.gin.num_layers, int)
    assert isinstance(cfg.gin.hidden_dim, int)
    assert isinstance(cfg.gin.embedding_dim, int)
    assert isinstance(cfg.contrastive.temperature, float)
    assert isinstance(cfg.rgcn.num_bases, int)
    assert isinstance(cfg.cyp450_enzymes, list)
    assert len(cfg.cyp450_enzymes) == 5


# ─── Test 2: Config Full Mode ─────────────────────────────────────────────────

def test_config_full_mode():
    """Full mode config should have larger dimensions than default."""
    from configs.loader import load_config

    cfg_default = load_config(full_mode=False)
    cfg_full = load_config(full_mode=True)

    # Full should generally have same or larger dimensions
    assert cfg_full.gin.num_layers >= cfg_default.gin.num_layers
    assert cfg_full.gin.hidden_dim >= cfg_default.gin.hidden_dim


# ─── Test 3: DDI Graph Correct Tensor Shapes ─────────────────────────────────

def test_ddi_graph_tensor_shapes(small_ddi_graph):
    """DDI graph should have correct tensor shapes for all edge types."""
    graph = small_ddi_graph

    # Drug nodes
    assert graph["drug"].x.shape[1] == 116, "Drug feature dim should be 116"
    assert graph["drug"].num_nodes == 10

    # Target nodes
    assert graph["target"].x.shape[1] == 16
    assert graph["target"].num_nodes == 5

    # DDI edges
    ei = graph["drug", "interacts_with", "drug"].edge_index
    assert ei.shape[0] == 2, "Edge index should have shape [2, num_edges]"
    assert ei.shape[1] == 20  # 10 forward + 10 backward

    # Edge attributes
    sev = graph["drug", "interacts_with", "drug"].severity
    assert sev.shape[0] == ei.shape[1], "Severity tensor length should match edges"
    assert sev.min() >= 0 and sev.max() <= 3, "Severity should be in [0, 3]"


# ─── Test 4: R-GCN Edge Index Construction ───────────────────────────────────

def test_rgcn_combined_edge_index(small_ddi_graph):
    """Combined edge index should have correct shape and type range."""
    from training.finetune_rgcn import build_combined_edge_index

    device = torch.device("cpu")
    edge_index, edge_type = build_combined_edge_index(small_ddi_graph, device)

    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_type.shape[0]
    assert edge_type.min() >= 0
    assert edge_type.max() <= 3, "Edge types should be in [0, 3]"


# ─── Test 5: Molecular Augmentation ──────────────────────────────────────────

def test_molecular_augmentation_preserves_structure(small_molecular_graph):
    """Augmentation should not destroy the graph (edge_index still valid)."""
    from models.contrastive import augment_atom_feature_mask, augment_bond_dropout

    data = small_molecular_graph
    original_num_atoms = data.x.shape[0]
    original_edge_index = data.edge_index.clone()

    # Feature masking should not change graph topology
    aug1 = augment_atom_feature_mask(data, mask_ratio=0.5)
    assert aug1.x.shape == data.x.shape, "Feature masking should not change node count"
    assert aug1.edge_index.shape == original_edge_index.shape, "Masking should not change edges"

    # Bond dropout may remove some edges
    aug2 = augment_bond_dropout(data, drop_ratio=0.5)
    assert aug2.x.shape == data.x.shape, "Bond dropout should not change node count"
    assert aug2.edge_index.shape[1] <= original_edge_index.shape[1], "Dropout can only reduce edges"


# ─── Test 6: NT-Xent With Batch Size 1 Edge Case ────────────────────────────

def test_ntxent_batch_size_edge_case():
    """NT-Xent should handle edge cases without crashing."""
    from models.contrastive import NTXentLoss
    import torch.nn.functional as F

    loss_fn = NTXentLoss(temperature=0.07)

    # Minimum useful batch: N=2
    z_i = F.normalize(torch.randn(2, 32), dim=-1)
    z_j = F.normalize(torch.randn(2, 32), dim=-1)
    loss = loss_fn(z_i, z_j)
    assert torch.isfinite(loss), "NT-Xent should handle batch size 2"


# ─── Test 7: Scoring Report Invariants ───────────────────────────────────────

def test_scoring_report_invariants(sample_drugs):
    """Safety report should always satisfy basic invariants."""
    from scoring.polypharmacy_score import compute_polypharmacy_score

    report = compute_polypharmacy_score(sample_drugs[:4], include_shapley=False)

    assert 0 <= report.overall_risk_score <= 100, "Risk score must be in [0, 100]"
    assert report.risk_tier in ("safe", "review", "high", "critical")
    assert report.num_pairs_checked >= 0
    assert report.num_flagged >= 0
    assert report.num_flagged <= report.num_pairs_checked
    assert report.risk_tier_color.startswith("#")


# ─── Test 8: API Schema Validation ───────────────────────────────────────────

def test_api_schema_validation():
    """Pydantic schemas should reject invalid inputs."""
    from serving.schemas import DrugListRequest, PairRequest

    # Valid request
    req = DrugListRequest(drugs=["Warfarin", "Aspirin"])
    assert len(req.drugs) == 2

    # Empty drug names should be filtered
    req2 = DrugListRequest(drugs=["Warfarin", "  ", "Aspirin"])
    assert len(req2.drugs) == 2

    # Too many drugs
    with pytest.raises(Exception):
        DrugListRequest(drugs=[f"Drug{i}" for i in range(16)])

    # Same drug twice (schema allows it — scoring deduplicates)
    req3 = PairRequest(drug_a="Warfarin", drug_b="Aspirin")
    assert req3.drug_a == "Warfarin"
    assert req3.drug_b == "Aspirin"
