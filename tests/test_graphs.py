"""
tests/test_graphs.py
=====================
Tests for DDI graph construction and structure validity.

6 tests:
  1. Graph has drug and target node types
  2. Edge index is valid (no self-loops in ddi edges by default)
  3. Severity labels are in range [0, 3]
  4. Support count attributes are non-negative
  5. CYP enzyme edge attributes contain valid enzyme indices
  6. Combined edge index from build_combined_edge_index has correct dtype
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─── Test 1: Graph Has Correct Node Types ────────────────────────────────────

def test_graph_has_drug_and_target_nodes(small_ddi_graph):
    """DDI graph must have both drug and target node types."""
    assert "drug" in small_ddi_graph.node_types, "Graph must have 'drug' nodes"
    assert "target" in small_ddi_graph.node_types, "Graph must have 'target' nodes"


# ─── Test 2: Drug Node Features Correct Shape ────────────────────────────────

def test_drug_node_features_shape(small_ddi_graph):
    """Drug node feature matrix should be [num_drugs, 116]."""
    x = small_ddi_graph["drug"].x
    assert x.dim() == 2, "Drug features should be 2D tensor"
    assert x.shape[1] == 116, f"Expected 116 features, got {x.shape[1]}"
    assert x.dtype == torch.float32, "Drug features should be float32"


# ─── Test 3: Edge Index Valid Shape ──────────────────────────────────────────

def test_ddi_edge_index_valid(small_ddi_graph):
    """DDI edge index should have shape [2, num_edges] with valid node indices."""
    ei = small_ddi_graph["drug", "interacts_with", "drug"].edge_index
    assert ei.shape[0] == 2, "Edge index first dim must be 2"
    num_drugs = small_ddi_graph["drug"].num_nodes

    assert ei.min() >= 0, "Edge indices cannot be negative"
    assert ei.max() < num_drugs, f"Edge index {ei.max()} exceeds num_drugs={num_drugs}"
    assert ei.dtype == torch.long, "Edge index must be int64"


# ─── Test 4: Severity Labels In Range ────────────────────────────────────────

def test_severity_labels_in_range(small_ddi_graph):
    """Severity labels must be in {0, 1, 2, 3}."""
    sev = small_ddi_graph["drug", "interacts_with", "drug"].severity
    assert sev.min() >= 0, "Severity cannot be negative"
    assert sev.max() <= 3, f"Severity max {sev.max()} exceeds 3 (contraindicated)"
    assert sev.dtype == torch.long, "Severity must be int64"


# ─── Test 5: CYP Enzyme Edge Attributes ──────────────────────────────────────

def test_cyp_enzyme_edges_exist(small_ddi_graph):
    """CYP enzyme-sharing edges should exist with valid enzyme IDs."""
    cyp_ei = small_ddi_graph["drug", "shares_cyp_enzyme", "drug"].edge_index
    assert cyp_ei.shape[0] == 2
    assert cyp_ei.shape[1] > 0, "Should have at least one CYP-sharing edge"

    enzyme_id = small_ddi_graph["drug", "shares_cyp_enzyme", "drug"].enzyme_id
    assert enzyme_id.min() >= 0
    assert enzyme_id.max() < 5, "Only 5 CYP450 enzymes defined (0–4)"


# ─── Test 6: Combined Edge Index Dtype ───────────────────────────────────────

def test_combined_edge_index_dtype(small_ddi_graph):
    """build_combined_edge_index must return long tensors."""
    from training.finetune_rgcn import build_combined_edge_index

    edge_index, edge_type = build_combined_edge_index(small_ddi_graph, torch.device("cpu"))

    assert edge_index.dtype == torch.long, "Edge index must be int64"
    assert edge_type.dtype == torch.long, "Edge type must be int64"
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_type.shape[0], "Edge index and type must have same length"


# ─── Test 7: Train/Val/Test Split Non-Overlapping ────────────────────────────

def test_train_val_test_splits_non_overlapping(small_ddi_graph):
    """Training, validation, and test sets should not share positive edges."""
    from training.finetune_rgcn import prepare_training_data

    drug_to_idx = getattr(small_ddi_graph, "drug_to_idx", {})
    splits = prepare_training_data(small_ddi_graph, {}, drug_to_idx)

    train_pos = set(map(tuple, splits["train"]["pos"].tolist()))
    val_pos   = set(map(tuple, splits["val"]["pos"].tolist()))
    test_pos  = set(map(tuple, splits["test"]["pos"].tolist()))

    overlap_tv = train_pos & val_pos
    overlap_tt = train_pos & test_pos
    overlap_vt = val_pos & test_pos

    assert len(overlap_tv) == 0, f"Train-val overlap: {len(overlap_tv)} edges"
    assert len(overlap_tt) == 0, f"Train-test overlap: {len(overlap_tt)} edges"
    assert len(overlap_vt) == 0, f"Val-test overlap: {len(overlap_vt)} edges"
