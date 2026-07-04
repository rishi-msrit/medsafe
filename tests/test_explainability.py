"""
tests/test_explainability.py
==============================
Tests for explainability modules: GNNExplainer, MC Dropout, Shapley, mechanism templates.

5 tests covering output structure, value bounds, and text quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─── Test 1: MC Dropout Returns Valid UncertaintyEstimate ────────────────────

def test_mc_dropout_returns_valid_estimate(rgcn_model, small_ddi_graph):
    """MC Dropout should return a valid UncertaintyEstimate with bounded values."""
    from training.finetune_rgcn import build_combined_edge_index
    from explainability.monte_carlo_dropout import mc_dropout_predict

    device = torch.device("cpu")
    drug_x = small_ddi_graph["drug"].x
    edge_index, edge_type = build_combined_edge_index(small_ddi_graph, device)

    result = mc_dropout_predict(
        model=rgcn_model,
        drug_x=drug_x,
        edge_index=edge_index,
        edge_type=edge_type,
        drug_i_idx=0,
        drug_j_idx=5,
        n_samples=10,  # Use fewer samples for speed in tests
        support_count=8,
    )

    assert 0.0 <= result.mean_prob <= 1.0, "Mean probability must be in [0, 1]"
    assert result.std_prob >= 0.0, "Std dev cannot be negative"
    assert result.confidence_level in ("high", "medium", "low")
    assert 0.0 <= result.confidence_score <= 1.0
    assert len(result.severity_mean) == 4, "Should have 4 severity probs"
    assert result.low_data_warning is False  # support_count=8 > 5


# ─── Test 2: MC Dropout Low-Data Warning ─────────────────────────────────────

def test_mc_dropout_low_data_warning(rgcn_model, small_ddi_graph):
    """When support_count < 5, low_data_warning should be True."""
    from training.finetune_rgcn import build_combined_edge_index
    from explainability.monte_carlo_dropout import mc_dropout_predict

    device = torch.device("cpu")
    drug_x = small_ddi_graph["drug"].x
    edge_index, edge_type = build_combined_edge_index(small_ddi_graph, device)

    result = mc_dropout_predict(
        model=rgcn_model,
        drug_x=drug_x,
        edge_index=edge_index,
        edge_type=edge_type,
        drug_i_idx=1,
        drug_j_idx=6,
        n_samples=5,
        support_count=2,  # Below threshold
    )

    assert result.low_data_warning is True
    assert result.confidence_level == "low"
    assert len(result.warning_message) > 0


# ─── Test 3: Batch MC Dropout Same Length As Input ───────────────────────────

def test_batch_mc_dropout_output_length(rgcn_model, small_ddi_graph):
    """batch_mc_dropout_predict should return one result per input pair."""
    from training.finetune_rgcn import build_combined_edge_index
    from explainability.monte_carlo_dropout import batch_mc_dropout_predict

    device = torch.device("cpu")
    drug_x = small_ddi_graph["drug"].x
    edge_index, edge_type = build_combined_edge_index(small_ddi_graph, device)

    pairs = [(0, 5), (1, 6), (2, 7)]
    results = batch_mc_dropout_predict(
        model=rgcn_model,
        drug_x=drug_x,
        edge_index=edge_index,
        edge_type=edge_type,
        drug_pairs=pairs,
        n_samples=5,
    )

    assert len(results) == len(pairs), "Should return one result per pair"
    for r in results:
        assert 0.0 <= r.mean_prob <= 1.0


# ─── Test 4: Shapley Exact Computation For Small N ───────────────────────────

def test_shapley_exact_small_list():
    """Exact Shapley for ≤7 drugs should produce correct structure."""
    from explainability.shapley_attribution import exact_shapley_small

    drugs = ["Warfarin", "Aspirin", "Metformin"]

    def risk_fn(d):
        if len(d) < 2:
            return 0.0
        return float(len(d) * 10 + (20 if "Warfarin" in d else 0))

    result = exact_shapley_small(drugs, risk_fn)

    assert len(result.shapley_values) == 3
    assert result.risk_culprit == "Warfarin", "Warfarin should be risk culprit"
    assert all(v >= 0 for v in result.shapley_values)
    assert abs(sum(result.shapley_values_normalized) - 1.0) < 1e-4, \
        "Normalized Shapley values should sum to 1"


# ─── Test 5: Mechanism Explanation QT Scenario ───────────────────────────────

def test_mechanism_template_qt_scenario():
    """QT prolongation scenario should produce a cardiology-specific warning."""
    from explainability.mechanism_templates import (
        generate_mechanism_explanation,
        detect_special_scenario,
    )

    scenario = detect_special_scenario("Amiodarone", "Sotalol")
    assert scenario == "cardiac_qt"

    result = generate_mechanism_explanation(
        drug_a="Amiodarone",
        drug_b="Sotalol",
        mechanism_type="cardiac_qt",
        severity=3,
        support_count=20,
    )

    assert "QT" in result.plain_english or "cardiac" in result.plain_english.lower(), \
        "QT explanation should mention QT or cardiac"
    assert result.severity_label == "contraindicated"
    assert result.is_special_flag is True
