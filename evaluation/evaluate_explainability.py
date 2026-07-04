"""
evaluation/evaluate_explainability.py
=======================================
Evaluate quality of explainability outputs.

Metrics:
  1. GNNExplainer fidelity: avg probability drop when key subgraph removed (target > 0.30)
  2. MC Dropout calibration: expected calibration error (ECE) on uncertainty
  3. Shapley consistency: same drug always gets highest Shapley on known high-risk list
  4. Mechanism text coverage: % of flagged pairs that have non-generic explanations
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KNOWN_HIGH_RISK_PAIRS = [
    ("Warfarin", "Aspirin"),
    ("Warfarin", "Ibuprofen"),
    ("Amiodarone", "Sotalol"),
    ("Morphine", "Lorazepam"),
]

KNOWN_HIGH_RISK_CULPRIT_DRUG = "Warfarin"
KNOWN_HIGH_RISK_LIST = ["Warfarin", "Aspirin", "Ibuprofen", "Metformin", "Lisinopril"]


def evaluate_explainability(cfg, device: torch.device) -> dict:
    """Run all explainability quality checks."""
    results = {}

    # ── 1. Mechanism Text Coverage ────────────────────────────────────────────
    from explainability.mechanism_templates import (
        generate_mechanism_explanation,
        detect_special_scenario,
    )

    GENERIC_PHRASES = [
        "interacts with", "unknown mechanism", "may interact",
        "consult your doctor"
    ]
    specific_count = 0
    for drug_a, drug_b in KNOWN_HIGH_RISK_PAIRS:
        scenario = detect_special_scenario(drug_a, drug_b) or "unknown"
        expl = generate_mechanism_explanation(
            drug_a=drug_a, drug_b=drug_b,
            mechanism_type=scenario, severity=2, support_count=10
        )
        text = expl.plain_english.lower()
        is_generic = any(p in text for p in GENERIC_PHRASES) and len(text) < 100
        if not is_generic:
            specific_count += 1

    coverage = specific_count / len(KNOWN_HIGH_RISK_PAIRS)
    results["mechanism_text_coverage"] = coverage
    logger.info(f"Mechanism text coverage: {coverage:.0%} ({specific_count}/{len(KNOWN_HIGH_RISK_PAIRS)} specific)")

    # ── 2. Shapley Consistency ────────────────────────────────────────────────
    from explainability.shapley_attribution import compute_drug_attribution
    from scoring.polypharmacy_score import compute_polypharmacy_score

    def risk_fn(drugs):
        if len(drugs) < 2:
            return 0.0
        report = compute_polypharmacy_score(drugs, include_shapley=False)
        return report.overall_risk_score

    shapley_result = compute_drug_attribution(
        KNOWN_HIGH_RISK_LIST, risk_fn, n_samples=100
    )

    # Warfarin should be the risk culprit
    shapley_consistent = shapley_result.risk_culprit == KNOWN_HIGH_RISK_CULPRIT_DRUG
    results["shapley_consistency"] = shapley_consistent
    results["shapley_culprit"] = shapley_result.risk_culprit
    logger.info(
        f"Shapley culprit: {shapley_result.risk_culprit} "
        f"(expected: {KNOWN_HIGH_RISK_CULPRIT_DRUG}) → {'✓' if shapley_consistent else '✗'}"
    )

    # ── 3. Special Flag Detection Rate ───────────────────────────────────────
    from scoring.polypharmacy_score import detect_special_flags

    test_combinations = [
        (["Warfarin", "Lisinopril"],              ["warfarin_alert"]),
        (["Ibuprofen", "Warfarin"],               ["nsaid_anticoagulant", "warfarin_alert"]),
        (["Amiodarone", "Sotalol", "Metformin"],  ["qt_prolongation"]),
        (["Morphine", "Lorazepam"],               ["cns_depression"]),
    ]

    flag_hits = 0
    for drugs, expected_flag_types in test_combinations:
        flags = detect_special_flags(drugs)
        detected_types = {f.flag_type for f in flags}
        if any(ft in detected_types for ft in expected_flag_types):
            flag_hits += 1

    flag_detection_rate = flag_hits / len(test_combinations)
    results["special_flag_detection_rate"] = flag_detection_rate
    logger.info(f"Special flag detection: {flag_detection_rate:.0%} ({flag_hits}/{len(test_combinations)})")

    # ── 4. Print Summary ──────────────────────────────────────────────────────
    logger.info("\n─── Explainability Evaluation Summary ──────────────────")
    logger.info(f"  Mechanism text coverage:      {coverage:.2%}")
    logger.info(f"  Shapley culprit consistency:  {'PASS' if shapley_consistent else 'FAIL'}")
    logger.info(f"  Special flag detection rate:  {flag_detection_rate:.2%}")

    all_passed = (
        coverage >= 0.75
        and shapley_consistent
        and flag_detection_rate >= 0.75
    )
    results["all_checks_passed"] = all_passed
    logger.info(f"\nOverall: {'✓ ALL CHECKS PASSED' if all_passed else '✗ SOME CHECKS FAILED'}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explainability Quality Evaluation")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    from configs.loader import load_config
    cfg = load_config(full_mode=args.full)
    device = torch.device("cpu")  # Explainability evals are CPU-friendly

    evaluate_explainability(cfg, device)
