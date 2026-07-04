"""
explainability/shapley_attribution.py
=======================================
Shapley value-based drug risk contribution for polypharmacy analysis.

For a patient taking N drugs, computes each drug's marginal contribution
to the total interaction risk score using permutation-based Shapley approximation.

The drug with the highest Shapley value is the "risk culprit" — it contributes
most to the overall interaction burden for this patient's medication profile.

Complexity: O(n_samples × N) — manageable for N ≤ 15 drugs.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class ShapleyResult:
    """Shapley value attribution results for a full drug list."""
    drug_names: list[str]
    shapley_values: list[float]           # One value per drug (contribution to total risk)
    shapley_values_normalized: list[float] # Normalized to sum to 1 (proportional contribution)
    risk_culprit: str                      # Drug with highest Shapley value
    risk_culprit_idx: int                  # Index of risk culprit
    total_risk_score: float               # Total interaction risk score
    explanation: str                       # Plain-English interpretation


def _compute_coalition_risk(
    drug_subset: list[str],
    risk_fn: Callable[[list[str]], float],
) -> float:
    """
    Compute interaction risk for a subset of drugs (coalition).

    Args:
        drug_subset: Subset of drug names
        risk_fn:     Function that takes a drug list and returns a risk score

    Returns:
        Risk score for this coalition
    """
    if len(drug_subset) < 2:
        return 0.0
    return risk_fn(drug_subset)


def shapley_drug_attribution(
    drug_names: list[str],
    risk_fn: Callable[[list[str]], float],
    n_samples: int = 200,
    seed: int = 42,
) -> ShapleyResult:
    """
    Compute approximate Shapley values for drug risk attribution.

    Uses the permutation sampling approximation (antithetic):
      For each random permutation of drugs, compute each drug's
      marginal contribution when it is added to the existing coalition.

    This gives an unbiased estimate of Shapley values with variance
    that decreases as n_samples increases.

    Args:
        drug_names: List of drug names in the patient's medication list
        risk_fn:    Function mapping list of drug names → risk score (0–100)
        n_samples:  Number of random permutations to sample (default 200)
        seed:       Random seed for reproducibility

    Returns:
        ShapleyResult with per-drug Shapley values and ranked attribution
    """
    n = len(drug_names)
    if n == 0:
        return ShapleyResult([], [], [], "", -1, 0.0, "No drugs provided.")
    if n == 1:
        return ShapleyResult(
            drug_names, [0.0], [1.0], drug_names[0], 0,
            risk_fn(drug_names),
            f"{drug_names[0]} is the only drug — no interactions possible.",
        )

    random.seed(seed)
    shapley_vals = np.zeros(n, dtype=np.float64)

    # Permutation sampling
    for _ in range(n_samples):
        perm = list(range(n))
        random.shuffle(perm)

        coalition = []
        prev_value = 0.0

        for drug_idx in perm:
            coalition.append(drug_names[drug_idx])
            curr_value = _compute_coalition_risk(coalition, risk_fn)
            marginal = curr_value - prev_value
            shapley_vals[drug_idx] += marginal
            prev_value = curr_value

    # Average over samples
    shapley_vals /= n_samples

    # Normalize to non-negative (contributions can't be negative in pure risk context)
    shapley_vals = np.maximum(shapley_vals, 0.0)

    # Compute total risk and normalize Shapley values proportionally
    total_risk = risk_fn(drug_names)
    sv_sum = shapley_vals.sum()
    if sv_sum > 0:
        shapley_norm = (shapley_vals / sv_sum).tolist()
    else:
        shapley_norm = [1.0 / n] * n  # Uniform if all zero

    # Identify risk culprit
    culprit_idx = int(np.argmax(shapley_vals))
    culprit_name = drug_names[culprit_idx]
    culprit_pct = shapley_norm[culprit_idx] * 100

    # Build explanation
    sorted_drugs = sorted(
        enumerate(drug_names), key=lambda x: shapley_vals[x[0]], reverse=True
    )
    ranked_str = ", ".join(
        f"{name} ({shapley_norm[i] * 100:.1f}%)"
        for i, name in sorted_drugs[:3]
    )
    explanation = (
        f"{culprit_name} is the primary risk contributor, responsible for "
        f"{culprit_pct:.1f}% of your total interaction risk. "
        f"Top 3 risk contributors: {ranked_str}. "
        f"Total polypharmacy risk score: {total_risk:.1f}/100."
    )

    return ShapleyResult(
        drug_names=drug_names,
        shapley_values=shapley_vals.tolist(),
        shapley_values_normalized=shapley_norm,
        risk_culprit=culprit_name,
        risk_culprit_idx=culprit_idx,
        total_risk_score=float(total_risk),
        explanation=explanation,
    )


def exact_shapley_small(
    drug_names: list[str],
    risk_fn: Callable[[list[str]], float],
) -> ShapleyResult:
    """
    Exact Shapley computation for small drug lists (N ≤ 8).

    Iterates over all 2^N - 1 non-empty subsets. Only use when N ≤ 8
    to avoid exponential blowup (2^8 = 256 subsets — manageable).

    Args:
        drug_names: ≤ 8 drug names
        risk_fn:    Risk score function

    Returns:
        ShapleyResult with exact Shapley values
    """
    n = len(drug_names)
    assert n <= 8, f"Exact Shapley is only feasible for N ≤ 8 drugs, got {n}"

    import math

    shapley_vals = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # For each drug i, iterate over all subsets NOT containing i
        others = [j for j in range(n) if j != i]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                # Marginal contribution of drug i to subset S
                S = list(subset)
                S_with_i = S + [i]

                v_S = _compute_coalition_risk([drug_names[j] for j in S], risk_fn)
                v_S_i = _compute_coalition_risk([drug_names[j] for j in S_with_i], risk_fn)

                # Shapley weight: |S|! × (n - |S| - 1)! / n!
                weight = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                shapley_vals[i] += weight * (v_S_i - v_S)

    shapley_vals = np.maximum(shapley_vals, 0.0)

    total_risk = risk_fn(drug_names)
    sv_sum = shapley_vals.sum()
    shapley_norm = (shapley_vals / sv_sum).tolist() if sv_sum > 0 else [1.0 / n] * n

    culprit_idx = int(np.argmax(shapley_vals))
    culprit_name = drug_names[culprit_idx]

    return ShapleyResult(
        drug_names=drug_names,
        shapley_values=shapley_vals.tolist(),
        shapley_values_normalized=shapley_norm,
        risk_culprit=culprit_name,
        risk_culprit_idx=culprit_idx,
        total_risk_score=float(total_risk),
        explanation=(
            f"{culprit_name} contributes {shapley_norm[culprit_idx] * 100:.1f}% "
            f"of the total interaction risk (exact Shapley computation)."
        ),
    )


def compute_drug_attribution(
    drug_names: list[str],
    risk_fn: Callable[[list[str]], float],
    n_samples: int = 200,
) -> ShapleyResult:
    """
    Adaptive Shapley computation: exact for small lists, sampled for larger.

    Args:
        drug_names: Patient's drug list (up to 15 drugs)
        risk_fn:    Risk score function
        n_samples:  MC samples for large lists

    Returns:
        ShapleyResult
    """
    n = len(drug_names)
    if n <= 1:
        total = risk_fn(drug_names) if n == 1 else 0.0
        return ShapleyResult(
            drug_names, [total], [1.0] if n == 1 else [],
            drug_names[0] if n == 1 else "",
            0 if n == 1 else -1, total,
            "Only one drug — no interactions possible." if n == 1 else "No drugs.",
        )
    elif n <= 7:
        return exact_shapley_small(drug_names, risk_fn)
    else:
        return shapley_drug_attribution(drug_names, risk_fn, n_samples=n_samples)


if __name__ == "__main__":
    # Test with a mock risk function
    def mock_risk(drugs: list[str]) -> float:
        """Mock risk: high if Warfarin is in list, moderate for combinations."""
        if len(drugs) < 2:
            return 0.0
        base = len(drugs) * 5.0
        if "Warfarin" in drugs:
            base += 20.0
        if "Aspirin" in drugs and "Warfarin" in drugs:
            base += 30.0
        return min(base, 100.0)

    test_drugs = ["Warfarin", "Aspirin", "Metformin", "Lisinopril", "Simvastatin"]
    result = compute_drug_attribution(test_drugs, mock_risk, n_samples=100)

    print("Shapley Attribution Results:")
    for name, sv, sv_norm in zip(result.drug_names, result.shapley_values, result.shapley_values_normalized):
        print(f"  {name:<20} Shapley: {sv:.2f}  ({sv_norm * 100:.1f}%)")
    print(f"\nRisk Culprit: {result.risk_culprit}")
    print(f"Total Risk:   {result.total_risk_score:.1f}")
    print(f"Explanation:  {result.explanation}")
