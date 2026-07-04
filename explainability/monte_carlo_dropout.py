"""
explainability/monte_carlo_dropout.py
======================================
Uncertainty quantification for DDI predictions using Monte Carlo Dropout.

Method:
  - Run model inference N=50 times with dropout ENABLED (even during eval)
  - Report mean ± std of predicted interaction probability
  - Classify confidence: high (std < 0.1), medium (0.1-0.25), low (> 0.25)
  - Low-confidence + sparse data → "Limited data" warning in UI
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class UncertaintyEstimate:
    """MC Dropout uncertainty estimate for a single drug pair."""
    mean_prob: float                # Mean predicted interaction probability
    std_prob: float                 # Std dev across MC samples
    confidence_level: str           # "high", "medium", "low"
    confidence_score: float         # 0.0 (uncertain) to 1.0 (confident)
    severity_mean: list[float]      # Mean severity distribution [p0, p1, p2, p3]
    severity_std: list[float]       # Std dev of severity distribution
    predicted_severity: int         # Argmax of severity_mean
    low_data_warning: bool          # True if support_count < 5
    warning_message: str            # Human-readable warning (if any)


def classify_confidence(std: float, low_data: bool = False) -> tuple[str, float]:
    """
    Classify uncertainty level and compute confidence score.

    Args:
        std:      Standard deviation across MC samples
        low_data: Whether the drug pair has < 5 supporting records

    Returns:
        (confidence_level, confidence_score)
    """
    if low_data:
        return "low", max(0.1, 0.5 - std * 2)

    if std < 0.10:
        level = "high"
        score = 1.0 - std * 5  # 0.10 std → 0.5 score, 0 std → 1.0
    elif std < 0.25:
        level = "medium"
        score = 0.75 - (std - 0.10) * 2
    else:
        level = "low"
        score = max(0.1, 0.5 - std)

    return level, float(np.clip(score, 0.0, 1.0))


def mc_dropout_predict(
    model: torch.nn.Module,
    drug_x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    drug_i_idx: int,
    drug_j_idx: int,
    target_x: Tensor | None = None,
    n_samples: int = 50,
    device: str = "cpu",
    support_count: int = 5,
) -> UncertaintyEstimate:
    """
    Run MC Dropout inference for a single drug pair.

    Dropout is kept active during inference to sample from the
    approximate posterior distribution over model parameters.

    Args:
        model:         Trained RGCNDDIPredictor (dropout must be present)
        drug_x:        Drug node features [num_drugs, feature_dim]
        edge_index:    [2, num_edges] combined edge index
        edge_type:     [num_edges] edge type indices
        drug_i_idx:    Node index of drug 1
        drug_j_idx:    Node index of drug 2
        target_x:      Optional target node features
        n_samples:     Number of MC Dropout samples (default 50)
        device:        "cuda" or "cpu"
        support_count: Number of data sources supporting this interaction

    Returns:
        UncertaintyEstimate with mean, std, and confidence classification
    """
    # Enable dropout even at eval time (this IS the MC Dropout mechanism)
    model.train()  # This activates dropout

    pair_src = torch.tensor([drug_i_idx], device=drug_x.device)
    pair_dst = torch.tensor([drug_j_idx], device=drug_x.device)

    binary_samples: list[float] = []
    severity_samples: list[list[float]] = []

    with torch.no_grad():
        for _ in range(n_samples):
            binary, severity, _, _ = model(
                drug_x, edge_index, edge_type, pair_src, pair_dst, target_x
            )
            prob = torch.sigmoid(binary).item()
            sev_probs = torch.softmax(severity, dim=-1).squeeze().tolist()

            binary_samples.append(prob)
            severity_samples.append(sev_probs if isinstance(sev_probs, list) else [sev_probs])

    model.eval()  # Restore eval mode

    # Compute statistics
    binary_arr = np.array(binary_samples)
    mean_prob = float(binary_arr.mean())
    std_prob = float(binary_arr.std())

    # Severity statistics (handle case where severity output is scalar)
    if severity_samples and len(severity_samples[0]) > 1:
        sev_arr = np.array(severity_samples)  # [n_samples, 4]
        severity_mean = sev_arr.mean(axis=0).tolist()
        severity_std = sev_arr.std(axis=0).tolist()
        predicted_severity = int(np.argmax(severity_mean))
    else:
        severity_mean = [0.0, 0.0, 0.0, 0.0]
        severity_std = [0.0, 0.0, 0.0, 0.0]
        predicted_severity = 0

    low_data = support_count < 5
    confidence_level, confidence_score = classify_confidence(std_prob, low_data)

    # Build warning message
    if low_data:
        warning = (
            f"⚠️  Limited data: Only {support_count} record(s) support this interaction. "
            "Interaction status is uncertain. Consult a pharmacist before making decisions."
        )
    elif confidence_level == "low":
        warning = (
            "⚠️  High uncertainty detected in this prediction. "
            "The model has inconsistent predictions for this drug pair. "
            "Consult a pharmacist for personalized guidance."
        )
    elif confidence_level == "medium":
        warning = (
            "ℹ️  Moderate confidence. This prediction is based on limited but consistent evidence."
        )
    else:
        warning = ""

    return UncertaintyEstimate(
        mean_prob=mean_prob,
        std_prob=std_prob,
        confidence_level=confidence_level,
        confidence_score=confidence_score,
        severity_mean=severity_mean,
        severity_std=severity_std,
        predicted_severity=predicted_severity,
        low_data_warning=low_data,
        warning_message=warning,
    )


def batch_mc_dropout_predict(
    model: torch.nn.Module,
    drug_x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    drug_pairs: list[tuple[int, int]],
    target_x: Tensor | None = None,
    n_samples: int = 50,
    support_counts: list[int] | None = None,
) -> list[UncertaintyEstimate]:
    """
    Run MC Dropout for multiple drug pairs efficiently.

    Instead of N passes per pair, runs N forward passes for all pairs
    simultaneously — more VRAM-efficient for batched inference.

    Args:
        model:          Trained RGCNDDIPredictor
        drug_x:         Drug node features
        edge_index:     Combined edge index
        edge_type:      Edge type indices
        drug_pairs:     List of (drug_i_idx, drug_j_idx) pairs
        n_samples:      MC Dropout samples
        support_counts: Support count per pair (for confidence)

    Returns:
        List of UncertaintyEstimate, one per input pair
    """
    if not drug_pairs:
        return []

    if support_counts is None:
        support_counts = [5] * len(drug_pairs)

    model.train()  # Enable dropout

    pair_src = torch.tensor([p[0] for p in drug_pairs], device=drug_x.device)
    pair_dst = torch.tensor([p[1] for p in drug_pairs], device=drug_x.device)

    # Collect samples: [n_samples, n_pairs]
    all_binary = []
    all_severity = []

    with torch.no_grad():
        for _ in range(n_samples):
            binary, severity, _, _ = model(
                drug_x, edge_index, edge_type, pair_src, pair_dst, target_x
            )
            all_binary.append(torch.sigmoid(binary).squeeze().cpu())
            all_severity.append(torch.softmax(severity, dim=-1).cpu())

    model.eval()

    binary_tensor = torch.stack(all_binary, dim=0)    # [n_samples, n_pairs]
    severity_tensor = torch.stack(all_severity, dim=0)  # [n_samples, n_pairs, 4]

    results = []
    for i in range(len(drug_pairs)):
        b_arr = binary_tensor[:, i].numpy()
        s_arr = severity_tensor[:, i].numpy()  # [n_samples, 4]

        mean_prob = float(b_arr.mean())
        std_prob = float(b_arr.std())
        sev_mean = s_arr.mean(axis=0).tolist()
        sev_std = s_arr.std(axis=0).tolist()
        pred_sev = int(np.argmax(sev_mean))

        low_data = support_counts[i] < 5
        conf_level, conf_score = classify_confidence(std_prob, low_data)

        if low_data:
            warning = (
                f"⚠️  Limited data ({support_counts[i]} records). Consult a pharmacist."
            )
        elif conf_level == "low":
            warning = "⚠️  High prediction uncertainty. Consult a pharmacist."
        else:
            warning = ""

        results.append(
            UncertaintyEstimate(
                mean_prob=mean_prob,
                std_prob=std_prob,
                confidence_level=conf_level,
                confidence_score=conf_score,
                severity_mean=sev_mean,
                severity_std=sev_std,
                predicted_severity=pred_sev,
                low_data_warning=low_data,
                warning_message=warning,
            )
        )

    return results
