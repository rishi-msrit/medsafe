"""
evaluation/evaluate_severity.py
================================
Evaluate the severity classification head of the R-GCN.

Computes:
  - Per-class precision, recall, F1
  - Confusion matrix
  - Weighted and macro F1
  - Calibration: expected vs actual severity rates

Clinical utility framing:
  - Contraindicated recall is the most clinically important metric
    (missing a contraindicated pair is a patient safety failure)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEVERITY_NAMES = ["Minor", "Moderate", "Major", "Contraindicated"]


def run_severity_eval(checkpoint_path: Path, cfg, device: torch.device) -> dict:
    """Evaluate severity classification head on test split."""
    from models.rgcn_predictor import build_rgcn_predictor
    from training.finetune_rgcn import (
        build_combined_edge_index,
        prepare_training_data,
    )

    graph_path = ROOT / cfg.paths.data_graphs / "ddi_hetero_graph.pt"
    if not graph_path.exists():
        logger.error("DDI graph not found")
        return {}

    ddi_graph = torch.load(graph_path, map_location=device, weights_only=False)
    drug_x = ddi_graph["drug"].x.to(device)
    model = build_rgcn_predictor(cfg, drug_x.shape[1]).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    edge_index, edge_type = build_combined_edge_index(ddi_graph, device)
    drug_to_idx = getattr(ddi_graph, "drug_to_idx", {})
    splits = prepare_training_data(ddi_graph, {}, drug_to_idx)
    test_data = splits["test"]

    pos_e = test_data["pos"].to(device)
    sev_true = test_data["severity"].numpy()

    with torch.no_grad():
        _, severity_logits, _, _ = model(
            drug_x, edge_index, edge_type, pos_e[:, 0], pos_e[:, 1]
        )
    sev_pred = severity_logits.argmax(dim=-1).cpu().numpy()

    # Metrics
    report = classification_report(
        sev_true, sev_pred,
        target_names=SEVERITY_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(sev_true, sev_pred)
    macro_f1 = f1_score(sev_true, sev_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(sev_true, sev_pred, average="weighted", zero_division=0)

    # Print results
    logger.info("\n─── Severity Classification Results ────────────────────")
    logger.info(classification_report(
        sev_true, sev_pred, target_names=SEVERITY_NAMES, zero_division=0
    ))
    logger.info(f"Macro F1:    {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")

    logger.info("\nConfusion Matrix (row=true, col=pred):")
    header = "        " + "  ".join(f"{n[:4]:>6}" for n in SEVERITY_NAMES)
    logger.info(header)
    for i, row in enumerate(cm):
        logger.info(f"{SEVERITY_NAMES[i][:12]:<12}" + "  ".join(f"{v:>6}" for v in row))

    # Clinical focus: contraindicated recall
    contra_recall = report.get("Contraindicated", {}).get("recall", 0.0)
    logger.info(f"\n⚕️  Contraindicated Recall: {contra_recall:.4f}")
    if contra_recall < 0.80:
        logger.warning("⚠️  Contraindicated recall < 0.80 — patient safety concern")

    results = {
        "timestamp": datetime.now().isoformat(),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "contraindicated_recall": float(contra_recall),
        "per_class": {
            k: {m: float(v) for m, v in v.items() if isinstance(v, float)}
            for k, v in report.items()
            if k in SEVERITY_NAMES
        },
        "confusion_matrix": cm.tolist(),
    }

    out_dir = ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"severity_eval_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Severity Classification Evaluation")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    from configs.loader import load_config
    cfg = load_config(full_mode=args.full)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.checkpoint or (ROOT / cfg.paths.checkpoints / "rgcn_finetune" / "rgcn_best.pt")

    if not ckpt.exists():
        logger.error(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    run_severity_eval(ckpt, cfg, device)
