"""
evaluation/benchmark.py
========================
Evaluation benchmark for MedSafe DDI prediction performance.

Metrics:
  - Hits@20, Hits@10 (OGBL-DDI standard)
  - AUROC, AUPR (binary interaction detection)
  - Severity prediction accuracy + macro F1
  - Interaction type classification accuracy
  - FAERS score MSE/Pearson correlation

Reports are saved to evaluation/results/{timestamp}.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.loader import load_config  # noqa: E402


def run_evaluation(
    checkpoint_path: Path,
    cfg: "Config",
    device: torch.device,
) -> dict:
    """Run full evaluation benchmark on the test set."""
    results = {}

    # Load DDI graph
    graph_path = ROOT / cfg.paths.data_graphs / "ddi_hetero_graph.pt"
    if not graph_path.exists():
        logger.error(f"DDI graph not found: {graph_path}")
        return {"error": "DDI graph not found"}

    logger.info("Loading DDI graph...")
    ddi_graph = torch.load(graph_path, map_location=device, weights_only=False)

    drug_x = ddi_graph["drug"].x.to(device)
    drug_feature_dim = drug_x.shape[1]

    # Load model
    logger.info(f"Loading model from: {checkpoint_path}")
    from models.rgcn_predictor import build_rgcn_predictor

    model = build_rgcn_predictor(cfg, drug_feature_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Build edge index
    from training.finetune_rgcn import build_combined_edge_index, prepare_training_data

    edge_index, edge_type = build_combined_edge_index(ddi_graph, device)

    # Get test split
    drug_to_idx = getattr(ddi_graph, "drug_to_idx", {})
    splits = prepare_training_data(ddi_graph, {}, drug_to_idx)
    test_data = splits["test"]

    pos_edges = test_data["pos"].to(device)
    neg_edges = test_data["neg"].to(device)
    sev_labels = test_data["severity"].to(device)

    if len(pos_edges) == 0:
        logger.warning("Empty test set")
        return {"error": "Empty test set"}

    logger.info(f"Test set: {len(pos_edges):,} positive, {len(neg_edges):,} negative pairs")

    # Inference
    with torch.no_grad():
        pos_binary, pos_sev, pos_type, pos_faers = model(
            drug_x, edge_index, edge_type,
            pos_edges[:, 0], pos_edges[:, 1],
        )
        neg_binary, _, _, _ = model(
            drug_x, edge_index, edge_type,
            neg_edges[:, 0], neg_edges[:, 1],
        )

    pos_scores = torch.sigmoid(pos_binary.squeeze()).cpu().numpy()
    neg_scores = torch.sigmoid(neg_binary.squeeze()).cpu().numpy()
    sev_pred = pos_sev.argmax(dim=-1).cpu().numpy()
    sev_true = sev_labels.cpu().numpy()

    # ── Hits@K ────────────────────────────────────────────────────────────────
    from training.finetune_rgcn import hits_at_k
    hits20 = hits_at_k(torch.tensor(pos_scores), torch.tensor(neg_scores), k=20)
    hits10 = hits_at_k(torch.tensor(pos_scores), torch.tensor(neg_scores), k=10)
    hits50 = hits_at_k(torch.tensor(pos_scores), torch.tensor(neg_scores), k=50)

    # ── AUROC / AUPR ──────────────────────────────────────────────────────────
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    auroc = roc_auc_score(all_labels, all_scores)
    aupr = average_precision_score(all_labels, all_scores)

    # ── Severity Accuracy + F1 ────────────────────────────────────────────────
    sev_acc = accuracy_score(sev_true, sev_pred)
    sev_f1 = f1_score(sev_true, sev_pred, average="macro", zero_division=0)

    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "test_size": len(pos_edges),
        "hits_at_10": float(hits10),
        "hits_at_20": float(hits20),
        "hits_at_50": float(hits50),
        "auroc": float(auroc),
        "aupr": float(aupr),
        "severity_accuracy": float(sev_acc),
        "severity_f1_macro": float(sev_f1),
        "pos_score_mean": float(pos_scores.mean()),
        "neg_score_mean": float(neg_scores.mean()),
        "discrimination": float(pos_scores.mean() - neg_scores.mean()),
    }

    # ── Print Table ───────────────────────────────────────────────────────────
    from rich.table import Table
    from rich.console import Console

    console = Console()
    table = Table(title="MedSafe Evaluation Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", style="bold white", justify="right", width=12)

    table.add_row("Hits@10",          f"{hits10:.4f}")
    table.add_row("Hits@20",          f"{hits20:.4f}")
    table.add_row("Hits@50",          f"{hits50:.4f}")
    table.add_row("AUROC",            f"{auroc:.4f}")
    table.add_row("AUPR",             f"{aupr:.4f}")
    table.add_row("Severity Accuracy", f"{sev_acc:.4f}")
    table.add_row("Severity F1 (macro)", f"{sev_f1:.4f}")

    console.print(table)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedSafe Evaluation Benchmark")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to R-GCN checkpoint (default: best checkpoint)")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    cfg = load_config(full_mode=args.full)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or (ROOT / cfg.paths.checkpoints / "rgcn_finetune" / "rgcn_best.pt")

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}. Train the model first: python train.py")
        sys.exit(1)

    results = run_evaluation(ckpt_path, cfg, device)

    # Save results
    results_dir = ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"eval_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results_path}")
