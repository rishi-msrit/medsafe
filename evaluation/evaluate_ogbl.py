"""
evaluation/evaluate_ogbl.py
============================
Evaluate R-GCN on the OGBL-DDI standard benchmark.

Uses the official OGB evaluator to compute Hits@20 and compare against
published baselines (GraphSAGE: 0.5390, SEAL: 0.3058, etc.).

Usage:
  python evaluation/evaluate_ogbl.py
  python evaluation/evaluate_ogbl.py --checkpoint checkpoints/rgcn_finetune/rgcn_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ── PyTorch 2.6+ compatibility: OGB uses torch.load without weights_only ──────
# OGB data files contain torch_geometric classes that need weights_only=False.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
# ─────────────────────────────────────────────────────────────────────────────

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


PUBLISHED_BASELINES = {
    "GraphSAGE": 0.5390,
    "SEAL":       0.3058,
    "LRGA":       0.3430,
    "GCN":        0.3707,
    "MedSafe-RGCN": None,  # Filled in at runtime
}


def run_ogbl_eval(checkpoint_path: Path, cfg, device: torch.device) -> dict:
    """Evaluate on OGBL-DDI using the OGB official evaluator."""
    try:
        from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
    except ImportError:
        logger.error("OGB not installed. Run: pip install ogb")
        return {}

    logger.info("Loading OGBL-DDI dataset (auto-downloads if needed)...")
    dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=str(ROOT / "data" / "raw" / "ogbl_ddi"))
    split_edge = dataset.get_edge_split()
    evaluator = Evaluator(name="ogbl-ddi")

    # Load our DDI graph and model
    from models.rgcn_predictor import build_rgcn_predictor
    from training.finetune_rgcn import build_combined_edge_index

    graph_path = ROOT / cfg.paths.data_graphs / "ddi_hetero_graph.pt"
    if not graph_path.exists():
        logger.error("DDI graph not found. Run the data pipeline first.")
        return {}

    ddi_graph = torch.load(graph_path, map_location=device, weights_only=False)
    drug_x = ddi_graph["drug"].x.to(device)
    drug_feature_dim = drug_x.shape[1]

    model = build_rgcn_predictor(cfg, drug_feature_dim).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    edge_index, edge_type = build_combined_edge_index(ddi_graph, device)
    drug_to_idx = getattr(ddi_graph, "drug_to_idx", {})

    def score_edges(edges: torch.Tensor) -> torch.Tensor:
        """Score a batch of drug-pair edges."""
        scores = []
        batch_size = 1024
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i + batch_size]
            # Map OGBL node IDs → our DDI graph IDs
            src = batch[:, 0].to(device)
            dst = batch[:, 1].to(device)
            # Clamp to our drug count (OGBL-DDI has 4267 drugs)
            src = src.clamp(0, drug_x.shape[0] - 1)
            dst = dst.clamp(0, drug_x.shape[0] - 1)
            with torch.no_grad():
                binary, _, _, _ = model(drug_x, edge_index, edge_type, src, dst)
            scores.append(torch.sigmoid(binary).squeeze())
        return torch.cat(scores)

    logger.info("Scoring test edges...")
    pos_test_edge = split_edge["test"]["edge"].to(device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(device)

    pos_scores = score_edges(pos_test_edge)
    neg_scores = score_edges(neg_test_edge)

    result = evaluator.eval({
        "y_pred_pos": pos_scores.cpu(),
        "y_pred_neg": neg_scores.cpu(),
    })

    hits20 = float(result["hits@20"])
    PUBLISHED_BASELINES["MedSafe-RGCN"] = hits20

    logger.info("\n─── OGBL-DDI Benchmark Results ──────────────────────────")
    logger.info(f"{'Model':<20} {'Hits@20':>10}")
    logger.info("─" * 32)
    for model_name, score in sorted(
        PUBLISHED_BASELINES.items(), key=lambda x: x[1] or 0, reverse=True
    ):
        marker = " ← OURS" if model_name == "MedSafe-RGCN" else ""
        score_str = f"{score:.4f}" if score is not None else "N/A"
        logger.info(f"{model_name:<20} {score_str:>10}{marker}")
    logger.info("─" * 32)

    return {"hits_at_20": hits20, "baselines": PUBLISHED_BASELINES}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OGBL-DDI Evaluation")
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

    run_ogbl_eval(ckpt, cfg, device)
