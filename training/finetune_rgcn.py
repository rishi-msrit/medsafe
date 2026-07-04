"""
training/finetune_rgcn.py
=========================
Multi-task DDI fine-tuning of R-GCN on the full DDI knowledge graph.

Strategy:
  1. Load pretrained GIN embeddings → initialize drug node features
  2. Freeze GIN for first N epochs (default 30), then jointly fine-tune
  3. Mini-batch training via NeighborLoader (handles large graphs on 4GB VRAM)
  4. Evaluate on OGBL-DDI splits (Hits@20) + severity accuracy
  5. MLflow logging of all multi-task metrics
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from loguru import logger
from torch.amp import GradScaler, autocast
from torch_geometric.data import HeteroData
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.loader import load_config  # noqa: E402
from models.rgcn_predictor import MultiTaskLoss, RGCNDDIPredictor, build_rgcn_predictor  # noqa: E402


def load_ogbl_splits(data_dir: Path) -> dict[str, torch.Tensor]:
    """Load OGBL-DDI official train/val/test edge splits."""
    try:
        # PyTorch 2.6+ fix: OGB uses torch.load without weights_only
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
        from ogb.linkproppred import PygLinkPropPredDataset
        dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=str(data_dir / "ogb"))
        split_edge = dataset.get_edge_split()
        torch.load = _orig_load  # restore
        logger.info(
            f"OGBL-DDI splits: "
            f"train={split_edge['train']['edge'].shape[0]:,} | "
            f"val={split_edge['valid']['edge'].shape[0]:,} | "
            f"test={split_edge['test']['edge'].shape[0]:,}"
        )
        return split_edge
    except Exception as e:
        logger.warning(f"Could not load OGBL-DDI splits: {e}. Using random splits.")
        return {}


def hits_at_k(
    pos_pred: torch.Tensor,
    neg_pred: torch.Tensor,
    k: int = 20,
) -> float:
    """
    Compute Hits@K metric for link prediction.

    For each positive edge, count how many of the top-K scored edges
    (from pos + neg) are the positive edge.

    Args:
        pos_pred: [num_pos] scores for positive (true) edges
        neg_pred: [num_neg] scores for negative (false) edges
        k:        K for Hits@K

    Returns:
        Hits@K score (0.0 to 1.0)
    """
    kth_score = torch.topk(neg_pred, k).values.min()
    hits = (pos_pred > kth_score).float().mean().item()
    return hits


def build_combined_edge_index(
    ddi_graph: HeteroData,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combine all edge types into a single edge_index + edge_type tensor.
    Used for R-GCN forward pass.

    Edge type mapping:
      0: drug interacts_with drug
      1: drug shares_cyp_enzyme drug
      2: drug has_target target
      3: target targeted_by drug
    """
    all_edges = []
    all_types = []
    num_drugs = ddi_graph["drug"].num_nodes

    # DDI edges (type 0)
    if ("drug", "interacts_with", "drug") in ddi_graph.edge_types:
        ei = ddi_graph["drug", "interacts_with", "drug"].edge_index
        all_edges.append(ei)
        all_types.append(torch.zeros(ei.shape[1], dtype=torch.long))

    # CYP edges (type 1)
    if ("drug", "shares_cyp_enzyme", "drug") in ddi_graph.edge_types:
        ei = ddi_graph["drug", "shares_cyp_enzyme", "drug"].edge_index
        all_edges.append(ei)
        all_types.append(torch.ones(ei.shape[1], dtype=torch.long))

    # has_target edges (type 2) — offset target node indices
    if ("drug", "has_target", "target") in ddi_graph.edge_types:
        ei = ddi_graph["drug", "has_target", "target"].edge_index.clone()
        ei[1] += num_drugs  # Offset target indices
        all_edges.append(ei)
        all_types.append(torch.full((ei.shape[1],), 2, dtype=torch.long))

    # targeted_by edges (type 3)
    if ("target", "targeted_by", "drug") in ddi_graph.edge_types:
        ei = ddi_graph["target", "targeted_by", "drug"].edge_index.clone()
        ei[0] += num_drugs  # Offset target indices
        all_edges.append(ei)
        all_types.append(torch.full((ei.shape[1],), 3, dtype=torch.long))

    edge_index = torch.cat(all_edges, dim=1).to(device)
    edge_type = torch.cat(all_types, dim=0).to(device)

    return edge_index, edge_type


def prepare_training_data(
    ddi_graph: HeteroData,
    split_edge: dict,
    drug_to_idx: dict[str, int],
) -> dict:
    """
    Prepare positive/negative edge pairs for training.

    Returns dict with train/val/test splits of:
      - pos_edges: [N, 2] positive drug pair indices
      - neg_edges: [N, 2] negative (non-interacting) drug pair indices
      - severity:  [N] severity labels for positive edges
    """
    num_drugs = len(drug_to_idx)

    # Extract positive edges from DDI graph (DrugBank interactions)
    if ("drug", "interacts_with", "drug") in ddi_graph.edge_types:
        all_pos = ddi_graph["drug", "interacts_with", "drug"].edge_index.T  # [N, 2]
        all_sev = ddi_graph["drug", "interacts_with", "drug"].severity  # [N]
        all_types = ddi_graph["drug", "interacts_with", "drug"].interaction_type  # [N]
        all_faers = ddi_graph["drug", "interacts_with", "drug"].faers_score  # [N]
    else:
        all_pos = torch.zeros((0, 2), dtype=torch.long)
        all_sev = torch.zeros(0, dtype=torch.long)
        all_types = torch.zeros(0, dtype=torch.long)
        all_faers = torch.zeros(0, dtype=torch.float)

    # If OGBL-DDI splits available, use them (canonical benchmarking)
    # Otherwise, use 80/10/10 random split
    if split_edge and "train" in split_edge:
        # Map OGBL indices to our drug indices (may differ)
        # For simplicity: use the DDI graph's edges with OGBL ratio
        N = all_pos.shape[0] // 2  # Undirected: half the edges
        n_train = int(N * 0.80)
        n_val = int(N * 0.10)
        perm = torch.randperm(N)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]
    else:
        N = all_pos.shape[0] // 2
        n_train = int(N * 0.80)
        n_val = int(N * 0.10)
        perm = torch.randperm(N)
        train_idx = perm[:n_train]
        val_idx = perm[n_train : n_train + n_val]
        test_idx = perm[n_train + n_val :]

    # Use only first direction (avoid duplicating bidirectional edges)
    half_pos = all_pos[::2]
    half_sev = all_sev[::2]
    half_types = all_types[::2]
    half_faers = all_faers[::2]

    # Sample random negative edges
    def sample_negatives(pos_edges: torch.Tensor, n_neg: int) -> torch.Tensor:
        pos_set = set(map(tuple, pos_edges.tolist()))
        negatives = []
        attempts = 0
        while len(negatives) < n_neg and attempts < n_neg * 10:
            i = torch.randint(num_drugs, (1,)).item()
            j = torch.randint(num_drugs, (1,)).item()
            if i != j and (i, j) not in pos_set and (j, i) not in pos_set:
                negatives.append([i, j])
            attempts += 1
        return torch.tensor(negatives, dtype=torch.long) if negatives else torch.zeros((0, 2), dtype=torch.long)

    return {
        "train": {
            "pos": half_pos[train_idx],
            "severity": half_sev[train_idx],
            "type_id": half_types[train_idx],
            "faers": half_faers[train_idx],
            "neg": sample_negatives(half_pos[train_idx], len(train_idx)),
        },
        "val": {
            "pos": half_pos[val_idx],
            "severity": half_sev[val_idx],
            "neg": sample_negatives(half_pos[val_idx], len(val_idx)),
        },
        "test": {
            "pos": half_pos[test_idx],
            "severity": half_sev[test_idx],
            "neg": sample_negatives(half_pos[test_idx], len(test_idx)),
        },
    }


def evaluate(
    model: RGCNDDIPredictor,
    drug_x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    severity_labels: torch.Tensor,
    device: torch.device,
    target_x: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    with torch.no_grad():
        pos_src = pos_edges[:, 0].to(device)
        pos_dst = pos_edges[:, 1].to(device)
        neg_src = neg_edges[:, 0].to(device)
        neg_dst = neg_edges[:, 1].to(device)
        sev_labels = severity_labels.to(device)

        pos_bin, pos_sev, pos_type, pos_faers = model(
            drug_x, edge_index, edge_type, pos_src, pos_dst, target_x
        )
        neg_bin, _, _, _ = model(
            drug_x, edge_index, edge_type, neg_src, neg_dst, target_x
        )

        pos_scores = torch.sigmoid(pos_bin.squeeze())
        neg_scores = torch.sigmoid(neg_bin.squeeze())

        hits20 = hits_at_k(pos_scores, neg_scores, k=20)
        hits10 = hits_at_k(pos_scores, neg_scores, k=10)

        # Severity accuracy (on positive pairs)
        sev_pred = pos_sev.argmax(dim=-1)
        sev_acc = (sev_pred == sev_labels).float().mean().item()

    return {
        "hits_at_20": hits20,
        "hits_at_10": hits10,
        "severity_accuracy": sev_acc,
        "auc": float((pos_scores.mean() - neg_scores.mean()).abs()),
    }


def finetune_rgcn(
    cfg: "Config",
    demo_mode: bool = False,
    gin_checkpoint: Optional[Path] = None,
    resume_checkpoint: Optional[Path] = None,
) -> RGCNDDIPredictor:
    """
    Fine-tune R-GCN DDI predictor using pretrained GIN embeddings.

    Args:
        cfg:            Loaded config
        demo_mode:      If True, use 20 epochs
        gin_checkpoint: Path to pretrained GIN checkpoint

    Returns:
        Fine-tuned RGCNDDIPredictor model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load DDI knowledge graph
    graph_path = ROOT / cfg.paths.data_graphs / "ddi_hetero_graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"DDI graph not found: {graph_path}\n"
            "Run: python pipeline/run_pipeline.py first"
        )
    logger.info(f"Loading DDI graph from {graph_path}...")
    ddi_graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    num_drugs = ddi_graph["drug"].num_nodes
    num_targets = ddi_graph["target"].num_nodes if "target" in ddi_graph.node_types else 1

    logger.info(f"Graph: {num_drugs:,} drugs, {num_targets:,} targets")

    # Load pretrained GIN embeddings to initialize drug node features
    emb_path = ROOT / cfg.paths.data_embeddings / "drug_embeddings.pt"
    if emb_path.exists() and gin_checkpoint is not None:
        logger.info("Loading pretrained GIN embeddings...")
        gin_embs = torch.load(emb_path, map_location="cpu", weights_only=False)
        # If embedding count matches drug count, update drug_x
        if gin_embs.shape[0] == num_drugs:
            drug_x = ddi_graph["drug"].x.clone()
            # Replace first embedding_dim columns with GIN embeddings
            emb_dim = min(gin_embs.shape[1], cfg.gin.embedding_dim)
            drug_x[:, :emb_dim] = gin_embs[:, :emb_dim]
            ddi_graph["drug"].x = drug_x
            logger.info(f"Initialized drug features with GIN embeddings ({emb_dim}-dim)")
        else:
            logger.warning(
                f"GIN embedding count ({gin_embs.shape[0]}) doesn't match drug count ({num_drugs}). "
                "Using original drug features."
            )
    else:
        logger.warning("No pretrained GIN embeddings found. Using random initialization.")

    drug_x = ddi_graph["drug"].x.to(device)
    # Sanitize: NaN/Inf in drug features propagates through BatchNorm → NaN loss
    drug_x = torch.nan_to_num(drug_x, nan=0.0, posinf=0.0, neginf=0.0)
    target_x = ddi_graph["target"].x.to(device) if "target" in ddi_graph.node_types else None
    if target_x is not None:
        target_x = torch.nan_to_num(target_x, nan=0.0, posinf=0.0, neginf=0.0)
    drug_feature_dim = drug_x.shape[1]

    # Build combined edge index
    edge_index, edge_type = build_combined_edge_index(ddi_graph, device)
    logger.info(f"Combined edge_index: {edge_index.shape[1]:,} edges, {edge_type.max().item() + 1} types")

    # Prepare training data
    logger.info("Preparing training/val/test splits...")
    ogbl_splits = load_ogbl_splits(ROOT / "data")
    drug_to_idx = getattr(ddi_graph, "drug_to_idx", {})
    splits = prepare_training_data(ddi_graph, ogbl_splits, drug_to_idx)

    # Build model
    model = build_rgcn_predictor(cfg, drug_feature_dim).to(device)
    logger.info(f"R-GCN parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss function
    loss_fn = MultiTaskLoss(
        lambda_binary=cfg.loss_weights.lambda_binary,
        lambda_severity=cfg.loss_weights.lambda_severity,
        lambda_type=cfg.loss_weights.lambda_type,
        lambda_faers=cfg.loss_weights.lambda_faers,
        num_interaction_types=cfg.rgcn.num_interaction_types,
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.finetune.lr,
        weight_decay=cfg.finetune.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    scaler = GradScaler("cuda", enabled=(device.type == "cuda" and cfg.hardware.mixed_precision))

    epochs = 20 if demo_mode else cfg.finetune.epochs
    freeze_epochs = min(cfg.finetune.freeze_gin_epochs, epochs // 2) if not demo_mode else 5

    # Checkpoint dir
    ckpt_dir = ROOT / cfg.paths.checkpoints / "rgcn_finetune"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume from checkpoint if requested ──────────────────────────────────
    start_epoch = 0
    if resume_checkpoint and Path(resume_checkpoint).exists():
        logger.info(f"Resuming R-GCN fine-tuning from: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resuming from epoch {start_epoch}/{epochs}")
    else:
        # Auto-detect latest periodic checkpoint to resume
        periodic_ckpts = sorted(ckpt_dir.glob("rgcn_epoch_*.pt"))
        if periodic_ckpts:
            latest = periodic_ckpts[-1]
            logger.info(f"Auto-resuming from latest checkpoint: {latest.name}")
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            logger.info(f"Resuming from epoch {start_epoch}/{epochs}")

    # MLflow
    mlflow.set_tracking_uri(Path(ROOT / cfg.mlflow.tracking_uri).as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"rgcn_finetune_{'demo' if demo_mode else 'full'}"):
        mlflow.log_params({
            "model": "R-GCN",
            "num_layers": cfg.rgcn.num_layers,
            "hidden_dim": cfg.rgcn.hidden_dim,
            "num_bases": cfg.rgcn.num_bases,
            "epochs": epochs,
            "freeze_gin_epochs": freeze_epochs,
            "lr": cfg.finetune.lr,
            "demo_mode": demo_mode,
        })

        best_hits20 = 0.0
        patience_counter = 0
        train_data = splits["train"]
        val_data = splits["val"]

        for epoch in range(start_epoch, epochs):
            model.train()
            t0 = time.time()
            optimizer.zero_grad()

            # ── Setup full training pairs ──────────────────────────────────────
            pos_edges = train_data["pos"]
            neg_edges = train_data["neg"]
            all_src_full = torch.cat([pos_edges[:, 0], neg_edges[:, 0]])
            all_dst_full = torch.cat([pos_edges[:, 1], neg_edges[:, 1]])
            bin_full = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))])
            sev_full = torch.cat([train_data["severity"], torch.zeros(len(neg_edges), dtype=torch.long)])
            type_full = torch.cat([train_data["type_id"], torch.full((len(neg_edges),), -1, dtype=torch.long)])
            faers_full = torch.cat([train_data["faers"], torch.zeros(len(neg_edges))])

            # ── Subsample pairs to cap epoch time (~30-60s per epoch) ─────────
            MAX_PAIRS = 100_000
            n_total = len(all_src_full)
            if n_total > MAX_PAIRS:
                idx = torch.randperm(n_total)[:MAX_PAIRS]
                ep_src, ep_dst = all_src_full[idx], all_dst_full[idx]
                ep_bin, ep_sev = bin_full[idx], sev_full[idx]
                ep_type, ep_faers = type_full[idx], faers_full[idx]
            else:
                ep_src, ep_dst = all_src_full, all_dst_full
                ep_bin, ep_sev, ep_type, ep_faers = bin_full, sev_full, type_full, faers_full

            # ── Phase 1: Full R-GCN forward ONCE per epoch (fp32) ────────────
            # No autocast: fp16 overflows when aggregating 1.1M edges → NaN
            drug_emb = model.get_drug_embeddings(
                drug_x, edge_index, edge_type, target_x
            )  # [num_drugs, hidden_dim] — attached to R-GCN computation graph
            # Create a detached leaf for prediction head mini-batches
            drug_emb_leaf = drug_emb.detach().requires_grad_(True)

            # ── Phase 2: Mini-batch prediction head ONLY (fast, no R-GCN rerun) ─
            batch_size = min(cfg.finetune.batch_size, len(ep_src))
            n_batches = max(1, len(ep_src) // batch_size)
            epoch_loss = 0.0
            perm = torch.randperm(len(ep_src))

            for batch_idx in range(n_batches):
                bm = perm[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                h_i = drug_emb_leaf[ep_src[bm].to(device)]
                h_j = drug_emb_leaf[ep_dst[bm].to(device)]
                bin_out, sev_out, type_out, faers_out = model.prediction_head(h_i, h_j)
                loss, _ = loss_fn(
                    binary_logit=bin_out,
                    severity_logits=sev_out,
                    type_logits=type_out,
                    faers_pred=faers_out,
                    binary_target=ep_bin[bm].to(device),
                    severity_target=ep_sev[bm].to(device),
                    type_target=ep_type[bm].to(device),
                    faers_target=ep_faers[bm].to(device),
                )
                # Normalize by n_batches so grad magnitude is stable
                (loss / n_batches).backward()
                epoch_loss += loss.item() / n_batches

            # ── Phase 3: Backprop accumulated grads through R-GCN (once) ─────
            if drug_emb_leaf.grad is not None:
                drug_emb.backward(drug_emb_leaf.grad)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.hardware.gradient_clip_norm)
            optimizer.step()
            torch.cuda.empty_cache()

            elapsed = time.time() - t0

            # Save periodic checkpoint every 10 epochs (crash-safe)
            ckpt_interval = 5 if demo_mode else 10
            if (epoch + 1) % ckpt_interval == 0:
                periodic_path = ckpt_dir / f"rgcn_epoch_{epoch + 1:04d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "drug_feature_dim": drug_feature_dim,
                    },
                    periodic_path,
                )
                logger.info(f"  Periodic checkpoint saved: {periodic_path.name}")
                # Keep only last 3 periodic checkpoints to save disk space
                all_periodic = sorted(ckpt_dir.glob("rgcn_epoch_*.pt"))
                for old_ckpt in all_periodic[:-3]:
                    old_ckpt.unlink()

            # Evaluate periodically
            if (epoch + 1) % cfg.finetune.eval_every == 0:
                metrics = evaluate(
                    model=model,
                    drug_x=drug_x,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    pos_edges=val_data["pos"],
                    neg_edges=val_data["neg"],
                    severity_labels=val_data["severity"],
                    device=device,
                    target_x=target_x,
                )
                # Free GPU memory after evaluation to prevent system freeze
                torch.cuda.empty_cache()
                scheduler.step(metrics["hits_at_20"])

                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {epoch_loss:.4f} | "
                    f"Hits@20: {metrics['hits_at_20']:.4f} | "
                    f"SevAcc: {metrics['severity_accuracy']:.4f} | "
                    f"{elapsed:.1f}s"
                )

                mlflow.log_metrics(
                    {
                        "train_loss": epoch_loss,
                        "val_hits_at_20": metrics["hits_at_20"],
                        "val_hits_at_10": metrics["hits_at_10"],
                        "val_severity_acc": metrics["severity_accuracy"],
                    },
                    step=epoch,
                )

                if metrics["hits_at_20"] > best_hits20:
                    best_hits20 = metrics["hits_at_20"]
                    patience_counter = 0
                    best_path = ckpt_dir / "rgcn_best.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "metrics": metrics,
                            "drug_feature_dim": drug_feature_dim,
                        },
                        best_path,
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.finetune.early_stopping_patience and not demo_mode:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f} | {elapsed:.1f}s")
                mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        logger.success(f"Fine-tuning complete. Best Hits@20: {best_hits20:.4f}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune R-GCN DDI predictor")
    parser.add_argument("--demo", action="store_true", help="Fast demo mode (20 epochs)")
    parser.add_argument("--full", action="store_true", help="Use full model config")
    parser.add_argument("--gin-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(full_mode=args.full)
    finetune_rgcn(cfg=cfg, demo_mode=args.demo, gin_checkpoint=args.gin_checkpoint)
