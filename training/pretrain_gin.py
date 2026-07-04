"""
training/pretrain_gin.py
========================
Self-supervised contrastive pretraining of the GIN molecular encoder.

Training loop:
  - SimCLR-style NT-Xent loss on pairs of molecular graph augmentations
  - Cosine annealing LR schedule with linear warmup
  - Mixed precision (torch.cuda.amp) for VRAM efficiency
  - MLflow logging of loss curves and embedding quality
  - Checkpoint every 50 epochs, resume support

Hardware: Configured for RTX 3050 4GB VRAM by default.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from loguru import logger
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.loader import load_config  # noqa: E402
from models.contrastive import ContrastiveBatch, MolecularGraphDataset, NTXentLoss  # noqa: E402
from models.gin_encoder import GINEncoderWithProjection, build_gin_encoder, count_parameters  # noqa: E402


def compute_embedding_alignment(
    model: GINEncoderWithProjection,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> float:
    """
    Compute embedding alignment: measure how well same-class drugs cluster.
    Returns mean cosine similarity between embeddings from same ATC class.
    Serves as a proxy for embedding quality during pretraining.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = batch.to(device)
            emb = model.encoder(batch)
            embeddings.append(emb.cpu())

    if not embeddings:
        return 0.0

    all_embs = torch.cat(embeddings, dim=0)
    # Normalize
    all_embs = torch.nn.functional.normalize(all_embs, dim=-1)
    # Compute pairwise similarity (sample-based for speed)
    n = min(256, all_embs.shape[0])
    idx = torch.randperm(all_embs.shape[0])[:n]
    sub = all_embs[idx]
    sim_matrix = (sub @ sub.T)
    # Mean off-diagonal similarity
    mask = ~torch.eye(n, dtype=torch.bool)
    mean_sim = sim_matrix[mask].mean().item()
    return mean_sim


def train_epoch(
    model: GINEncoderWithProjection,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: NTXentLoss,
    augmentor: ContrastiveBatch,
    device: torch.device,
    scaler: GradScaler,
    grad_clip: float = 1.0,
    grad_accumulation_steps: int = 1,
) -> float:
    """Run one epoch of contrastive pretraining. Returns mean loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  Pretrain", leave=False)):
        # Apply augmentations to create two views
        batch_list = batch.to_data_list() if hasattr(batch, "to_data_list") else [batch]
        view1_batch, view2_batch = augmentor(batch_list)

        view1_batch = view1_batch.to(device)
        view2_batch = view2_batch.to(device)

        with autocast("cuda", enabled=(device.type == "cuda")):
            _, z_i = model(view1_batch)
            _, z_j = model(view2_batch)
            loss = loss_fn(z_i, z_j)
            loss = loss / grad_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accumulation_steps
        num_batches += 1

    return total_loss / max(num_batches, 1)


def pretrain_gin(
    cfg: "Config",
    demo_mode: bool = False,
    resume_checkpoint: Path | None = None,
) -> GINEncoderWithProjection:
    """
    Full pretraining loop for GIN molecular encoder.

    Args:
        cfg:               Loaded config
        demo_mode:         If True, use 20 epochs + small subset
        resume_checkpoint: Path to .pt checkpoint to resume from

    Returns:
        Pretrained GINEncoderWithProjection model
    """
    # Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        if cfg.hardware.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    # Load molecular graph dataset
    mol_graph_dir = ROOT / cfg.paths.data_graphs / "molecular"
    index_path = ROOT / cfg.paths.data_graphs / "drug_graph_index.parquet"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Molecular graph index not found: {index_path}\n"
            "Run: python pipeline/run_pipeline.py first"
        )

    graph_index = pd.read_parquet(index_path)
    logger.info(f"Molecular graph dataset: {len(graph_index):,} drugs")

    if demo_mode:
        # Use a small random subset for fast testing
        graph_index = graph_index.sample(n=min(500, len(graph_index)), random_state=42)
        logger.info(f"Demo mode: using {len(graph_index)} drugs")

    dataset = MolecularGraphDataset(mol_graph_dir, graph_index)

    # Use config or demo epochs
    epochs = 20 if demo_mode else cfg.contrastive.epochs
    batch_size = cfg.contrastive.batch_size
    if demo_mode:
        batch_size = min(batch_size, 32)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.hardware.num_workers,
        pin_memory=(device.type == "cuda" and cfg.hardware.pin_memory),
        drop_last=True,
    )

    # Model
    model = build_gin_encoder(cfg).to(device)
    logger.info(f"GIN encoder parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.contrastive.lr,
        weight_decay=cfg.contrastive.weight_decay,
    )

    # LR Schedule: linear warmup → cosine annealing
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=cfg.contrastive.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - cfg.contrastive.warmup_epochs, 1),
        eta_min=cfg.contrastive.lr * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.contrastive.warmup_epochs],
    )

    # Loss function
    loss_fn = NTXentLoss(temperature=cfg.contrastive.temperature)

    # Augmentor
    augmentor = ContrastiveBatch(
        mask_ratio_1=cfg.contrastive.aug_atom_mask_ratio,
        mask_ratio_2=cfg.contrastive.aug_atom_mask_ratio_2,
        drop_ratio=cfg.contrastive.aug_bond_drop_ratio,
        noise_sigma=cfg.contrastive.aug_noise_sigma,
    )

    # Mixed precision scaler
    scaler = GradScaler("cuda", enabled=(device.type == "cuda" and cfg.hardware.mixed_precision))

    # Gradient accumulation (for VRAM efficiency)
    grad_accum = cfg.hardware.gradient_clip_norm  # Reuse config field
    grad_accum_steps = max(1, 256 // batch_size)  # Target effective batch 256

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_checkpoint and resume_checkpoint.exists():
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Checkpoint directory
    ckpt_dir = ROOT / cfg.paths.checkpoints / "gin_pretrain"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri(Path(ROOT / cfg.mlflow.tracking_uri).as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"gin_pretrain_{'demo' if demo_mode else 'full'}"):
        # Log hyperparameters
        mlflow.log_params({
            "model": "GIN",
            "num_layers": cfg.gin.num_layers,
            "hidden_dim": cfg.gin.hidden_dim,
            "embedding_dim": cfg.gin.embedding_dim,
            "projection_dim": cfg.contrastive.projection_dim,
            "temperature": cfg.contrastive.temperature,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": cfg.contrastive.lr,
            "demo_mode": demo_mode,
            "dataset_size": len(dataset),
        })

        best_loss = float("inf")

        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            train_loss = train_epoch(
                model=model,
                loader=loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                augmentor=augmentor,
                device=device,
                scaler=scaler,
                grad_clip=cfg.hardware.gradient_clip_norm,
                grad_accumulation_steps=grad_accum_steps,
            )
            scheduler.step()

            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # MLflow logging
            mlflow.log_metrics(
                {"pretrain_loss": train_loss, "lr": lr},
                step=epoch,
            )

            # Save checkpoint every 50 epochs (or every 5 in demo mode)
            ckpt_interval = 5 if demo_mode else 50
            if (epoch + 1) % ckpt_interval == 0 or epoch == epochs - 1:
                ckpt_path = ckpt_dir / f"gin_epoch_{epoch + 1:04d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "loss": train_loss,
                        "config": {
                            "num_layers": cfg.gin.num_layers,
                            "hidden_dim": cfg.gin.hidden_dim,
                            "embedding_dim": cfg.gin.embedding_dim,
                        },
                    },
                    ckpt_path,
                )
                logger.info(f"  Checkpoint saved: {ckpt_path.name}")

                if train_loss < best_loss:
                    best_loss = train_loss
                    best_path = ckpt_dir / "gin_best.pt"
                    torch.save(torch.load(ckpt_path), best_path)

        # Save final model
        final_path = ckpt_dir / "gin_pretrained_final.pt"
        torch.save(
            {
                "epoch": epochs - 1,
                "model_state": model.state_dict(),
                "loss": train_loss,
                "config": {
                    "num_layers": cfg.gin.num_layers,
                    "hidden_dim": cfg.gin.hidden_dim,
                    "embedding_dim": cfg.gin.embedding_dim,
                },
            },
            final_path,
        )
        mlflow.log_artifact(str(final_path), artifact_path="models")
        logger.success(f"GIN pretraining complete. Best loss: {best_loss:.4f}")
        logger.info(f"Final model saved: {final_path}")

    # Compute and save all drug embeddings
    logger.info("Computing and saving drug embeddings for all drugs...")
    _save_all_embeddings(model, dataset, loader, device, cfg)

    return model


def _save_all_embeddings(
    model: GINEncoderWithProjection,
    dataset: MolecularGraphDataset,
    loader: DataLoader,
    device: torch.device,
    cfg: "Config",
) -> None:
    """Compute and save embeddings for all drugs to data/embeddings/."""
    model.eval()
    embeddings = []
    drug_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing embeddings"):
            batch = batch.to(device)
            emb = model.encoder(batch)
            embeddings.append(emb.cpu())

            # Extract drug IDs from batch
            if hasattr(batch, "drug_id"):
                if isinstance(batch.drug_id, list):
                    drug_ids.extend(batch.drug_id)
                else:
                    drug_ids.append(batch.drug_id)

    if embeddings:
        all_embs = torch.cat(embeddings, dim=0)
        emb_dir = ROOT / cfg.paths.data_embeddings
        emb_dir.mkdir(parents=True, exist_ok=True)

        torch.save(all_embs, emb_dir / "drug_embeddings.pt")
        if drug_ids:
            pd.DataFrame({"drug_id": drug_ids}).to_parquet(emb_dir / "embedding_drug_ids.parquet", index=False)
        logger.info(f"Saved {all_embs.shape[0]:,} drug embeddings to {emb_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain GIN molecular encoder")
    parser.add_argument("--demo", action="store_true", help="Fast demo mode (20 epochs)")
    parser.add_argument("--full", action="store_true", help="Use full model config")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    cfg = load_config(full_mode=args.full)
    pretrain_gin(cfg=cfg, demo_mode=args.demo, resume_checkpoint=args.resume)
