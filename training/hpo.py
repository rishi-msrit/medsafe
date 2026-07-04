"""
training/hpo.py
===============
Hyperparameter optimisation for the R-GCN DDI predictor using Optuna.

Searches over:
  - R-GCN hidden_dim, num_layers, num_bases, dropout
  - Learning rate, weight decay
  - Multi-task loss weights

Optimises for: Hits@20 on validation split.
Respects 4 GB VRAM budget by bounding hidden_dim and enforcing basis decomp.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import optuna
import torch
from loguru import logger
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.loader import load_config  # noqa: E402


def objective(trial: optuna.Trial, cfg, device: torch.device) -> float:
    """Single Optuna trial: build model, train briefly, return Hits@20."""
    from models.rgcn_predictor import build_rgcn_predictor
    from training.finetune_rgcn import (
        build_combined_edge_index,
        compute_multitask_loss,
        evaluate_hits_at_k,
        prepare_training_data,
    )

    # ── Hyperparameter search space ────────────────────────────────────────────
    hidden_dim   = trial.suggest_categorical("hidden_dim",   [64, 128, 192])
    num_layers   = trial.suggest_int("num_layers",   2, 3)
    num_bases    = trial.suggest_categorical("num_bases",    [8, 16, 24])
    dropout      = trial.suggest_float("dropout",    0.1, 0.35, step=0.05)
    lr           = trial.suggest_float("lr",         1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    lambda_sev   = trial.suggest_float("lambda_severity", 0.3, 1.2, step=0.1)
    lambda_type  = trial.suggest_float("lambda_type",     0.2, 0.8, step=0.1)
    lambda_faers = trial.suggest_float("lambda_faers",    0.1, 0.5, step=0.1)

    # Override cfg fields for this trial
    class TrialCfg:
        class rgcn:
            pass
        class loss_weights:
            pass

    trial_cfg = TrialCfg()
    trial_cfg.rgcn = type("rgcn", (), {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "num_bases":  num_bases,
        "dropout":    dropout,
        "num_severity_levels": cfg.rgcn.num_severity_levels,
        "num_interaction_types": cfg.rgcn.num_interaction_types,
    })()
    trial_cfg.loss_weights = type("lw", (), {
        "lambda_binary":   1.0,
        "lambda_severity": lambda_sev,
        "lambda_type":     lambda_type,
        "lambda_faers":    lambda_faers,
    })()
    trial_cfg.finetune = cfg.finetune
    trial_cfg.hardware = cfg.hardware

    # ── Load graph ────────────────────────────────────────────────────────────
    graph_path = ROOT / cfg.paths.data_graphs / "ddi_hetero_graph.pt"
    if not graph_path.exists():
        raise optuna.exceptions.TrialPruned()

    ddi_graph = torch.load(graph_path, map_location=device, weights_only=False)
    drug_x = ddi_graph["drug"].x.to(device)
    drug_feature_dim = drug_x.shape[1]
    drug_to_idx = getattr(ddi_graph, "drug_to_idx", {})

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_rgcn_predictor(trial_cfg, drug_feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    edge_index, edge_type = build_combined_edge_index(ddi_graph, device)
    splits = prepare_training_data(ddi_graph, {}, drug_to_idx)
    train_data = splits["train"]
    val_data   = splits["val"]

    # ── Train for 30 epochs (quick budget) ───────────────────────────────────
    MAX_EPOCHS = 30
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        pos_e = train_data["pos"].to(device)
        neg_e = train_data["neg"].to(device)
        n = min(len(pos_e), 512)  # cap batch
        pos_e = pos_e[:n]
        neg_e = neg_e[:n]

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pos_pred = model(
                drug_x, edge_index, edge_type,
                pos_e[:, 0], pos_e[:, 1]
            )
            neg_pred = model(
                drug_x, edge_index, edge_type,
                neg_e[:, 0], neg_e[:, 1]
            )
            loss = compute_multitask_loss(
                pos_pred, neg_pred,
                pos_sev_labels=train_data["severity"][:n].to(device),
                pos_type_labels=train_data.get(
                    "interaction_type", torch.zeros(n, dtype=torch.long)
                ).to(device),
                pos_faers_labels=train_data.get(
                    "faers_score", torch.zeros(n)
                ).to(device),
                cfg=trial_cfg,
            )

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # ── Intermediate pruning ─────────────────────────────────────────────
        if epoch % 10 == 0:
            val_hits = evaluate_hits_at_k(
                model, drug_x, edge_index, edge_type,
                val_data, device=device, k=20
            )
            trial.report(val_hits, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # ── Final evaluation ──────────────────────────────────────────────────────
    final_hits = evaluate_hits_at_k(
        model, drug_x, edge_index, edge_type,
        val_data, device=device, k=20
    )
    return float(final_hits)


def run_hpo(cfg, device: torch.device, n_trials: int = 50, timeout: int = 7200) -> dict:
    """
    Run hyperparameter optimisation study.

    Args:
        cfg:      MedSafe config
        device:   Torch device
        n_trials: Number of Optuna trials (default 50)
        timeout:  Max wall-clock seconds (default 7200 = 2h)

    Returns:
        Best hyperparameters dict
    """
    sampler = TPESampler(seed=42, n_startup_trials=10)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=cfg.hpo.study_name,
        storage=None,  # In-memory (no DB needed for local runs)
    )

    def wrapped_objective(trial):
        return objective(trial, cfg, device)

    logger.info(f"Starting HPO: {n_trials} trials, timeout={timeout}s")
    study.optimize(wrapped_objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    best = study.best_params
    logger.info(f"Best Hits@20: {study.best_value:.4f}")
    logger.info(f"Best params: {best}")

    # Save best params
    import json
    out_path = ROOT / "checkpoints" / "best_hpo_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"best_value": study.best_value, "params": best}, f, indent=2)
    logger.info(f"Saved HPO results → {out_path}")

    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedSafe HPO")
    parser.add_argument("--trials",  type=int, default=50)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--full",    action="store_true")
    args = parser.parse_args()

    cfg = load_config(full_mode=args.full)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"HPO device: {device}")

    run_hpo(cfg, device, n_trials=args.trials, timeout=args.timeout)
