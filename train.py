
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedSafe — Full Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Fast demo mode: 20 epochs, small subset, completes in ~10-15 minutes",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full model config (larger dims, more layers) — for GPUs >8GB VRAM",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip GIN pretraining — use existing checkpoint if available",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Only run GIN pretraining, skip R-GCN fine-tuning",
    )
    parser.add_argument(
        "--gin-checkpoint",
        type=Path,
        default=None,
        help="Path to existing GIN checkpoint to use for fine-tuning",
    )
    parser.add_argument(
        "--resume-pretrain",
        type=Path,
        default=None,
        help="Resume GIN pretraining from this checkpoint",
    )
    parser.add_argument(
        "--resume-finetune",
        type=Path,
        default=None,
        help="Resume R-GCN fine-tuning from this specific checkpoint (auto-detects latest if not set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_data_readiness():
    status = {}

    mol_idx = ROOT / "data" / "graphs" / "drug_graph_index.parquet"
    ddi_graph = ROOT / "data" / "graphs" / "ddi_hetero_graph.pt"
    drugs_pq = ROOT / "data" / "processed" / "drugs.parquet"

    status["drugs_parquet"] = drugs_pq.exists()
    status["molecular_graphs"] = mol_idx.exists()
    status["ddi_graph"] = ddi_graph.exists()

    return status


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    from configs.loader import load_config
    cfg = load_config(full_mode=args.full)

    # ── Header ────────────────────────────────────────────────────────────────
    mode_tag = "DEMO MODE (20 epochs)" if args.demo else "FULL PRODUCTION TRAINING"
    model_tag = "Full (large GPU)" if args.full else "Default (RTX 3050 4GB)"
    console.print(
        Panel(
            f"[bold cyan]MedSafe — Training Pipeline[/bold cyan]\n"
            f"Mode:   [yellow]{mode_tag}[/yellow]\n"
            f"Config: [green]{model_tag}[/green]\n"
            f"Seed:   {args.seed}",
            border_style="cyan",
        )
    )

    if args.demo:
        console.print(
            "[yellow]Demo mode: using fast 20-epoch training for quick verification.\n"
            "For full production training, run without --demo flag.[/yellow]\n"
        )

    # ── Data readiness check ──────────────────────────────────────────────────
    console.rule("[bold]Data Readiness Check")
    data_status = check_data_readiness()

    for component, ready in data_status.items():
        status_str = "[green]✓ Ready[/green]" if ready else "[red]✗ Missing[/red]"
        console.print(f"  {component:<25} {status_str}")

    if not data_status["molecular_graphs"] and not args.skip_pretrain:
        console.print(
            "\n[red]Molecular graphs not found.[/red]\n"
            "Run first: [bold]python pipeline/run_pipeline.py[/bold]\n"
        )
        sys.exit(1)

    if not data_status["ddi_graph"] and not args.skip_finetune:
        console.print(
            "\n[red]DDI knowledge graph not found.[/red]\n"
            "Run first: [bold]python pipeline/run_pipeline.py[/bold]\n"
        )
        sys.exit(1)

    total_start = time.time()

    # ── Stage 1: GIN Pretraining ──────────────────────────────────────────────
    gin_ckpt_path: Path | None = args.gin_checkpoint
    default_gin_path = ROOT / cfg.paths.checkpoints / "gin_pretrain" / "gin_pretrained_final.pt"

    if args.skip_pretrain:
        console.rule("[bold dim]Stage 1/2 — GIN Pretraining (SKIPPED)")
        if default_gin_path.exists():
            gin_ckpt_path = default_gin_path
            console.print(f"[dim]Using existing checkpoint: {default_gin_path.name}[/dim]")
        elif gin_ckpt_path is None:
            console.print("[yellow]No GIN checkpoint found. Fine-tuning will use random drug features.[/yellow]")
    else:
        console.rule("[bold]Stage 1/2 — GIN Molecular Pretraining")
        t0 = time.time()
        try:
            from training.pretrain_gin import pretrain_gin

            pretrain_gin(
                cfg=cfg,
                demo_mode=args.demo,
                resume_checkpoint=args.resume_pretrain,
            )
            gin_ckpt_path = default_gin_path
            elapsed = time.time() - t0
            console.print(f"\n[green]✓ GIN pretraining complete in {elapsed/60:.1f} minutes[/green]")
        except Exception as e:
            logger.exception(f"GIN pretraining failed: {e}")
            console.print(f"[red]✗ GIN pretraining failed: {e}[/red]")
            console.print("[yellow]Continuing to fine-tuning with existing/random embeddings...[/yellow]")

    # ── Stage 2: R-GCN Fine-tuning ────────────────────────────────────────────
    if not args.skip_finetune:
        console.rule("[bold]Stage 2/2 — R-GCN DDI Fine-tuning")
        t0 = time.time()
        try:
            from training.finetune_rgcn import finetune_rgcn

            model = finetune_rgcn(
                cfg=cfg,
                demo_mode=args.demo,
                gin_checkpoint=gin_ckpt_path,
                resume_checkpoint=args.resume_finetune,
            )
            elapsed = time.time() - t0
            console.print(f"\n[green]✓ R-GCN fine-tuning complete in {elapsed/60:.1f} minutes[/green]")
        except Exception as e:
            logger.exception(f"R-GCN fine-tuning failed: {e}")
            console.print(f"[red]✗ R-GCN fine-tuning failed: {e}[/red]")
    else:
        console.rule("[bold dim]Stage 2/2 — R-GCN Fine-tuning (SKIPPED)")

    # ── Summary ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    console.rule("[bold]Training Complete")
    console.print(f"Total training time: {total_elapsed/60:.1f} minutes")
    console.print(
        "\nNext steps:\n"
        "  [bold]python serving/api.py[/bold]           → Start FastAPI server\n"
        "  [bold]cd frontend && npm run dev[/bold]       → Start React frontend\n"
        "  [bold]mlflow ui[/bold]                        → View training metrics\n"
        "  [bold]pytest tests/[/bold]                    → Run test suite\n"
    )


if __name__ == "__main__":
    main()
