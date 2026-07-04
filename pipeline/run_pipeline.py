"""
pipeline/run_pipeline.py
=========================
Full pipeline orchestrator. Run this to reproduce all processed data.

Usage:
    python pipeline/run_pipeline.py
    python pipeline/run_pipeline.py --skip-faers
    python pipeline/run_pipeline.py --drugbank-path /custom/path/to/drugbank.xml
    python pipeline/run_pipeline.py --demo --max-drugs 500
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.loader import load_config  # noqa: E402

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedSafe — Full Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--drugbank-path",
        type=Path,
        default=None,
        help="Path to DrugBank CSV or XML (auto-detected; default: data/raw/drugbank_full_database.csv)",
    )
    parser.add_argument(
        "--no-smiles",
        action="store_true",
        help="Skip PubChem SMILES fetch (much faster, but molecular graphs will be descriptor-only)",
    )
    parser.add_argument(
        "--skip-faers",
        action="store_true",
        help="Skip FAERS harm signal computation",
    )
    parser.add_argument(
        "--skip-twosides",
        action="store_true",
        help="Skip TWOSIDES download/processing",
    )
    parser.add_argument(
        "--skip-graphs",
        action="store_true",
        help="Skip molecular graph construction",
    )
    parser.add_argument(
        "--skip-ddi-graph",
        action="store_true",
        help="Skip DDI knowledge graph construction",
    )
    parser.add_argument(
        "--max-drugs",
        type=int,
        default=None,
        help="Limit DrugBank drugs parsed (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full MedSafe data pipeline."""
    import random
    import numpy as np

    # Set reproducibility seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config()
    processed_dir = ROOT / cfg.paths.data_processed
    graphs_dir = ROOT / cfg.paths.data_graphs
    processed_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            "[bold cyan]MedSafe — Data Pipeline[/bold cyan]\n"
            "Building all processed datasets for training.",
            border_style="cyan",
        )
    )

    results: dict[str, str] = {}
    timings: dict[str, float] = {}

    # ── Step 1: Parse DrugBank ────────────────────────────────────────────────
    console.rule("[bold]Step 1/5 — DrugBank Parsing (CSV auto-detected)")
    t0 = time.time()
    try:
        from pipeline.parse_drugbank import parse_drugbank_from_config, parse_drugbank

        # Auto-detect: prefer CSV, fall back to XML
        csv_path = args.drugbank_path or ROOT / "data" / "raw" / "drugbank_full_database.csv"
        xml_path = ROOT / cfg.paths.drugbank_xml

        if csv_path.exists():
            drugs_df, interactions_df, targets_df = parse_drugbank(
                csv_path=csv_path,
                output_dir=processed_dir,
                max_drugs=args.max_drugs,
                fetch_smiles=not getattr(args, "no_smiles", False),
            )
        elif xml_path.exists():
            logger.info("CSV not found, falling back to XML parser")
            drugs_df, interactions_df, targets_df = parse_drugbank_from_config(cfg, args.max_drugs)
        else:
            raise FileNotFoundError(
                "DrugBank file not found.\n"
                f"  CSV: {csv_path}\n"
                f"  XML: {xml_path}\n"
                "Place drugbank_full_database.csv in data/raw/"
            )

        results["DrugBank"] = f"✓ {len(drugs_df):,} drugs, {len(interactions_df):,} interactions"
        timings["DrugBank"] = time.time() - t0
        logger.success(results["DrugBank"])
    except FileNotFoundError as e:
        results["DrugBank"] = "✗ DrugBank file not found"
        timings["DrugBank"] = time.time() - t0
        logger.error(str(e))
        console.print("[red]DrugBank file not found. Continuing with other steps.[/red]")
    except Exception as e:
        results["DrugBank"] = f"✗ Error: {type(e).__name__}: {e}"
        timings["DrugBank"] = time.time() - t0
        logger.exception(f"DrugBank parsing failed: {e}")

    # ── Step 2: TWOSIDES ──────────────────────────────────────────────────────
    if not args.skip_twosides:
        console.rule("[bold]Step 2/5 — TWOSIDES via PyTDC")
        t0 = time.time()
        try:
            from pipeline.load_twosides import load_twosides

            twosides_df = load_twosides(output_dir=processed_dir)
            results["TWOSIDES"] = f"✓ {len(twosides_df):,} unique drug pairs"
            timings["TWOSIDES"] = time.time() - t0
            logger.success(results["TWOSIDES"])
        except Exception as e:
            results["TWOSIDES"] = f"✗ Error: {type(e).__name__}: {e}"
            timings["TWOSIDES"] = time.time() - t0
            logger.exception(f"TWOSIDES loading failed: {e}")
    else:
        results["TWOSIDES"] = "— Skipped"
        timings["TWOSIDES"] = 0.0

    # ── Step 3: FAERS ─────────────────────────────────────────────────────────
    if not args.skip_faers:
        console.rule("[bold]Step 3/5 — FAERS Harm Signals")
        t0 = time.time()
        faers_root = ROOT / "data" / "raw" / "faers"
        if not faers_root.exists() or not any(faers_root.iterdir()):
            results["FAERS"] = "⚠ Not downloaded — run scripts/download_all.py first"
            timings["FAERS"] = 0.0
            logger.warning("FAERS data not found. Run: python scripts/download_all.py")
        else:
            try:
                from pipeline.load_faers import compute_faers_harm_signals

                drugs_parquet = processed_dir / "drugs.parquet"
                harm_df = compute_faers_harm_signals(
                    faers_root=faers_root,
                    output_dir=processed_dir,
                    drugs_parquet=drugs_parquet if drugs_parquet.exists() else None,
                )
                results["FAERS"] = f"✓ {len(harm_df):,} drug-pair harm signals"
                timings["FAERS"] = time.time() - t0
                logger.success(results["FAERS"])
            except Exception as e:
                results["FAERS"] = f"✗ Error: {type(e).__name__}: {e}"
                timings["FAERS"] = time.time() - t0
                logger.exception(f"FAERS processing failed: {e}")
    else:
        results["FAERS"] = "— Skipped"
        timings["FAERS"] = 0.0

    # ── Step 4: Molecular Graphs ───────────────────────────────────────────────
    if not args.skip_graphs:
        console.rule("[bold]Step 4/5 — Molecular Graph Construction")
        t0 = time.time()
        drugs_parquet = processed_dir / "drugs.parquet"
        if not drugs_parquet.exists():
            results["Mol Graphs"] = "⚠ Skipped — drugs.parquet not available"
            timings["Mol Graphs"] = 0.0
        else:
            try:
                from pipeline.build_molecular_graphs import build_molecular_graphs

                mol_graph_dir = graphs_dir / "molecular"
                index = build_molecular_graphs(
                    drugs_parquet=drugs_parquet,
                    output_dir=mol_graph_dir,
                )
                results["Mol Graphs"] = f"✓ {len(index):,} molecular graphs"
                timings["Mol Graphs"] = time.time() - t0
                logger.success(results["Mol Graphs"])
            except Exception as e:
                results["Mol Graphs"] = f"✗ Error: {type(e).__name__}: {e}"
                timings["Mol Graphs"] = time.time() - t0
                logger.exception(f"Molecular graph construction failed: {e}")
    else:
        results["Mol Graphs"] = "— Skipped"
        timings["Mol Graphs"] = 0.0

    # ── Step 5: DDI Knowledge Graph ───────────────────────────────────────────
    if not args.skip_ddi_graph:
        console.rule("[bold]Step 5/5 — DDI Knowledge Graph")
        t0 = time.time()
        drugs_parquet = processed_dir / "drugs.parquet"
        if not drugs_parquet.exists():
            results["DDI Graph"] = "⚠ Skipped — drugs.parquet not available"
            timings["DDI Graph"] = 0.0
        else:
            try:
                from pipeline.build_ddi_graph import build_ddi_knowledge_graph

                ddi_graph = build_ddi_knowledge_graph(
                    processed_dir=processed_dir,
                    graphs_dir=graphs_dir,
                    cfg=cfg,
                )
                num_drugs = ddi_graph["drug"].num_nodes
                num_ddi = ddi_graph["drug", "interacts_with", "drug"].edge_index.shape[1]
                results["DDI Graph"] = f"✓ {num_drugs:,} drugs, {num_ddi:,} DDI edges"
                timings["DDI Graph"] = time.time() - t0
                logger.success(results["DDI Graph"])
            except Exception as e:
                results["DDI Graph"] = f"✗ Error: {type(e).__name__}: {e}"
                timings["DDI Graph"] = time.time() - t0
                logger.exception(f"DDI graph construction failed: {e}")
    else:
        results["DDI Graph"] = "— Skipped"
        timings["DDI Graph"] = 0.0

    # ── Summary Table ──────────────────────────────────────────────────────────
    console.rule("[bold]Pipeline Complete")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Step", style="cyan", width=15)
    table.add_column("Result", width=50)
    table.add_column("Time (s)", justify="right", width=10)

    for step, result in results.items():
        timing_str = f"{timings[step]:.1f}s" if timings.get(step, 0) > 0 else "—"
        color = "green" if result.startswith("✓") else "yellow" if result.startswith("⚠") else "red" if result.startswith("✗") else "dim"
        table.add_row(step, f"[{color}]{result}[/{color}]", timing_str)

    console.print(table)
    total_time = sum(timings.values())
    console.print(f"\nTotal pipeline time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    console.print("\nNext step: [bold]python train.py[/bold]")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
