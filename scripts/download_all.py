"""
MedSafe — scripts/download_all.py
==================================
One-stop setup script for all automated dataset downloads.

Usage:
    python scripts/download_all.py [--skip-faers] [--faers-quarters N] [--data-dir PATH]

Handles:
  1. OGBL-DDI        → auto via OGB library
  2. TWOSIDES        → auto via PyTDC
  3. FAERS           → pulls latest quarterly ZIP files from FDA public server
  4. DrugBank        → prints clear manual instructions (requires registration)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

console = Console()

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path("data")
FAERS_BASE_URL = "https://fis.fda.gov/content/Exports"
FAERS_INDEX_URL = "https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html"

# Known recent FAERS quarterly file naming pattern
# Pattern: faers_ascii_<YYYY>q<Q>.zip
FAERS_QUARTERS = [
    ("2024", "q4"),
    ("2024", "q3"),
    ("2024", "q2"),
    ("2024", "q1"),
    ("2023", "q4"),
    ("2023", "q3"),
    ("2023", "q2"),
    ("2023", "q1"),
]


# ─── OGBL-DDI Download ────────────────────────────────────────────────────────

def download_ogbl_ddi(data_dir: Path) -> bool:
    """Download OGBL-DDI via OGB library. Saves to OGB's default cache."""
    console.rule("[bold cyan]1/4  OGBL-DDI")
    try:
        from ogb.linkproppred import PygLinkPropPredDataset

        logger.info("Downloading OGBL-DDI via OGB (this may take a few minutes)...")
        dataset = PygLinkPropPredDataset(name="ogbl-ddi", root=str(data_dir / "ogb"))
        graph = dataset[0]
        console.print(
            f"[green]✓ OGBL-DDI downloaded.[/green] "
            f"Nodes: {graph.num_nodes:,}  |  "
            f"Edges: {graph.num_edges:,}"
        )
        return True
    except ImportError:
        logger.error("OGB not installed. Run: pip install ogb")
        return False
    except Exception as e:
        logger.error(f"OGBL-DDI download failed: {e}")
        return False


# ─── TWOSIDES Download ────────────────────────────────────────────────────────

def download_twosides(data_dir: Path) -> bool:
    """Download TWOSIDES via PyTDC library."""
    console.rule("[bold cyan]2/4  TWOSIDES")
    try:
        from tdc.multi_pred import DDI

        logger.info("Downloading TWOSIDES via PyTDC (this may take several minutes — ~500 MB)...")
        tdc_data_dir = str(data_dir / "tdc")
        os.makedirs(tdc_data_dir, exist_ok=True)
        data = DDI(name="TWOSIDES", path=tdc_data_dir)
        df = data.get_data()
        console.print(
            f"[green]✓ TWOSIDES downloaded.[/green] "
            f"Records: {len(df):,}  |  "
            f"Unique drug pairs: {df[['Drug1_ID', 'Drug2_ID']].drop_duplicates().shape[0]:,}"
        )
        return True
    except ImportError:
        logger.error("PyTDC not installed. Run: pip install PyTDC")
        return False
    except Exception as e:
        logger.error(f"TWOSIDES download failed: {e}")
        return False


# ─── FAERS Download ───────────────────────────────────────────────────────────

def _get_faers_file_url(year: str, quarter: str) -> str:
    """Construct the FDA FAERS ASCII download URL for a given quarter."""
    return f"{FAERS_BASE_URL}/faers_ascii_{year}{quarter}.zip"


def _download_file_with_progress(url: str, dest: Path) -> bool:
    """Stream-download a file with a rich progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        with Progress(
            TextColumn("[bold blue]{task.fields[filename]}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "Downloading", total=total, filename=dest.name
            )
            with open(dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))
        return True
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"  FAERS file not found at {url} (quarter may not be released yet)")
        else:
            logger.error(f"  HTTP error downloading {url}: {e}")
        return False
    except requests.RequestException as e:
        logger.error(f"  Network error downloading {url}: {e}")
        return False


def download_faers(data_dir: Path, num_quarters: int = 4) -> bool:
    """Download the most recent N quarterly FAERS ASCII files from FDA."""
    console.rule("[bold cyan]3/4  FAERS (FDA Adverse Event Reporting System)")
    faers_dir = data_dir / "raw" / "faers"
    faers_dir.mkdir(parents=True, exist_ok=True)

    quarters_to_download = FAERS_QUARTERS[:num_quarters]
    downloaded = 0
    skipped = 0

    for year, quarter in quarters_to_download:
        filename = f"faers_ascii_{year}{quarter}.zip"
        dest_zip = faers_dir / filename
        dest_dir = faers_dir / f"{year}{quarter}"

        # Skip if already extracted
        if dest_dir.exists() and any(dest_dir.iterdir()):
            logger.info(f"  Skipping {filename} — already extracted to {dest_dir}")
            skipped += 1
            continue

        url = _get_faers_file_url(year, quarter)
        logger.info(f"  Downloading {filename} from FDA...")

        if not dest_zip.exists():
            success = _download_file_with_progress(url, dest_zip)
            if not success:
                logger.warning(f"  Skipping {filename}")
                continue

        # Extract ZIP
        try:
            logger.info(f"  Extracting {filename}...")
            with zipfile.ZipFile(dest_zip, "r") as z:
                z.extractall(dest_dir)
            console.print(f"[green]  ✓ {filename} extracted to {dest_dir.name}/[/green]")
            downloaded += 1
            # Remove ZIP after extraction to save space
            dest_zip.unlink()
        except zipfile.BadZipFile:
            logger.error(f"  {filename} is corrupted. Deleting and retrying next run.")
            dest_zip.unlink(missing_ok=True)

    total = downloaded + skipped
    if total > 0:
        console.print(
            f"[green]✓ FAERS: {downloaded} quarter(s) downloaded, "
            f"{skipped} already present. "
            f"Total: {total}/{num_quarters} quarters ready.[/green]"
        )
        return True
    else:
        logger.error("No FAERS quarters could be downloaded.")
        return False


# ─── DrugBank Instructions ────────────────────────────────────────────────────

def print_drugbank_instructions(data_dir: Path) -> None:
    """Print clear manual instructions for DrugBank download."""
    console.rule("[bold cyan]4/4  DrugBank (Manual Download Required)")

    expected_path = data_dir / "raw" / "drugbank_full_database.xml"
    already_present = expected_path.exists()

    if already_present:
        size_mb = expected_path.stat().st_size / (1024 * 1024)
        console.print(
            Panel(
                f"[green]✓ DrugBank XML already present![/green]\n"
                f"  Path: {expected_path}\n"
                f"  Size: {size_mb:.1f} MB\n\n"
                f"The pipeline will use this file automatically.",
                title="DrugBank Status",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[bold yellow]DrugBank requires a free account to download.[/bold yellow]\n\n"
                "[bold]Steps:[/bold]\n"
                "  1. Go to: [link=https://go.drugbank.com/users/sign_up]https://go.drugbank.com/users/sign_up[/link]\n"
                "  2. Register for a free academic/researcher account\n"
                "  3. After email verification, go to:\n"
                "     [link=https://go.drugbank.com/releases/latest]https://go.drugbank.com/releases/latest[/link]\n"
                "  4. Download: [bold]DrugBank Complete Database[/bold] (XML format, ~1.5 GB)\n"
                "  5. Rename the file to: [bold]drugbank_full_database.xml[/bold]\n"
                f"  6. Place it at: [bold]{expected_path}[/bold]\n\n"
                "[yellow]Note:[/yellow] The pipeline will fail with a clear error if this file "
                "is missing. Everything else can proceed without it.",
                title="⚠  DrugBank Manual Download Required",
                border_style="yellow",
            )
        )


# ─── Summary Table ────────────────────────────────────────────────────────────

def print_summary(results: dict[str, bool | str]) -> None:
    """Print a rich summary table of all download statuses."""
    console.rule("[bold]Download Summary")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset", style="cyan", width=20)
    table.add_column("Status", width=12)
    table.add_column("Notes", style="dim")

    status_map = {
        True: "[green]✓ Ready[/green]",
        False: "[red]✗ Failed[/red]",
        "manual": "[yellow]⚠ Manual[/yellow]",
        "skipped": "[dim]─ Skipped[/dim]",
    }

    for dataset, status in results.items():
        table.add_row(dataset, status_map.get(status, str(status)), "")

    console.print(table)
    console.print()

    all_auto_ok = all(v is True for k, v in results.items() if k != "DrugBank")
    if all_auto_ok:
        console.print(
            "[bold green]All automated downloads complete![/bold green] "
            "Place the DrugBank XML and run: [bold]python pipeline/run_pipeline.py[/bold]"
        )
    else:
        console.print(
            "[bold yellow]Some downloads failed. Check logs above and retry.[/bold yellow]"
        )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedSafe — One-stop dataset downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-faers",
        action="store_true",
        help="Skip FAERS download (large files, ~500 MB per quarter)",
    )
    parser.add_argument(
        "--faers-quarters",
        type=int,
        default=4,
        metavar="N",
        help="Number of recent FAERS quarters to download (default: 4)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--skip-ogbl",
        action="store_true",
        help="Skip OGBL-DDI download",
    )
    parser.add_argument(
        "--skip-twosides",
        action="store_true",
        help="Skip TWOSIDES download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            "[bold cyan]MedSafe — Dataset Setup[/bold cyan]\n"
            "Polypharmacy Drug Interaction Intelligence System\n\n"
            f"Data directory: {data_dir.resolve()}",
            border_style="cyan",
        )
    )
    console.print()

    results: dict[str, bool | str] = {}

    # 1. OGBL-DDI
    if args.skip_ogbl:
        results["OGBL-DDI"] = "skipped"
    else:
        results["OGBL-DDI"] = download_ogbl_ddi(data_dir)

    # 2. TWOSIDES
    if args.skip_twosides:
        results["TWOSIDES"] = "skipped"
    else:
        results["TWOSIDES"] = download_twosides(data_dir)

    # 3. FAERS
    if args.skip_faers:
        results["FAERS"] = "skipped"
        console.rule("[bold cyan]3/4  FAERS")
        console.print("[dim]Skipped via --skip-faers flag[/dim]")
    else:
        results["FAERS"] = download_faers(data_dir, num_quarters=args.faers_quarters)

    # 4. DrugBank (manual — always print instructions)
    print_drugbank_instructions(data_dir)
    drugbank_path = data_dir / "raw" / "drugbank_full_database.xml"
    results["DrugBank"] = "manual" if not drugbank_path.exists() else True

    # Summary
    print_summary(results)


if __name__ == "__main__":
    main()
