"""
pipeline/load_faers.py
======================
Parse downloaded FDA Adverse Event Reporting System (FAERS) quarterly files.

Computes a "real-world harm signal" per drug pair using:
  - Reporting Odds Ratio (ROR): standard disproportionality analysis method
  - Co-occurrence frequency threshold filter

FAERS files (after extraction from quarterly ZIPs) contain:
  - DRUG{YY}Q{N}.txt  — drug records per adverse event report
  - REAC{YY}Q{N}.txt  — reaction records per adverse event report
  - DEMO{YY}Q{N}.txt  — demographic data per report

Output:
  - data/processed/faers_harm_signals.parquet
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.loader import load_config  # noqa: E402

# Minimum co-occurrences to compute a reliable ROR
MIN_CO_OCCURRENCES = 5

# FAERS ASCII file column specs (may vary slightly by quarter)
DRUG_COLUMNS = ["primaryid", "caseid", "drug_seq", "role_cod", "drugname", "prod_ai"]
REAC_COLUMNS = ["primaryid", "caseid", "pt"]  # PT = preferred term (reaction)


def _find_faers_files(faers_root: Path) -> Iterator[tuple[str, Path]]:
    """Yield (file_type, path) for all FAERS ASCII text files found."""
    for quarter_dir in sorted(faers_root.iterdir()):
        if not quarter_dir.is_dir():
            continue
        for sub in quarter_dir.rglob("*.txt"):
            fname = sub.name.upper()
            if fname.startswith("DRUG"):
                yield ("drug", sub)
            elif fname.startswith("REAC"):
                yield ("reac", sub)


def _read_faers_file(path: Path, expected_columns: list[str]) -> pd.DataFrame | None:
    """Read a FAERS ASCII delimited file robustly."""
    for sep in ["$", "\t", ","]:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                encoding="latin-1",
                low_memory=False,
                on_bad_lines="skip",
                dtype=str,
            )
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]

            # Check required columns are present
            missing = [c for c in expected_columns if c not in df.columns]
            if len(missing) == 0:
                return df
            elif len(missing) <= 2:
                # Try to fill missing with NaN
                for col in missing:
                    df[col] = np.nan
                return df
        except Exception:
            continue
    logger.warning(f"  Could not parse {path.name} with any separator")
    return None


def _normalize_drug_name(name: str) -> str:
    """Normalize drug name for matching: lowercase, strip salts/formulations."""
    if not name or pd.isna(name):
        return ""
    name = str(name).lower().strip()
    # Remove common formulations
    name = re.sub(r"\s+(hcl|hydrochloride|sodium|potassium|mg|mcg|tablet|cap\w*).*$", "", name)
    # Remove punctuation
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def compute_faers_harm_signals(
    faers_root: Path,
    output_dir: Path,
    drugs_parquet: Path | None = None,
) -> pd.DataFrame:
    """
    Compute drug-pair harm signals from FAERS quarterly files.

    Method: Reporting Odds Ratio (ROR) disproportionality analysis.
    ROR = (a/b) / (c/d) where:
      a = reports with both Drug1 + Drug2
      b = reports with Drug1 only
      c = reports with Drug2 only
      d = reports with neither

    Args:
        faers_root:    Root directory of extracted FAERS quarters
        output_dir:    Directory to write output Parquet
        drugs_parquet: Optional DrugBank drugs.parquet for name normalization

    Returns:
        DataFrame with drug pair harm signals
    """
    logger.info(f"Loading FAERS files from: {faers_root}")

    drug_files = []
    for ftype, fpath in _find_faers_files(faers_root):
        if ftype == "drug":
            drug_files.append(fpath)

    if not drug_files:
        raise FileNotFoundError(
            f"No FAERS DRUG*.txt files found under {faers_root}. "
            "Run scripts/download_all.py first."
        )
    logger.info(f"  Found {len(drug_files)} DRUG files")

    # Load known drug names from DrugBank for normalization
    known_drugs: set[str] = set()
    if drugs_parquet and drugs_parquet.exists():
        db_df = pd.read_parquet(drugs_parquet, columns=["name", "synonyms"])
        for _, row in db_df.iterrows():
            known_drugs.add(_normalize_drug_name(row["name"]))
            if row["synonyms"]:
                for syn in str(row["synonyms"]).split("|"):
                    known_drugs.add(_normalize_drug_name(syn))
        logger.info(f"  DrugBank reference: {len(known_drugs):,} normalized names")

    # ── Step 1: Build report_id → set of drugs mapping ────────────────────────
    logger.info("  Building report→drug mapping (this takes a few minutes)...")
    report_to_drugs: dict[str, set[str]] = {}

    for fpath in tqdm(drug_files, desc="FAERS DRUG files"):
        df = _read_faers_file(fpath, DRUG_COLUMNS)
        if df is None:
            continue

        # Primary suspect and concomitant drugs only (not interacting = "I")
        if "role_cod" in df.columns:
            df = df[df["role_cod"].str.upper().isin(["PS", "SS", "C"])].copy()

        for _, row in df.iterrows():
            report_id = str(row.get("primaryid", "")).strip()
            if not report_id:
                continue

            # Use prod_ai (active ingredient) if available, else drugname
            drug_name = str(row.get("prod_ai") or row.get("drugname", "")).strip()
            normalized = _normalize_drug_name(drug_name)

            if not normalized:
                continue

            # If we have DrugBank reference, filter to known drugs only
            if known_drugs and normalized not in known_drugs:
                continue

            if report_id not in report_to_drugs:
                report_to_drugs[report_id] = set()
            report_to_drugs[report_id].add(normalized)

    total_reports = len(report_to_drugs)
    logger.info(f"  Total reports with ≥1 known drug: {total_reports:,}")

    # ── Step 2: Count drug-pair co-occurrences ────────────────────────────────
    logger.info("  Counting drug-pair co-occurrences...")
    pair_counts: dict[tuple[str, str], int] = {}
    drug_counts: dict[str, int] = {}

    for drugs_in_report in tqdm(report_to_drugs.values(), desc="Computing co-occurrences"):
        drugs_list = sorted(drugs_in_report)
        for drug in drugs_list:
            drug_counts[drug] = drug_counts.get(drug, 0) + 1

        for i in range(len(drugs_list)):
            for j in range(i + 1, len(drugs_list)):
                pair = (drugs_list[i], drugs_list[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    logger.info(
        f"  Drug co-occurrences: {len(pair_counts):,} pairs "
        f"({sum(1 for v in pair_counts.values() if v >= MIN_CO_OCCURRENCES):,} with ≥{MIN_CO_OCCURRENCES} reports)"
    )

    # ── Step 3: Compute ROR for each pair ─────────────────────────────────────
    logger.info("  Computing Reporting Odds Ratios...")
    records = []

    N = total_reports

    for (drug1, drug2), ab in tqdm(pair_counts.items(), desc="Computing ROR"):
        if ab < MIN_CO_OCCURRENCES:
            continue

        a = ab  # reports with both
        b = drug_counts.get(drug1, 0) - a  # drug1 only
        c = drug_counts.get(drug2, 0) - a  # drug2 only
        d = N - a - b - c  # neither

        if b <= 0 or c <= 0 or d <= 0:
            continue

        # ROR calculation
        ror = (a / b) / (c / d)
        ror_log = np.log(ror)

        # 95% CI (Woolf method on log scale)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        ror_lower = np.exp(ror_log - 1.96 * se)
        ror_upper = np.exp(ror_log + 1.96 * se)

        # Signal if lower bound > 1 (conventional threshold)
        is_signal = ror_lower > 1.0

        # Normalize harm signal to 0–1 for use as auxiliary label
        # Use min-max normalization later; store raw ROR now
        records.append({
            "drug1_name": drug1,
            "drug2_name": drug2,
            "co_occurrences": a,
            "ror": float(ror),
            "ror_lower": float(ror_lower),
            "ror_upper": float(ror_upper),
            "is_signal": is_signal,
            "total_reports": N,
        })

    harm_df = pd.DataFrame(records)

    if len(harm_df) > 0:
        # Normalize ROR to 0–3 scale for compatibility with severity labels
        # Use 95th percentile as cap
        cap = harm_df["ror"].quantile(0.95)
        harm_df["faers_harm_score"] = (harm_df["ror"] / cap * 3).clip(0, 3)
    else:
        harm_df["faers_harm_score"] = 0.0

    logger.info(
        f"  Harm signals computed: {len(harm_df):,} pairs "
        f"({harm_df['is_signal'].sum():,} with ROR lower bound > 1)"
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "faers_harm_signals.parquet"
    harm_df.to_parquet(out_path, index=False)
    logger.info(f"  Saved: {out_path}")

    return harm_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute FAERS harm signals")
    parser.add_argument("--faers-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config()
    faers_root = args.faers_dir or ROOT / "data" / "raw" / "faers"
    output_dir = ROOT / cfg.paths.data_processed
    drugs_parquet = output_dir / "drugs.parquet"

    harm_df = compute_faers_harm_signals(
        faers_root=faers_root,
        output_dir=output_dir,
        drugs_parquet=drugs_parquet if drugs_parquet.exists() else None,
    )
    logger.success(f"FAERS processing complete: {len(harm_df):,} drug-pair harm signals")
