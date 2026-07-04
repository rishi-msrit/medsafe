"""
pipeline/load_twosides.py
=========================
Load TWOSIDES dataset via PyTDC and merge with DrugBank drug IDs.

Real schema (verified from PyTDC 0.4.1 download):
    Drug1_ID  : str  — PubChem CID (e.g. 'CID000002173')
    Drug1     : str  — SMILES string of drug 1
    Drug2_ID  : str  — PubChem CID
    Drug2     : str  — SMILES string of drug 2
    Y         : int  — Side effect ID (numeric, not a name string)

TWOSIDES contains 4,649,441 rows (645 unique drugs × multiple side effects).
Y is an integer index into a side-effect vocabulary — we treat it as an
interaction type label (many Y values per drug pair = strong interaction signal).

Output:
  - data/processed/twosides.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.loader import load_config  # noqa: E402


def load_twosides(
    output_dir: Path,
    tdc_data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Download and process TWOSIDES via PyTDC.

    Real TWOSIDES columns from PyTDC 0.4.1:
        Drug1_ID  : PubChem CID string
        Drug1     : SMILES string
        Drug2_ID  : PubChem CID string
        Drug2     : SMILES string
        Y         : int — side effect ID (numeric label)

    We aggregate per (Drug1_ID, Drug2_ID) pair:
      - Count unique side effect IDs → num_side_effects
      - Derive severity_proxy (0–3) from num_side_effects percentile
      - Keep SMILES for downstream molecular graph construction

    Args:
        output_dir:   Directory to write Parquet output
        tdc_data_dir: Where PyTDC stores its cache

    Returns:
        DataFrame with one row per unique drug pair
    """
    try:
        from tdc.multi_pred import DDI
    except ImportError:
        raise ImportError("PyTDC not installed. Run: pip install 'PyTDC==0.4.1'")

    if tdc_data_dir is None:
        tdc_data_dir = ROOT / "data" / "raw" / "tdc"
    tdc_data_dir.mkdir(parents=True, exist_ok=True)

    # Use pre-saved parquet if already downloaded (saves 677MB re-download)
    cached_raw = tdc_data_dir / "twosides_raw.parquet"
    if cached_raw.exists():
        logger.info(f"Loading TWOSIDES from local cache: {cached_raw}")
        df = pd.read_parquet(cached_raw)
    else:
        logger.info("Downloading TWOSIDES via PyTDC (677 MB)...")
        data = DDI(name="TWOSIDES", path=str(tdc_data_dir))
        df = data.get_data()
        df.to_parquet(cached_raw, index=False)
        logger.info(f"Cached to: {cached_raw}")

    logger.info(f"Raw TWOSIDES records: {len(df):,}")
    logger.info(f"Columns: {list(df.columns)}")

    # ── Real column names from PyTDC 0.4.1 ───────────────────────────────────
    # Drug1_ID, Drug1 (SMILES), Drug2_ID, Drug2 (SMILES), Y (int side_effect_id)
    df = df.rename(columns={
        "Drug1_ID": "drug1_pubchem_cid",
        "Drug1":    "drug1_smiles",       # NOTE: Drug1/Drug2 are SMILES, not names
        "Drug2_ID": "drug2_pubchem_cid",
        "Drug2":    "drug2_smiles",
        "Y":        "side_effect_id",     # int, not a name string
    })

    # Ensure canonical pair order (smaller CID first)
    mask = df["drug1_pubchem_cid"] > df["drug2_pubchem_cid"]
    df.loc[mask, ["drug1_pubchem_cid", "drug2_pubchem_cid"]] = (
        df.loc[mask, ["drug2_pubchem_cid", "drug1_pubchem_cid"]].values
    )
    df.loc[mask, ["drug1_smiles", "drug2_smiles"]] = (
        df.loc[mask, ["drug2_smiles", "drug1_smiles"]].values
    )

    # ── Attempt to map PubChem CID → DrugBank ID ─────────────────────────────
    drugbank_path = output_dir / "drugs.parquet"
    smiles_cache  = ROOT / "data" / "processed" / "smiles_cache.parquet"

    # Build SMILES → DrugBank ID map from PubChem cache
    smiles_to_dbid: dict[str, str] = {}
    if smiles_cache.exists():
        sc = pd.read_parquet(smiles_cache)
        # smiles_cache has {name, smiles} — we need smiles→drugbank_id
        # Cross with drugs.parquet
        if drugbank_path.exists():
            db_df = pd.read_parquet(drugbank_path, columns=["drugbank_id", "name", "smiles"])
            for _, row in db_df.iterrows():
                if row["smiles"]:
                    smiles_to_dbid[row["smiles"]] = row["drugbank_id"]

    df["drug1_drugbank_id"] = df["drug1_smiles"].map(smiles_to_dbid).fillna(pd.NA)
    df["drug2_drugbank_id"] = df["drug2_smiles"].map(smiles_to_dbid).fillna(pd.NA)

    matched = (df["drug1_drugbank_id"].notna() & df["drug2_drugbank_id"].notna()).sum()
    logger.info(
        f"DrugBank SMILES match: {matched:,}/{len(df):,} records "
        f"({matched / len(df) * 100:.1f}%)"
    )

    # ── Aggregate: one row per unique drug pair ───────────────────────────────
    logger.info("Aggregating to unique drug pairs...")
    df_agg = (
        df.groupby(
            ["drug1_pubchem_cid", "drug2_pubchem_cid",
             "drug1_smiles", "drug2_smiles"],
            dropna=False,
        )
        .agg(
            num_side_effects  =("side_effect_id", "nunique"),
            side_effect_ids   =("side_effect_id", lambda x: "|".join(str(i) for i in sorted(x.unique())[:50])),
            drug1_drugbank_id =("drug1_drugbank_id", "first"),
            drug2_drugbank_id =("drug2_drugbank_id", "first"),
        )
        .reset_index()
    )

    # Severity proxy: percentile-based mapping to 0–3
    p33 = df_agg["num_side_effects"].quantile(0.33)
    p66 = df_agg["num_side_effects"].quantile(0.66)
    p90 = df_agg["num_side_effects"].quantile(0.90)

    def _sev_proxy(n):
        if n <= p33:   return 0  # minor
        if n <= p66:   return 1  # moderate
        if n <= p90:   return 2  # major
        return 3                  # contraindicated proxy

    df_agg["twosides_severity_proxy"] = df_agg["num_side_effects"].apply(_sev_proxy)

    logger.info(
        f"Unique drug pairs: {len(df_agg):,} "
        f"| Severity distribution: {df_agg['twosides_severity_proxy'].value_counts().to_dict()}"
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "twosides.parquet"
    df_agg.to_parquet(out_path, index=False)
    logger.success(f"Saved TWOSIDES: {out_path} ({len(df_agg):,} rows)")

    return df_agg


if __name__ == "__main__":
    cfg = load_config()
    output_dir = ROOT / cfg.paths.data_processed
    twosides_df = load_twosides(output_dir=output_dir)
    logger.success(f"TWOSIDES load complete: {len(twosides_df):,} unique drug pairs")
