"""
pipeline/parse_drugbank.py
==========================
Parse DrugBank Complete Database CSV into structured DataFrames.

DrugBank provides a flat CSV (drugbank_full_database.csv) with 40 columns.
The drug-interactions column contains space-separated DrugBank IDs of interacting drugs.
SMILES are not included in the CSV — we fetch them from PubChem via PubChemPy.

Extracts:
  - Drug metadata (name, ID, groups, ATC class, categories, description)
  - Drug-drug interactions (inferred severity from description text)
  - CYP450 enzyme relationships (parsed from 'enzymes' column)
  - Drug-target relationships (parsed from 'targets' column)
  - SMILES via PubChem lookup (batched, with local cache)

Output (Parquet):
  - data/processed/drugs.parquet
  - data/processed/interactions_drugbank.parquet
  - data/processed/drug_targets.parquet
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.loader import load_config  # noqa: E402

# ─── CYP450 enzymes to track ──────────────────────────────────────────────────
CYP_ENZYMES = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
ENZYME_ROLES = {"substrate", "inhibitor", "inducer"}

# ─── Severity keywords for interaction text ───────────────────────────────────
SEVERITY_MAP = {
    "contraindicated": 3,
    "do not use": 3,
    "major": 2,
    "significant": 2,
    "clinically significant": 2,
    "not recommended": 2,
    "moderate": 1,
    "minor": 0,
    "minimal": 0,
}

# ─── Mechanism pattern detection ─────────────────────────────────────────────
_MECHANISM_PATTERNS = [
    (r"CYP[0-9][A-Z][0-9]+", "cyp450_metabolic"),
    (r"metab\w+", "metabolic"),
    (r"absorpt\w+|bioavailab\w+", "absorption"),
    (r"transport\w+|p-glycoprotein|pgp", "transporter"),
    (r"QT|cardiac|arrhythmia|torsade", "cardiac_qt"),
    (r"bleed\w+|hemorrhag\w+|anticoagul\w+|platelet", "bleeding"),
    (r"CNS|sedati\w+|depress\w+", "cns_depression"),
    (r"nephrotox\w+|renal", "renal"),
    (r"seroton\w+", "serotonin_syndrome"),
    (r"pharmacodynam\w+|synerg\w+|antagoni\w+", "pharmacodynamic"),
]


def _normalize_severity(text: str | None) -> int:
    """Map severity text → integer 0–3. Defaults to 1 (moderate) if unknown."""
    if not text:
        return 1
    text_lower = text.strip().lower()
    for key, val in SEVERITY_MAP.items():
        if key in text_lower:
            return val
    return 1


def _classify_mechanism(text: str) -> str:
    """Classify interaction mechanism from free text."""
    if not text:
        return "unknown"
    for pattern, mtype in _MECHANISM_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return mtype
    return "unknown"


def _extract_atc_level1(atc_text: str) -> str:
    """
    Extract first ATC level letter from DrugBank CSV atc-codes column.
    The column contains long text like:
    'BLOOD AND BLOOD FORMING ORGANS ANTITHROMBOTIC AGENTS Direct thrombin inhibitors ...'

    We map known ATC anatomical main groups.
    """
    if not atc_text or pd.isna(atc_text):
        return ""

    atc_text = str(atc_text).upper()

    # Try to find actual ATC code pattern (e.g. B01AE06)
    code_match = re.search(r'\b([A-Z])(\d{2}[A-Z]{2}\d{2})\b', atc_text)
    if code_match:
        return code_match.group(1)

    # Fall back to anatomical group keyword mapping
    ATC_KEYWORDS = {
        "A": ["alimentary", "metabolism", "gastro", "antidiabetic", "vitamin"],
        "B": ["blood", "hematolog", "antithrombotic", "antianemia"],
        "C": ["cardiovascular", "cardiac", "antihypertensive", "lipid"],
        "D": ["dermatolog", "skin"],
        "G": ["genito", "urinary", "sex hormone"],
        "H": ["hormone", "thyroid", "pituitary", "adrenal"],
        "J": ["antiinfective", "antibiotic", "antiviral", "antifungal", "antimicrobial"],
        "L": ["antineoplastic", "immunosuppressant", "oncolog"],
        "M": ["musculoskeletal", "anti-inflammatory", "nsaid", "gout"],
        "N": ["nervous system", "analgesic", "anesthetic", "psycholeptic", "antiepileptic",
              "opioid", "antidepressant", "antipsychotic"],
        "P": ["antiparasitic", "antimalarial"],
        "R": ["respiratory", "antihistamine", "bronchodilator"],
        "S": ["sensory organ", "ophthalmolog", "otolog"],
        "V": ["various", "diagnostic", "contrast"],
    }
    for code, keywords in ATC_KEYWORDS.items():
        if any(kw in atc_text.lower() for kw in keywords):
            return code
    return ""


def _parse_cyp_from_text(enzymes_text: str) -> dict[str, dict[str, bool]]:
    """
    Parse CYP450 relationships from the 'enzymes' column text.

    The column contains pipe-separated or space-separated enzyme entries
    that look like: 'CYP3A4 substrate inhibitor | CYP2D6 substrate'
    """
    cyp_flags: dict[str, dict[str, bool]] = {
        cyp: {"substrate": False, "inhibitor": False, "inducer": False}
        for cyp in CYP_ENZYMES
    }

    if not enzymes_text or pd.isna(enzymes_text):
        return cyp_flags

    text = str(enzymes_text).upper()
    for cyp in CYP_ENZYMES:
        if cyp in text:
            # Find the segment around this CYP mention
            idx = text.find(cyp)
            segment = text[idx:idx + 80].lower()
            if "substrate" in segment:
                cyp_flags[cyp]["substrate"] = True
            if "inhibitor" in segment:
                cyp_flags[cyp]["inhibitor"] = True
            if "inducer" in segment:
                cyp_flags[cyp]["inducer"] = True

    return cyp_flags


def _fetch_smiles_pubchem(drug_names: list[str]) -> dict[str, str]:
    """
    Batch fetch SMILES from PubChem by drug name.
    Returns {name: smiles} dict. Missing drugs get empty string.
    Uses local cache to avoid redundant API calls.
    """
    cache_path = ROOT / "data" / "processed" / "smiles_cache.parquet"
    cache: dict[str, str] = {}

    # Load existing cache
    if cache_path.exists():
        cache_df = pd.read_parquet(cache_path)
        cache = dict(zip(cache_df["name"], cache_df["smiles"]))
        logger.info(f"SMILES cache loaded: {len(cache):,} entries")

    to_fetch = [n for n in drug_names if n not in cache]
    if not to_fetch:
        return {n: cache.get(n, "") for n in drug_names}

    logger.info(f"Fetching SMILES for {len(to_fetch):,} drugs from PubChem...")

    try:
        import pubchempy as pcp
    except ImportError:
        logger.warning("pubchempy not installed. SMILES will be empty. Run: pip install pubchempy")
        return {n: cache.get(n, "") for n in drug_names}

    batch_size = 50
    new_entries = []

    for i in tqdm(range(0, len(to_fetch), batch_size), desc="PubChem SMILES"):
        batch = to_fetch[i:i + batch_size]
        for name in batch:
            try:
                compounds = pcp.get_compounds(name, "name", record_type="2d")
                smiles = compounds[0].isomeric_smiles if compounds else ""
                cache[name] = smiles
                new_entries.append({"name": name, "smiles": smiles})
            except Exception:
                cache[name] = ""
                new_entries.append({"name": name, "smiles": ""})
        # Polite rate limiting
        time.sleep(0.2)

    # Save updated cache
    if new_entries:
        new_df = pd.DataFrame(new_entries)
        if cache_path.exists():
            old_df = pd.read_parquet(cache_path)
            combined = pd.concat([old_df, new_df]).drop_duplicates(subset=["name"])
        else:
            combined = new_df
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(cache_path, index=False)
        logger.info(f"SMILES cache saved: {len(combined):,} entries → {cache_path}")

    return {n: cache.get(n, "") for n in drug_names}


def parse_drugbank(
    csv_path: Path,
    output_dir: Path,
    max_drugs: int | None = None,
    fetch_smiles: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse DrugBank CSV and save structured outputs as Parquet files.

    Args:
        csv_path:    Path to drugbank_full_database.csv
        output_dir:  Directory to write Parquet files
        max_drugs:   Optional row limit for testing (None = all)
        fetch_smiles: Whether to fetch SMILES from PubChem (can take 30+ min for full DB)

    Returns:
        Tuple of (drugs_df, interactions_df, targets_df)
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"\n\nDrugBank CSV not found at: {csv_path}\n"
            "Please place drugbank_full_database.csv in data/raw/\n"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading DrugBank CSV: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(str(csv_path), low_memory=False)
    logger.info(f"Loaded {len(df):,} total drug entries")

    # Keep only small molecules (filter out biologics, which have no SMILES)
    sm_mask = df["type"].str.lower().str.strip() == "small molecule"
    df_sm = df[sm_mask].copy().reset_index(drop=True)
    logger.info(f"Small molecules: {len(df_sm):,}")

    if max_drugs:
        df_sm = df_sm.head(max_drugs)
        logger.info(f"Limited to {max_drugs} drugs for testing")

    # ── Build drugs DataFrame ─────────────────────────────────────────────────
    drugs_records = []
    interactions_records = []
    targets_records = []

    for _, row in tqdm(df_sm.iterrows(), total=len(df_sm), desc="Parsing drugs"):
        drug_id = str(row.get("drugbank-id", "")).strip()
        name    = str(row.get("name", "")).strip()

        if not drug_id or not name or drug_id == "nan":
            continue

        # ATC
        atc_text  = str(row.get("atc-codes", ""))
        atc_level1 = _extract_atc_level1(atc_text)
        # Try to extract the actual ATC code (e.g., N06AA04)
        atc_code_match = re.search(r'\b[A-Z]\d{2}[A-Z]{2}\d{2}\b', atc_text)
        atc_class = atc_code_match.group(0) if atc_code_match else ""

        # CYP flags
        cyp_flags = _parse_cyp_from_text(str(row.get("enzymes", "")))
        cyp_columns: dict[str, Any] = {}
        for cyp, roles in cyp_flags.items():
            for role, flag in roles.items():
                cyp_columns[f"{cyp}_{role}"] = flag

        # Groups
        groups = str(row.get("groups", "")).strip()

        # All clinical text columns (use all available for mechanism/ATC inference)
        description    = str(row.get("description", "") or "")[:500]
        mechanism      = str(row.get("mechanism-of-action", "") or "")[:500]
        indication     = str(row.get("indication", "") or "")[:300]
        pharmacodyn    = str(row.get("pharmacodynamics", "") or "")[:300]
        metabolism_txt = str(row.get("metabolism", "") or "")[:200]

        # Combined clinical text for mechanism classification (richest source)
        combined_clinical = f"{description} {mechanism} {indication} {pharmacodyn} {metabolism_txt}"

        # Categories: derive from the richest available text
        categories = (indication[:300] if indication and indication != "nan"
                      else (atc_text[:300] if atc_text and atc_text != "nan" else ""))

        drugs_records.append({
            "drugbank_id":      drug_id,
            "name":             name,
            "smiles":           "",  # Will be filled by PubChem below
            "molecular_weight": _safe_float(row.get("average-mass")),
            "description":      description if description != "nan" else "",
            "mechanism":        mechanism if mechanism != "nan" else "",
            "atc_codes":        atc_text[:300] if atc_text != "nan" else "",
            "atc_level1":       atc_level1,
            "atc_class":        atc_class,
            "categories":       categories,
            "groups":           groups if groups != "nan" else "",
            "synonyms":         "",  # Not in CSV format
            **cyp_columns,
        })

        # ── Interactions ──────────────────────────────────────────────────────
        # 'drug-interactions' column = space-separated DrugBank IDs only (no clinical text)
        ddi_text = str(row.get("drug-interactions", ""))
        if ddi_text and ddi_text != "nan":
            partner_ids = [x.strip() for x in ddi_text.split() if x.strip().startswith("DB")]

            for partner_id in partner_ids[:200]:  # Cap at 200 partners per drug
                id1, id2 = sorted([drug_id, partner_id])
                # Mechanism classified from this drug's own clinical text
                # (best proxy available — CSV has no per-pair clinical text)
                interactions_records.append({
                    "drug1_id":       id1,
                    "drug2_id":       id2,
                    "severity":       1,   # Default moderate; enriched by curated pairs
                    "description":    f"{name} interacts with {partner_id}",
                    "mechanism_type": _classify_mechanism(combined_clinical),
                    "source":         "drugbank_csv",
                })

        # ── Targets ───────────────────────────────────────────────────────────
        targets_text = str(row.get("targets", ""))
        if targets_text and targets_text != "nan":
            target_entries = _parse_targets_text(targets_text, drug_id)
            targets_records.extend(target_entries)

    # ── Build DataFrames ──────────────────────────────────────────────────────
    drugs_df = pd.DataFrame(drugs_records).drop_duplicates(subset=["drugbank_id"])
    logger.info(f"Parsed {len(drugs_df):,} small molecule drugs")

    interactions_df = (
        pd.DataFrame(interactions_records)
        .drop_duplicates(subset=["drug1_id", "drug2_id"])
        if interactions_records
        else pd.DataFrame(columns=["drug1_id", "drug2_id", "severity", "description", "mechanism_type"])
    )
    logger.info(f"Parsed {len(interactions_df):,} interaction pairs")

    targets_df = (
        pd.DataFrame(targets_records).drop_duplicates()
        if targets_records
        else pd.DataFrame(columns=["drug_id", "target_id", "target_name"])
    )
    logger.info(f"Parsed {len(targets_df):,} drug-target relations")

    # ── Fetch SMILES from PubChem ─────────────────────────────────────────────
    if fetch_smiles:
        logger.info("Fetching SMILES from PubChem (this may take a while for the full DB)...")
        drug_names = drugs_df["name"].tolist()
        smiles_map = _fetch_smiles_pubchem(drug_names)
        drugs_df["smiles"] = drugs_df["name"].map(smiles_map).fillna("")
        n_with_smiles = (drugs_df["smiles"] != "").sum()
        logger.info(f"SMILES fetched: {n_with_smiles:,}/{len(drugs_df):,} drugs")
    else:
        logger.info("Skipping PubChem SMILES fetch (--no-smiles flag). SMILES will be empty.")

    # ── Save Parquet ──────────────────────────────────────────────────────────
    drugs_path        = output_dir / "drugs.parquet"
    interactions_path = output_dir / "interactions_drugbank.parquet"
    targets_path      = output_dir / "drug_targets.parquet"

    drugs_df.to_parquet(drugs_path, index=False)
    interactions_df.to_parquet(interactions_path, index=False)
    targets_df.to_parquet(targets_path, index=False)

    logger.success(f"Saved: {drugs_path}")
    logger.success(f"Saved: {interactions_path}")
    logger.success(f"Saved: {targets_path}")

    return drugs_df, interactions_df, targets_df


def _safe_float(val) -> float | None:
    """Convert value to float safely."""
    try:
        return float(val) if pd.notna(val) else None
    except (ValueError, TypeError):
        return None


def _parse_targets_text(targets_text: str, drug_id: str) -> list[dict[str, Any]]:
    """Parse target IDs from the 'targets' column."""
    targets = []
    # The column may contain UniProt IDs or target names as space-separated values
    entries = [x.strip() for x in str(targets_text).split() if x.strip() and x != "nan"]
    for entry in entries[:50]:  # Cap at 50 targets per drug
        targets.append({
            "drug_id":    drug_id,
            "target_id":  entry,
            "target_name": entry,
        })
    return targets


# ─── Compatibility shim: update config path reference ────────────────────────

def parse_drugbank_from_config(cfg, max_drugs: int | None = None, fetch_smiles: bool = True):
    """Load DrugBank using paths from config. Handles both CSV and XML."""
    # Try CSV first
    csv_path = ROOT / "data" / "raw" / "drugbank_full_database.csv"
    xml_path = ROOT / cfg.paths.drugbank_xml

    if csv_path.exists():
        logger.info(f"Found DrugBank CSV: {csv_path}")
        return parse_drugbank(
            csv_path=csv_path,
            output_dir=ROOT / cfg.paths.data_processed,
            max_drugs=max_drugs,
            fetch_smiles=fetch_smiles,
        )
    elif xml_path.exists():
        logger.info(f"Found DrugBank XML: {xml_path} (using XML parser)")
        from pipeline._parse_drugbank_xml import parse_drugbank as parse_xml
        return parse_xml(xml_path=xml_path, output_dir=ROOT / cfg.paths.data_processed)
    else:
        raise FileNotFoundError(
            f"DrugBank file not found. Tried:\n"
            f"  CSV: {csv_path}\n"
            f"  XML: {xml_path}\n"
            "Please place drugbank_full_database.csv in data/raw/"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse DrugBank CSV")
    parser.add_argument("--csv", type=Path,
                        default=ROOT / "data" / "raw" / "drugbank_full_database.csv",
                        help="Path to DrugBank CSV")
    parser.add_argument("--max-drugs", type=int, default=None, help="Limit for testing")
    parser.add_argument("--no-smiles", action="store_true", help="Skip PubChem SMILES fetch")
    args = parser.parse_args()

    cfg = load_config()
    output_dir = ROOT / cfg.paths.data_processed

    drugs_df, interactions_df, targets_df = parse_drugbank(
        csv_path=args.csv,
        output_dir=output_dir,
        max_drugs=args.max_drugs,
        fetch_smiles=not args.no_smiles,
    )
    logger.success(
        f"DrugBank parse complete: {len(drugs_df):,} drugs, "
        f"{len(interactions_df):,} interactions, "
        f"{len(targets_df):,} targets"
    )
