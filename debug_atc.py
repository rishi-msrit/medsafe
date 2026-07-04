"""
Quick check: what ATC codes do Simvastatin + bad suggestions have?
Run: python debug_atc.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from configs.loader import load_config

cfg = load_config()
drug_path = Path(cfg.paths.data_processed) / "drugs.parquet"
df = pd.read_parquet(drug_path)

DRUGS = ["Simvastatin", "Milrinone", "Moexipril", "Zofenopril",
         "Atorvastatin", "Pravastatin", "Rosuvastatin", "Lovastatin"]

atc_cols = [c for c in df.columns if "atc" in c.lower()]
print(f"\nATC columns available: {atc_cols}\n")
print(f"{'Drug':<20} {'DB ID':<12} {' '.join(f'{c:<20}' for c in atc_cols)}")
print("-" * 120)

for drug in DRUGS:
    row = df[df["name"].str.lower() == drug.lower()]
    if row.empty:
        print(f"{drug:<20} NOT FOUND")
        continue
    r = row.iloc[0]
    db_id = r.get("drugbank_id", r.get("drug_id", "?"))
    vals = "  ".join(f"{str(r.get(c, ''))[:18]:<20}" for c in atc_cols)
    print(f"{drug:<20} {str(db_id):<12} {vals}")
