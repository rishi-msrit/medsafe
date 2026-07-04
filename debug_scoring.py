"""
debug_scoring.py  — Full diagnostic using the same data paths as the API.
Run: python debug_scoring.py
"""
import sys, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scoring.polypharmacy_score import compute_polypharmacy_score, compute_pair_severity

SEP = "=" * 65

TEST_COMBOS = [
    ["Fluoxetine", "Tramadol", "Linezolid"],
    ["Warfarin", "Fluconazole"],
    ["Warfarin", "Amiodarone", "Simvastatin"],
    ["Metformin", "Lisinopril", "Atorvastatin"],
    ["Carbamazepine", "Warfarin"],
]

# ── 1. Load drugs.parquet ─────────────────────────────────────────────────────
print(SEP)
print("1. Loading drugs.parquet ...")
drugs_path = ROOT / "data" / "processed" / "drugs.parquet"
drugs_df = pd.read_parquet(drugs_path)
print(f"   Rows: {len(drugs_df):,}   Columns: {list(drugs_df.columns)}")

# Build name → ID map
drug_name_to_id: dict[str, str] = {}
for _, row in drugs_df.iterrows():
    name = str(row.get("name", "")).lower().strip()
    did  = str(row.get("drugbank_id", "")).strip()
    if name and did:
        drug_name_to_id[name] = did
    syns = row.get("synonyms", "")
    if syns:
        for s in str(syns).split("|"):
            s = s.strip().lower()
            if s:
                drug_name_to_id[s] = did

print(f"   Name→ID map: {len(drug_name_to_id):,} entries")

# ── 2. Drug resolution ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("2. Drug name resolution:")
all_test_drugs = sorted(set(d for combo in TEST_COMBOS for d in combo))
for drug in all_test_drugs:
    did = drug_name_to_id.get(drug.lower())
    status = f"✓  {did}" if did else "✗  NOT FOUND"
    print(f"   {drug:<22} → {status}")

# ── 3. Load interactions_drugbank.parquet ─────────────────────────────────────
print(f"\n{SEP}")
print("3. Loading interactions_drugbank.parquet ...")
idf_path = ROOT / "data" / "processed" / "interactions_drugbank.parquet"
idf = pd.read_parquet(idf_path)
print(f"   Rows: {len(idf):,}   Columns: {list(idf.columns)}")
print(f"   Sample:\n{idf.head(3).to_string()}")

# Build interactions_lookup (same as API)
interactions_lookup: dict = {}
for _, row in idf.iterrows():
    key = tuple(sorted([str(row["drug1_id"]), str(row["drug2_id"])]))
    interactions_lookup[key] = {
        "severity":       int(row.get("severity", 1)),
        "mechanism_type": str(row.get("mechanism_type", "unknown")),
        "description":    str(row.get("description", "")),
        "confidence":     float(row.get("confidence", 0.9)),
        "support_count":  int(row.get("support_count", 1)),
    }
print(f"\n   Lookup built: {len(interactions_lookup):,} pairs")

# ── 4. Pair-level lookup for each combo ──────────────────────────────────────
print(f"\n{SEP}")
print("4. Pair-level lookup for test combos:")
for combo in TEST_COMBOS:
    print(f"\n   [{' + '.join(combo)}]")
    for da, db in itertools.combinations(combo, 2):
        id_a = drug_name_to_id.get(da.lower(), da.lower())
        id_b = drug_name_to_id.get(db.lower(), db.lower())
        key  = tuple(sorted([id_a, id_b]))
        rec  = interactions_lookup.get(key)
        if rec:
            sev  = rec["severity"]
            conf = rec["confidence"]
            mech = rec["mechanism_type"]
            print(f"     ✓  {da} + {db}:  sev={sev}, conf={conf:.2f}, mech={mech}")
        else:
            # Try name-based key fallback
            key2 = tuple(sorted([da.lower(), db.lower()]))
            rec2 = interactions_lookup.get(key2)
            if rec2:
                print(f"     ✓  {da} + {db}:  (name-key) sev={rec2['severity']}, conf={rec2['confidence']:.2f}")
            else:
                print(f"     ✗  {da} + {db}:  NOT IN LOOKUP  (tried: {key})")

# ── 5. Full scoring ───────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("5. Full polypharmacy scoring:")
for combo in TEST_COMBOS:
    id_map = {}
    for d in combo:
        did = drug_name_to_id.get(d.lower())
        if did:
            id_map[d.lower()] = did
            id_map[d]         = did

    report = compute_polypharmacy_score(
        drug_names=combo,
        drug_id_map=id_map,
        interactions_lookup=interactions_lookup,
        include_shapley=False,
    )
    print(f"\n   {' + '.join(combo)}")
    print(f"   Score: {report.overall_risk_score:.1f}/100  [{report.risk_tier}]")
    print(f"   Flagged: {report.num_flagged}/{report.num_pairs_checked} pairs checked")
    if report.flagged_interactions:
        for p in report.flagged_interactions:
            print(f"     → {p.drug_a} + {p.drug_b}: sev={p.severity} ({p.severity_label}), conf={p.confidence:.2f}")
    else:
        print(f"     (all pairs returned sev=0 — not in lookup or model returned 0)")

# ── 6. Check a sample of what IS in the lookup (to see key format) ────────────
print(f"\n{SEP}")
print("6. First 5 lookup keys (to verify ID format):")
for k, v in list(interactions_lookup.items())[:5]:
    print(f"   {k}  →  sev={v['severity']}, mech={v['mechanism_type'][:40]}")

print(f"\n{SEP}")
print("Diagnostic complete.")
