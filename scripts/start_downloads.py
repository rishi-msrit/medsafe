"""
scripts/start_downloads.py
===========================
Run after pip install completes.
Downloads TWOSIDES + OGBL-DDI + FAERS in sequence.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.makedirs(ROOT / "data" / "raw" / "tdc", exist_ok=True)
os.makedirs(ROOT / "data" / "raw" / "faers", exist_ok=True)

print("=" * 60)
print(" MedSafe Dataset Downloader")
print("=" * 60)

# ── 1. TWOSIDES ────────────────────────────────────────────────
print("\n[1/3] Downloading TWOSIDES (PyTDC)...")
try:
    from tdc.multi_pred import DDI
    data = DDI(name='TWOSIDES', path=str(ROOT / "data" / "raw" / "tdc"))
    df = data.get_data()
    print(f"  OK: {len(df):,} rows")
    out = ROOT / "data" / "raw" / "tdc" / "twosides_raw.parquet"
    df.to_parquet(str(out), index=False)
    print(f"  Saved: {out}")
except Exception as e:
    print(f"  FAILED: {e}")

# ── 2. OGBL-DDI ────────────────────────────────────────────────
print("\n[2/3] Downloading OGBL-DDI (OGB)...")
try:
    from ogb.linkproppred import PygLinkPropPredDataset
    dataset = PygLinkPropPredDataset(
        name='ogbl-ddi',
        root=str(ROOT / "data" / "raw" / "ogbl_ddi")
    )
    print(f"  OK: {dataset.data}")
except Exception as e:
    print(f"  FAILED: {e}")

# ── 3. FAERS ───────────────────────────────────────────────────
print("\n[3/3] Downloading FAERS (FDA FTP)...")
try:
    from scripts.download_all import download_faers
    download_faers()
except Exception as e:
    try:
        # Inline fallback
        import urllib.request
        import zipfile

        FAERS_URL = "https://fis.fda.gov/content/Exports/faers_ascii_{quarter}.zip"
        quarters = ["2024Q2", "2024Q1", "2023Q4", "2023Q3"]
        faers_dir = ROOT / "data" / "raw" / "faers"

        for q in quarters:
            url = FAERS_URL.format(quarter=q)
            out_zip = faers_dir / f"{q}.zip"
            if out_zip.exists():
                print(f"  {q}: already downloaded")
                continue
            print(f"  Downloading {q}...")
            try:
                urllib.request.urlretrieve(url, str(out_zip))
                print(f"  {q}: OK ({out_zip.stat().st_size / 1e6:.1f} MB)")
            except Exception as e2:
                print(f"  {q}: FAILED ({e2})")
    except Exception as e2:
        print(f"  FAERS download failed: {e2}")

print("\n" + "=" * 60)
print("Downloads complete. Next: python pipeline/run_pipeline.py")
print("=" * 60)
