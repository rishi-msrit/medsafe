"""scripts/merge_smiles.py — Merge smiles_cache.parquet into fresh drugs.parquet"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')
import pandas as pd
from pathlib import Path

ROOT = Path('.')
drugs = pd.read_parquet(ROOT / 'data/processed/drugs.parquet')
cache = pd.read_parquet(ROOT / 'data/processed/smiles_cache.parquet')

print(f'Fresh drugs.parquet : {len(drugs):,} drugs')
print(f'SMILES empty before : {(drugs["smiles"].fillna("") == "").sum():,}')
print(f'smiles_cache entries: {len(cache):,}  non-empty: {(cache["smiles"].fillna("") != "").sum():,}')

# Build lookup maps from cache
name_map = {}
id_map = {}
for _, row in cache.iterrows():
    s = str(row.get('smiles', '') or '')
    if s and s != 'nan':
        name = str(row.get('name', '') or '').strip().lower()
        if name:
            name_map[name] = s
        did = str(row.get('drugbank_id', '') or '')
        if did and did != 'nan':
            id_map[did] = s

# Merge SMILES into fresh drugs
filled = 0
for idx, row in drugs.iterrows():
    existing = str(row['smiles'] or '').strip()
    if existing and existing != 'nan':
        continue  # already has SMILES
    # Try drugbank_id first, then name
    s = id_map.get(row['drugbank_id'], '')
    if not s:
        s = name_map.get(str(row['name']).strip().lower(), '')
    if s:
        drugs.at[idx, 'smiles'] = s
        filled += 1

after = (drugs['smiles'].fillna('') != '').sum()
print(f'Filled from cache   : {filled:,}')
print(f'SMILES coverage now : {after:,}/{len(drugs):,} ({after/len(drugs)*100:.1f}%)')

drugs.to_parquet(ROOT / 'data/processed/drugs.parquet', index=False)
print('Saved drugs.parquet.')
