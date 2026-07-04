"""
scripts/fetch_missing_smiles.py
================================
Fetches SMILES for the 535 drugs missing them in drugs.parquet using PubChem.
Updates both smiles_cache.parquet and drugs.parquet in-place.

Run: python scripts/fetch_missing_smiles.py
Resumable: already-fetched entries are skipped on re-run.
"""
import sys, io, time, requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRUGS_PATH  = ROOT / 'data/processed/drugs.parquet'
CACHE_PATH  = ROOT / 'data/processed/smiles_cache.parquet'
PUBCHEM_URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES,CanonicalSMILES,MolecularFormula,MolecularWeight/JSON'

def fetch_smiles_pubchem(name: str) -> str | None:
    """Query PubChem REST API for the best SMILES of a drug name."""
    try:
        url = PUBCHEM_URL.format(requests.utils.quote(name))
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        props = data.get('PropertyTable', {}).get('Properties', [])
        if not props:
            return None
        # Prefer isomeric, fall back to canonical
        return props[0].get('IsomericSMILES') or props[0].get('CanonicalSMILES')
    except Exception:
        return None


def main():
    print('Loading data...')
    drugs  = pd.read_parquet(DRUGS_PATH)
    cache  = pd.read_parquet(CACHE_PATH)

    # Find drugs missing SMILES
    missing_mask = drugs['smiles'].fillna('') == ''
    missing = drugs[missing_mask][['drugbank_id', 'name']].copy()
    print(f'Drugs needing SMILES: {len(missing):,}')

    # Build cache dict for quick lookup
    cache_dict: dict[str, str] = {}
    for _, row in cache.iterrows():
        s = str(row.get('smiles', '') or '')
        if s:
            cache_dict[row['name'].strip().lower()] = s
        # Also by drugbank_id if available
        did = str(row.get('drugbank_id', '') or '')
        if did and s:
            cache_dict[did] = s

    fetched = 0
    failed  = []
    total   = len(missing)

    for i, (_, row) in enumerate(missing.iterrows(), 1):
        name = row['name']
        did  = row['drugbank_id']
        name_lower = name.strip().lower()

        # Already in cache?
        smiles = cache_dict.get(name_lower) or cache_dict.get(did, '')
        if smiles:
            # Patch drugs.parquet directly
            drugs.loc[drugs['drugbank_id'] == did, 'smiles'] = smiles
            fetched += 1
            continue

        print(f'  [{i}/{total}] Fetching: {name}...', end=' ', flush=True)
        smiles = fetch_smiles_pubchem(name)

        if not smiles:
            # Try with synonyms from drugbank_id (sometimes the DB name differs from PubChem)
            # Try common alternative names
            alt_names = [
                name.replace(' dipropionate', '').strip(),
                name.split(' ')[0],   # first word
            ]
            for alt in alt_names:
                if alt.lower() != name_lower:
                    smiles = fetch_smiles_pubchem(alt)
                    if smiles:
                        print(f'(via alt name: {alt})', end=' ')
                        break

        if smiles:
            print(f'OK  [{len(smiles)} chars]')
            fetched += 1
            cache_dict[name_lower] = smiles
            cache_dict[did] = smiles
            drugs.loc[drugs['drugbank_id'] == did, 'smiles'] = smiles
            # Update cache row
            cache.loc[cache['drugbank_id'] == did, 'smiles'] = smiles
        else:
            print('MISS')
            failed.append(name)

        # PubChem rate limit: max 5 req/sec
        if i % 5 == 0:
            time.sleep(1.0)
        else:
            time.sleep(0.2)

    print()
    print(f'Fetched: {fetched:,}/{total:,}')
    print(f'Failed:  {len(failed):,}  ({", ".join(failed[:15])}{"..." if len(failed) > 15 else ""})')

    # Save updated files
    print()
    print('Saving updated drugs.parquet...')
    drugs.to_parquet(DRUGS_PATH, index=False)

    print('Saving updated smiles_cache.parquet...')
    cache.to_parquet(CACHE_PATH, index=False)

    final_smiles = (drugs['smiles'].fillna('') != '').sum()
    print(f'Final SMILES coverage: {final_smiles:,}/{len(drugs):,}  ({final_smiles/len(drugs)*100:.1f}%)')
    print('Done.')


if __name__ == '__main__':
    main()
