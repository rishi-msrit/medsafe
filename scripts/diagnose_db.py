import sys; sys.path.insert(0,'.')
import pandas as pd, json

df = pd.read_parquet('data/processed/drugs.parquet')

# Check Aspirin synonyms
asa = df[df['drugbank_id'] == 'DB00945']
if len(asa):
    row = asa.iloc[0]
    print("Acetylsalicylic acid synonyms:", row['synonyms'])
    print("Groups:", row['groups'])
    print("SMILES:", str(row['smiles'])[:60])
print()

# Check actual SMILES values for key drugs
for did, name in [('DB00331','Metformin'),('DB00682','Warfarin'),('DB00722','Lisinopril'),('DB00641','Simvastatin')]:
    row = df[df['drugbank_id'] == did].iloc[0]
    smiles = row['smiles']
    print(f"{name} ({did}): smiles_type={type(smiles).__name__}, smiles_value={repr(str(smiles)[:50])}")

print()
# Simulate what api.py does to check JSON serialization
import math
for did, name in [('DB00331','Metformin'),('DB00682','Warfarin')]:
    row = df[df['drugbank_id'] == did].iloc[0]
    smiles_val = row['smiles']
    is_nan = isinstance(smiles_val, float) and math.isnan(smiles_val)
    print(f"{name}: is_nan={is_nan}, bool_check={bool(smiles_val) if not is_nan else 'N/A'}")

print()
# Check smiles_cache
import pathlib
cache_path = pathlib.Path('data/processed/smiles_cache.parquet')
if cache_path.exists():
    cache = pd.read_parquet(cache_path)
    print(f"smiles_cache columns: {list(cache.columns)}, rows: {len(cache)}")
    for name in ['Metformin','Warfarin','Aspirin','Acetylsalicylic acid']:
        hit = cache[cache['name'].str.lower() == name.lower()]
        if len(hit):
            print(f"  {name}: smiles={str(hit.iloc[0].get('smiles','N/A'))[:50]}")
        else:
            print(f"  {name}: NOT in cache")
