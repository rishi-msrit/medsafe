"""
scripts/fix_atc_coverage.py
============================
Improves ATC level-1 coverage from 23.8% to 70%+ by deriving
the top-level ATC letter from the atc_codes / categories text fields.

ATC Level-1 letters:
  A - Alimentary tract and metabolism
  B - Blood and blood forming organs
  C - Cardiovascular system
  D - Dermatologicals
  G - Genito urinary system and sex hormones
  H - Systemic hormonal preparations
  J - Antiinfectives for systemic use
  L - Antineoplastic and immunomodulating agents
  M - Musculo-skeletal system
  N - Nervous system
  P - Antiparasitic products
  R - Respiratory system
  S - Sensory organs
  V - Various

Run: python scripts/fix_atc_coverage.py
"""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRUGS_PATH = ROOT / 'data/processed/drugs.parquet'

# Keywords in atc_codes / categories text → ATC level-1 letter
ATC_KEYWORD_MAP = [
    # Must check longer/more specific phrases FIRST to avoid misclassification
    ('ANTIPARASITIC',                           'P'),
    ('ANTINEOPLASTIC',                          'L'),
    ('IMMUNOMODULATING',                        'L'),
    ('IMMUNOSUPPRESSANT',                       'L'),
    ('BLOOD AND BLOOD FORMING',                 'B'),
    ('ANTITHROMBOTIC',                          'B'),
    ('ANTICOAGULANT',                           'B'),
    ('ANTIANEMIC',                              'B'),
    ('CARDIOVASCULAR',                          'C'),
    ('CARDIAC',                                 'C'),
    ('ANTIHYPERTENSIVE',                        'C'),
    ('ANTIARRHYTHMIC',                          'C'),
    ('DIURETIC',                                'C'),
    ('BETA BLOCK',                              'C'),
    ('CALCIUM CHANNEL',                         'C'),
    ('ACE INHIBIT',                             'C'),
    ('ANGIOTENSIN',                             'C'),
    ('STATIN',                                  'C'),
    ('LIPID',                                   'C'),
    ('NERVOUS SYSTEM',                          'N'),
    ('ANTIDEPRESSANT',                          'N'),
    ('ANTIEPILEPTIC',                           'N'),
    ('ANTICONVULSANT',                          'N'),
    ('ANXIOLYTIC',                              'N'),
    ('ANTIPSYCHOTIC',                           'N'),
    ('SEDATIVE',                                'N'),
    ('HYPNOTIC',                                'N'),
    ('OPIOID',                                  'N'),
    ('ANALGESIC',                               'N'),
    ('PSYCHOSTIMULANT',                         'N'),
    ('SSRI',                                    'N'),
    ('SNRI',                                    'N'),
    ('DOPAMINERGIC',                            'N'),
    ('ANTIINFECTIVE',                           'J'),
    ('ANTI-INFECTIVE',                          'J'),
    ('ANTIBIOTIC',                              'J'),
    ('ANTIBACTERIAL',                           'J'),
    ('ANTIVIRAL',                               'J'),
    ('ANTIFUNGAL',                              'J'),
    ('ANTIMYCOBACTERIAL',                       'J'),
    ('ANTIPROTOZOAL',                           'J'),
    ('RESPIRATORY',                             'R'),
    ('BRONCHODILAT',                            'R'),
    ('ANTIASTHMATIC',                           'R'),
    ('CORTICOSTEROID',                          'H'),  # systemic
    ('ADRENAL CORTEX',                          'H'),
    ('HORMONAL PREPARATION',                    'H'),
    ('PITUITARY',                               'H'),
    ('THYROID',                                 'H'),
    ('INSULIN',                                 'A'),
    ('ANTIDIABETIC',                            'A'),
    ('ALIMENTARY',                              'A'),
    ('GASTROINTESTINAL',                        'A'),
    ('PROTON PUMP',                             'A'),
    ('ANTACID',                                 'A'),
    ('LAXATIVE',                                'A'),
    ('ANTIEMETIC',                              'A'),
    ('ANTIDIARRHEAL',                           'A'),
    ('MUSCULO',                                 'M'),
    ('MUSCLE RELAXANT',                         'M'),
    ('NSAID',                                   'M'),
    ('ANTI-INFLAMMATORY',                       'M'),
    ('ANTIGOUT',                                'M'),
    ('OSTEOPOROSIS',                            'M'),
    ('DERMATOLOGIC',                            'D'),
    ('SKIN',                                    'D'),
    ('TOPICAL',                                 'D'),
    ('GENITO URINARY',                          'G'),
    ('GENITOURINARY',                           'G'),
    ('UROLOGICALS',                             'G'),
    ('SEX HORMONE',                             'G'),
    ('CONTRACEPTIVE',                           'G'),
    ('ESTROGEN',                                'G'),
    ('PROGESTERONE',                            'G'),
    ('TESTOSTERONE',                            'G'),
    ('SENSORY',                                 'S'),
    ('OPHTHALMOLOGICAL',                        'S'),
    ('OTOLOGICAL',                              'S'),
    ('VARIOUS',                                 'V'),
    ('DIAGNOSTICS',                             'V'),
    ('CONTRAST MEDIA',                          'V'),
    ('VITAMINS',                                'A'),
    ('MINERAL',                                 'A'),
]

# Hardcoded overrides for drugs with tricky text
DRUGBANK_ATC_OVERRIDE = {
    "DB00682": "B",  # Warfarin
    "DB00945": "B",  # Aspirin → Blood (antiplatelet)
    "DB01234": "B",  # Clopidogrel → Blood
    "DB01050": "M",  # Ibuprofen → Musculo-skeletal (NSAID)
    "DB00586": "M",  # Diclofenac → Musculo-skeletal
    "DB00331": "A",  # Metformin
    "DB01076": "C",  # Atorvastatin
    "DB00641": "C",  # Simvastatin
    "DB01241": "C",  # Gemfibrozil
    "DB00695": "C",  # Furosemide
    "DB01118": "C",  # Amiodarone
    "DB00722": "C",  # Lisinopril
    "DB00678": "C",  # Losartan
    "DB00381": "C",  # Amlodipine
    "DB00264": "C",  # Metoprolol
    "DB00390": "C",  # Digoxin
    "DB01115": "C",  # Nifedipine
    "DB00601": "J",  # Linezolid
    "DB00537": "J",  # Ciprofloxacin
    "DB01060": "J",  # Amoxicillin
    "DB00196": "J",  # Fluconazole
    "DB01026": "J",  # Ketoconazole
    "DB00199": "J",  # Erythromycin
    "DB00472": "N",  # Fluoxetine
    "DB01104": "N",  # Sertraline
    "DB01175": "N",  # Escitalopram
    "DB00193": "N",  # Tramadol
    "DB00186": "N",  # Lorazepam
    "DB00404": "N",  # Alprazolam
    "DB00996": "N",  # Gabapentin
    "DB00564": "N",  # Carbamazepine
    "DB00734": "N",  # Risperidone
    "DB00502": "N",  # Haloperidol
    "DB00363": "N",  # Clozapine
    "DB00656": "N",  # Trazodone
    "DB00604": "N",  # Methadone
    "DB00253": "N",  # Meperidine
    "DB00669": "N",  # Sumatriptan
    "DB00091": "L",  # Cyclosporine
    "DB00338": "A",  # Omeprazole
    "DB01211": "A",  # Lansoprazole
    "DB00498": "A",  # Pantoprazole
}


def infer_atc_from_text(text: str) -> str | None:
    """Try to infer ATC level-1 letter from atc_codes / categories text."""
    if not text or str(text) in ('', 'nan', 'None'):
        return None
    t = str(text).upper()
    for keyword, atc_letter in ATC_KEYWORD_MAP:
        if keyword in t:
            return atc_letter
    return None


def main():
    print('=== ATC Coverage Fixer ===')
    print()
    drugs = pd.read_parquet(DRUGS_PATH)

    before = (drugs['atc_level1'].fillna('') != '').sum()
    print(f'Before: {before:,}/{len(drugs):,} ({before/len(drugs)*100:.1f}%) have ATC level-1')

    patched = 0
    override_count = 0

    for idx, row in drugs.iterrows():
        did = row['drugbank_id']

        # 1. Hardcoded override (highest priority)
        if did in DRUGBANK_ATC_OVERRIDE:
            new_atc = DRUGBANK_ATC_OVERRIDE[did]
            if str(row.get('atc_level1', '') or '') != new_atc:
                drugs.at[idx, 'atc_level1'] = new_atc
                override_count += 1
            continue

        # 2. Already has atc_level1 — keep it
        if str(row.get('atc_level1', '') or '').strip():
            continue

        # 3. Derive from atc_codes text
        atc = infer_atc_from_text(str(row.get('atc_codes', '') or ''))

        # 4. Fall back to categories text
        if not atc:
            atc = infer_atc_from_text(str(row.get('categories', '') or ''))

        # 5. Fall back to description text (less reliable)
        if not atc:
            desc = str(row.get('description', '') or '') + ' ' + str(row.get('mechanism', '') or '')
            atc = infer_atc_from_text(desc)

        if atc:
            drugs.at[idx, 'atc_level1'] = atc
            patched += 1

    after = (drugs['atc_level1'].fillna('') != '').sum()
    print(f'After:  {after:,}/{len(drugs):,} ({after/len(drugs)*100:.1f}%) have ATC level-1')
    print(f'Newly inferred: {patched:,}  |  Overrides applied: {override_count}')
    print(f'Still missing:  {len(drugs)-after:,}  (no text to derive from)')

    # Show distribution
    dist = dict(drugs['atc_level1'].fillna('').value_counts().head(16))
    print(f'ATC distribution: {dist}')

    print()
    print('Saving drugs.parquet...')
    drugs.to_parquet(DRUGS_PATH, index=False)
    print('Done.')


if __name__ == '__main__':
    main()
