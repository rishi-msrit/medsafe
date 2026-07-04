"""
scripts/inject_embeddings.py
==============================
Injects fresh GIN drug embeddings into ddi_hetero_graph.pt.

Must be run AFTER:
  - GIN pretraining (produces data/embeddings/drug_embeddings.pt)
  - Graph building (produces data/graphs/ddi_hetero_graph.pt)

Run: python scripts/inject_embeddings.py

What it does:
  - Loads drug_embeddings.pt  [N_emb x 64]
  - Loads ddi_hetero_graph.pt [drug.x shape: N_drugs x 116]
  - Fills graph.drug.x[:, 0:64] with matched embeddings
  - Drugs without GIN embeddings get mean embedding (not zeros)
  - Re-saves graph in-place
  - Verifies zero count after injection
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

import torch
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = ROOT / 'data/graphs/ddi_hetero_graph.pt'
EMB_PATH   = ROOT / 'data/embeddings/drug_embeddings.pt'
EMB_IDS_PATH = ROOT / 'data/embeddings/embedding_drug_ids.parquet'


def main():
    print('=== Embedding Injection ===')
    print()

    # ── Load graph ────────────────────────────────────────────────────────────
    if not GRAPH_PATH.exists():
        print(f'ERROR: Graph not found at {GRAPH_PATH}')
        print('Run: python pipeline/build_ddi_graph.py first')
        return

    print(f'Loading graph: {GRAPH_PATH}...')
    g = torch.load(str(GRAPH_PATH), map_location='cpu', weights_only=False)
    drug_x = g['drug'].x  # [num_drugs, 116]
    num_drugs = drug_x.shape[0]
    feat_dim = drug_x.shape[1]
    print(f'  Drug nodes: {num_drugs:,}  Feature dim: {feat_dim}')

    zeros_before = (drug_x[:, :64].abs().sum(dim=1) == 0).sum().item()
    print(f'  Zero embeddings before injection: {zeros_before:,}/{num_drugs:,}')

    # ── Load embeddings ───────────────────────────────────────────────────────
    if not EMB_PATH.exists():
        print(f'ERROR: Embeddings not found at {EMB_PATH}')
        print('Run: python train.py --skip-finetune first')
        return

    print()
    print(f'Loading embeddings: {EMB_PATH}...')
    embeddings = torch.load(str(EMB_PATH), map_location='cpu', weights_only=False)
    emb_ids = pd.read_parquet(EMB_IDS_PATH)
    print(f'  Embeddings shape: {embeddings.shape}  ({len(emb_ids):,} drug IDs)')

    # Validate dimension
    emb_dim = embeddings.shape[1]
    if emb_dim != 64:
        print(f'WARNING: Expected 64-dim embeddings, got {emb_dim}. Adjust injection slice.')

    # ── Build ID → embedding row index map ───────────────────────────────────
    id_to_emb_idx = {row['drug_id']: i for i, row in emb_ids.iterrows()}

    # ── Build graph drug_to_idx map ───────────────────────────────────────────
    drug_to_idx = getattr(g, 'drug_to_idx', {})
    idx_to_drug = {v: k for k, v in drug_to_idx.items()}

    print()
    print(f'Injecting embeddings...')

    mean_emb = embeddings.mean(dim=0)  # fallback for unknown drugs

    injected = 0
    used_mean = 0

    new_x = drug_x.clone()

    for graph_idx in range(num_drugs):
        drug_id = idx_to_drug.get(graph_idx)
        if drug_id and drug_id in id_to_emb_idx:
            emb_idx = id_to_emb_idx[drug_id]
            new_x[graph_idx, :emb_dim] = embeddings[emb_idx]
            injected += 1
        else:
            # Fallback: use mean embedding (not zeros)
            new_x[graph_idx, :emb_dim] = mean_emb
            used_mean += 1

    g['drug'].x = new_x

    zeros_after = (new_x[:, :emb_dim].abs().sum(dim=1) == 0).sum().item()

    print(f'  Injected from GIN: {injected:,}/{num_drugs:,}')
    print(f'  Used mean embedding: {used_mean:,}')
    print(f'  Zero embeddings after injection: {zeros_after:,} (should be 0)')

    # ── Save updated graph ─────────────────────────────────────────────────────
    print()
    print(f'Saving graph to {GRAPH_PATH}...')
    torch.save(g, str(GRAPH_PATH))

    # ── Also save a full graph-ordered embedding tensor ────────────────────────
    # finetune_rgcn.py checks: gin_embs.shape[0] == num_drugs (12,227)
    # Original drug_embeddings.pt has only 11,584 rows → triggers WARNING.
    # Fix: save a 12,227-row tensor ordered by graph node index.
    full_emb = torch.zeros(num_drugs, emb_dim)
    for graph_idx in range(num_drugs):
        drug_id = idx_to_drug.get(graph_idx)
        if drug_id and drug_id in id_to_emb_idx:
            full_emb[graph_idx] = embeddings[id_to_emb_idx[drug_id]]
        else:
            full_emb[graph_idx] = mean_emb
    torch.save(full_emb, str(EMB_PATH))
    print(f'Saved full {num_drugs:,}-row embedding tensor → {EMB_PATH.name}')

    print()
    print('=== Injection complete ===')
    print(f'  GIN embeddings: {injected:,} drugs  \u2713')
    print(f'  Mean fallback:  {used_mean:,} drugs  (no SMILES available)')
    print(f'  Zero-vectors:   {zeros_after:,} ({"GOOD" if zeros_after == 0 else "PROBLEM"})')


if __name__ == '__main__':
    main()
