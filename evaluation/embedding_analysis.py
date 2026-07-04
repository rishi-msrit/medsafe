"""
evaluation/embedding_analysis.py
==================================
Analyse the GIN drug embedding space for quality metrics.

Tests:
  1. Pharmacological cluster coherence — ATC-class drugs should cluster together
     (Average intra-class cosine similarity > inter-class similarity)
  2. t-SNE / PCA projection saved as PNG for visual inspection
  3. Nearest-neighbour drug analogy test:
     "Simvastatin is to statins as Amoxicillin is to ... ?" (expect penicillin-class drug)

Usage:
  python evaluation/embedding_analysis.py
  python evaluation/embedding_analysis.py --plot  # Saves t-SNE to evaluation/results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_embeddings(cfg) -> tuple[torch.Tensor, list[str]]:
    """Load drug embeddings and drug ID list from disk."""
    emb_path = ROOT / cfg.paths.data_embeddings / "drug_embeddings.pt"
    ids_path  = ROOT / cfg.paths.data_embeddings / "embedding_drug_ids.parquet"

    if not emb_path.exists() or not ids_path.exists():
        logger.error(
            f"Embeddings not found at {emb_path}. "
            "Run: python train.py --skip-finetune (to generate GIN embeddings only)."
        )
        return None, None

    import pandas as pd
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    drug_ids = pd.read_parquet(ids_path)["drug_id"].tolist()
    logger.info(f"Loaded {len(drug_ids):,} drug embeddings of dim {embeddings.shape[1]}")
    return embeddings, drug_ids


def cluster_coherence_score(
    embeddings: torch.Tensor,
    drug_ids: list[str],
    drugs_df,
) -> dict:
    """
    Compute ATC class cluster coherence.

    Intra-class similarity = avg cosine sim within same ATC class.
    Inter-class similarity = avg cosine sim between different classes.
    Coherence score = intra / (intra + inter).
    Target: coherence > 0.55 (random baseline ~0.50).
    """
    if drugs_df is None or "atc_level1" not in drugs_df.columns:
        logger.warning("No ATC metadata available — skipping cluster coherence")
        return {}

    emb_norm = F.normalize(embeddings, dim=-1)
    id_to_idx = {did: i for i, did in enumerate(drug_ids)}

    # Group by ATC class
    from collections import defaultdict
    class_groups: dict[str, list[int]] = defaultdict(list)
    for _, row in drugs_df.iterrows():
        did = row.get("drugbank_id", "")
        atc = row.get("atc_level1", "")
        if did in id_to_idx and atc and len(atc) == 1:
            class_groups[atc].append(id_to_idx[did])

    # Filter to classes with ≥ 3 drugs
    class_groups = {k: v for k, v in class_groups.items() if len(v) >= 3}
    if not class_groups:
        logger.warning("Not enough ATC class data for coherence analysis")
        return {}

    intra_sims, inter_sims = [], []

    atc_classes = list(class_groups.keys())
    for atc in atc_classes:
        indices = class_groups[atc]
        embs = emb_norm[indices]
        # Intra: pairwise within class
        sim_matrix = (embs @ embs.T)
        mask = ~torch.eye(len(indices), dtype=torch.bool)
        intra_sims.extend(sim_matrix[mask].tolist())

    # Inter: sample cross-class pairs
    for i, atc_a in enumerate(atc_classes[:5]):
        for atc_b in atc_classes[i + 1:i + 3]:
            ea = emb_norm[class_groups[atc_a][:5]]
            eb = emb_norm[class_groups[atc_b][:5]]
            inter_sims.extend((ea @ eb.T).flatten().tolist())

    avg_intra = float(np.mean(intra_sims)) if intra_sims else 0.0
    avg_inter = float(np.mean(inter_sims)) if inter_sims else 0.0
    coherence = avg_intra / (avg_intra + avg_inter + 1e-8)

    logger.info(f"Cluster coherence: intra={avg_intra:.3f}, inter={avg_inter:.3f}, score={coherence:.3f}")
    return {"intra_sim": avg_intra, "inter_sim": avg_inter, "coherence": coherence}


def nearest_neighbour_analogy(
    embeddings: torch.Tensor,
    drug_ids: list[str],
    drugs_df,
    query_drug: str = "Simvastatin",
    k: int = 5,
) -> list[str]:
    """Find the K nearest drugs to query_drug in embedding space."""
    if drugs_df is None:
        return []

    id_to_idx = {}
    name_to_idx = {}
    for _, row in drugs_df.iterrows():
        did = row.get("drugbank_id", "")
        name = row.get("name", "")
        idx = drug_ids.index(did) if did in drug_ids else -1
        if idx >= 0:
            id_to_idx[did] = idx
            name_to_idx[name.lower()] = idx

    query_idx = name_to_idx.get(query_drug.lower(), -1)
    if query_idx < 0:
        logger.warning(f"Drug '{query_drug}' not found in embeddings")
        return []

    emb_norm = F.normalize(embeddings, dim=-1)
    query_emb = emb_norm[query_idx]
    sims = (emb_norm @ query_emb).numpy()
    sims[query_idx] = -1  # Exclude self

    top_k_idx = np.argsort(sims)[::-1][:k]

    # Map back to names
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    neighbours = [idx_to_name.get(i, drug_ids[i]) for i in top_k_idx]

    logger.info(f"Nearest neighbours to '{query_drug}': {neighbours}")
    return neighbours


def run_embedding_analysis(cfg, plot: bool = False) -> dict:
    """Full embedding analysis pipeline."""
    embeddings, drug_ids = load_embeddings(cfg)
    if embeddings is None:
        return {"error": "Embeddings not found"}

    drugs_df = None
    drugs_path = ROOT / cfg.paths.data_processed / "drugs.parquet"
    if drugs_path.exists():
        import pandas as pd
        drugs_df = pd.read_parquet(drugs_path)

    results = {}

    # Cluster coherence
    coherence = cluster_coherence_score(embeddings, drug_ids, drugs_df)
    results.update(coherence)

    # Nearest neighbours
    for drug in ["Simvastatin", "Amoxicillin", "Warfarin"]:
        nn = nearest_neighbour_analogy(embeddings, drug_ids, drugs_df, query_drug=drug)
        results[f"nn_{drug.lower()}"] = nn

    # t-SNE plot
    if plot:
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

            logger.info("Running t-SNE (this may take a minute)...")
            n = min(2000, len(drug_ids))
            sample_idx = np.random.choice(len(drug_ids), n, replace=False)
            emb_sample = embeddings[sample_idx].numpy()

            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            emb_2d = tsne.fit_transform(emb_sample)

            plt.figure(figsize=(12, 10))
            plt.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.4, s=8, c="#6366f1")
            plt.title("GIN Drug Embedding Space (t-SNE)", fontsize=14)
            plt.axis("off")

            out_path = ROOT / "evaluation" / "results" / "tsne_embeddings.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"t-SNE saved: {out_path}")
            results["tsne_path"] = str(out_path)
        except ImportError:
            logger.warning("matplotlib/sklearn not available — skipping t-SNE plot")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drug Embedding Analysis")
    parser.add_argument("--plot", action="store_true", help="Generate t-SNE plot")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    from configs.loader import load_config
    cfg = load_config(full_mode=args.full)
    run_embedding_analysis(cfg, plot=args.plot)
