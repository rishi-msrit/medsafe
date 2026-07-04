"""
serving/api.py
==============
FastAPI server for MedSafe — all endpoints.

Endpoints:
  POST /analyze/polypharmacy  — Full patient safety report (up to 15 drugs)
  POST /analyze/pairwise      — Single drug pair interaction lookup
  POST /recommend/alternative — Safer drug alternatives
  GET  /drugs/search          — Fuzzy drug name autocomplete

Launch:
  python serving/api.py
  uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs.loader import load_config  # noqa: E402
from serving.schemas import (  # noqa: E402
    AlternativeRequest,
    AlternativeResponse,
    AlternativeRecommendationModel,
    DrugListRequest,
    DrugSearchResponse,
    DrugSearchResult,
    InteractionCard,
    PairRequest,
    PairwiseResponse,
    SafeAddRequest,
    SafeAddResponse,
    SafeAddInteraction,
    SafetyReportResponse,
    SpecialFlagModel,
)


# ─── Global State ─────────────────────────────────────────────────────────────

class AppState:
    """Holds all loaded models and data — initialized at startup."""
    cfg = None
    drugs_df = None
    drug_names: list[str] = []
    drug_name_to_id: dict[str, str] = {}
    drug_id_to_smiles: dict[str, str] = {}
    interactions_lookup: dict = {}
    ddi_graph = None
    drug_to_idx: dict[str, int] = {}
    idx_to_drug: dict[int, str] = {}
    rgcn_model = None
    rgcn_drug_embeddings = None  # precomputed [num_drugs, hidden_dim] tensor
    gin_model = None
    recommender = None
    drug_embeddings = None
    model_loaded: bool = False


state = AppState()


# ─── Startup / Shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and data on startup."""
    logger.info("MedSafe API starting up...")
    _load_all_resources()
    logger.info(f"API ready. Model loaded: {state.model_loaded}")
    yield
    logger.info("MedSafe API shutting down.")


def _load_all_resources() -> None:
    """Load drugs database, models, and interaction graph."""
    try:
        import pandas as pd
        state.cfg = load_config()
        cfg = state.cfg

        # ── Load drugs DataFrame ──────────────────────────────────────────────
        drugs_path = ROOT / cfg.paths.data_processed / "drugs.parquet"
        if drugs_path.exists():
            state.drugs_df = pd.read_parquet(drugs_path)

            # Merge SMILES from cache (cache has 11,692 entries vs sparse drugs.parquet)
            smiles_cache_path = ROOT / cfg.paths.data_processed / "smiles_cache.parquet"
            if smiles_cache_path.exists():
                smiles_cache = pd.read_parquet(smiles_cache_path)
                smiles_map = dict(zip(smiles_cache["name"], smiles_cache["smiles"]))
                # Fill missing SMILES in drugs_df from the cache
                mask = state.drugs_df["smiles"].isna() | (state.drugs_df["smiles"] == "")
                state.drugs_df.loc[mask, "smiles"] = (
                    state.drugs_df.loc[mask, "name"].map(smiles_map)
                )
                n_smiles = (state.drugs_df["smiles"].fillna("") != "").sum()
                logger.info(f"SMILES available: {n_smiles:,}/{len(state.drugs_df):,} drugs (after cache merge)")

            state.drug_names = state.drugs_df["name"].tolist()

            # Build name→ID map — primary name + synonyms
            state.drug_name_to_id = {}
            for _, row in state.drugs_df.iterrows():
                did = row["drugbank_id"]
                name_lower = str(row["name"]).strip().lower()
                state.drug_name_to_id[name_lower] = did
                # Also store title-case version
                state.drug_name_to_id[str(row["name"]).strip()] = did
                # Synonyms (pipe-separated)
                syns = row.get("synonyms", "")
                if syns and str(syns) not in ("", "nan"):
                    for syn in str(syns).split("|"):
                        s = syn.strip().lower()
                        if s:
                            state.drug_name_to_id[s] = did

            # Common brand-name / INN aliases missing from DrugBank synonyms.
            # Without this, "Aspirin" fuzzy-matches to "Nitroaspirin" (score 83).
            _ALIASES = {
                "aspirin": "DB00945", "acetylsalicylic acid": "DB00945",
                "tylenol": "DB00316", "paracetamol": "DB00316", "acetaminophen": "DB00316",
                "advil": "DB01050", "motrin": "DB01050",
                "naproxen": "DB00788", "aleve": "DB09216",
                "lipitor": "DB01076", "zocor": "DB00641", "crestor": "DB01098",
                "plavix": "DB00758", "coumadin": "DB00682",
                "glucophage": "DB00331",
                "prozac": "DB00472", "zoloft": "DB01104",
                "xanax": "DB00404", "ativan": "DB00186",
                "prilosec": "DB00338", "nexium": "DB00736",
                "synthroid": "DB00451",
                "prinivil": "DB00722", "zestril": "DB00722",
                "norvasc": "DB00381", "lasix": "DB00695",
                "toprol": "DB00264",
                "diflucan": "DB00196",
                "ultram": "DB00193", "cordarone": "DB01118",
            }
            for alias, did in _ALIASES.items():
                state.drug_name_to_id[alias] = did
                state.drug_name_to_id[alias.title()] = did
                # Also add to drug_names so aliases appear in autocomplete search
                if alias.title() not in state.drug_names:
                    state.drug_names.append(alias.title())

            # SMILES lookup by DrugBank ID
            state.drug_id_to_smiles = {
                row["drugbank_id"]: row["smiles"]
                for _, row in state.drugs_df.iterrows()
                if row.get("smiles") and str(row.get("smiles", "")) not in ("", "nan")
            }
            logger.info(f"Drugs database: {len(state.drug_names):,} drugs")
        else:
            logger.warning(f"Drugs database not found: {drugs_path}")
            _load_fallback_drug_list()

        # ── Load interactions lookup (DrugBank backbone) ───────────────────────
        interactions_path = ROOT / cfg.paths.data_processed / "interactions_drugbank.parquet"
        if interactions_path.exists():
            idf = pd.read_parquet(interactions_path)

            # Severity from description keywords
            SEVERITY_KEYWORDS = {
                3: ["contraindicated", "do not use", "avoid concurrent", "should not be used"],
                2: ["major", "significantly", "clinically significant", "not recommended",
                    "serious", "severe", "markedly", "substantially"],
                1: ["moderate", "moderately", "may increase", "may decrease"],
                0: ["minor", "minimal", "slight", "negligible"],
            }
            # Mechanism types that always imply higher severity
            MECH_SEVERITY_BOOST = {
                "serotonin_syndrome": 3,
                "cardiac_qt": 2,
                "bleeding": 2,
                "cyp450_metabolic": 1,
            }

            # Vectorized severity inference — far faster than per-row iteration
            # Apply keyword masks to description column
            desc_lower = idf["description"].fillna("").str.lower()
            mech_col = idf["mechanism_type"].fillna("unknown").str.lower()

            # Start at default severity 1
            sev_col = pd.Series(1, index=idf.index)

            # Override with keyword matches (check in priority order: 0→1→2→3)
            sev_col[desc_lower.str.contains("minor|minimal|slight|negligible", na=False)] = 0
            sev_col[desc_lower.str.contains("moderate|moderately|may increase|may decrease", na=False)] = 1
            sev_col[desc_lower.str.contains(
                "major|significantly|clinically significant|not recommended|serious|severe|markedly|substantially",
                na=False, regex=True
            )] = 2
            sev_col[desc_lower.str.contains(
                "contraindicated|do not use|avoid concurrent|should not be used",
                na=False, regex=True
            )] = 3

            # Mechanism-based severity boost
            sev_col[mech_col == "serotonin_syndrome"] = sev_col[mech_col == "serotonin_syndrome"].clip(lower=3)
            sev_col[mech_col == "cardiac_qt"]         = sev_col[mech_col == "cardiac_qt"].clip(lower=2)
            sev_col[mech_col == "bleeding"]            = sev_col[mech_col == "bleeding"].clip(lower=2)
            sev_col[mech_col == "cyp450_metabolic"]    = sev_col[mech_col == "cyp450_metabolic"].clip(lower=1)

            conf_col = pd.Series(0.75, index=idf.index)
            conf_col[sev_col >= 2] = 0.88

            # Build lookup dict from vectorized columns
            id1_col = idf["drug1_id"].astype(str)
            id2_col = idf["drug2_id"].astype(str)

            for i in range(len(idf)):
                key = tuple(sorted([id1_col.iloc[i], id2_col.iloc[i]]))
                state.interactions_lookup[key] = {
                    "severity": int(sev_col.iloc[i]),
                    "mechanism_type": mech_col.iloc[i],
                    "description": desc_lower.iloc[i],
                    "confidence": float(conf_col.iloc[i]),
                    "support_count": 1,
                }
            logger.info(f"DrugBank interactions loaded: {len(state.interactions_lookup):,} pairs")

        # ── Curated high-risk pair overrides (clinical pharmacology ground truth) ─
        # These are well-established dangerous pairs that DrugBank CSV may miss
        # or under-represent. Keyed by DrugBank ID tuples.
        HIGH_RISK_PAIRS = [
            # Serotonin syndrome risks
            (("DB00472", "DB00193"), 3, "serotonin_syndrome", 0.95),  # Fluoxetine + Tramadol
            (("DB00472", "DB00601"), 3, "serotonin_syndrome", 0.95),  # Fluoxetine + Linezolid
            (("DB00193", "DB00601"), 3, "serotonin_syndrome", 0.92),  # Tramadol + Linezolid
            (("DB01104", "DB00601"), 3, "serotonin_syndrome", 0.95),  # Sertraline + Linezolid
            (("DB01175", "DB00601"), 3, "serotonin_syndrome", 0.95),  # Escitalopram + Linezolid
            (("DB01104", "DB00193"), 2, "serotonin_syndrome", 0.88),  # Sertraline + Tramadol
            (("DB01175", "DB00193"), 2, "serotonin_syndrome", 0.88),  # Escitalopram + Tramadol
            # QT prolongation
            (("DB01118", "DB00641"), 2, "cardiac_qt", 0.90),   # Amiodarone + Simvastatin (CYP3A4)
            (("DB01118", "DB00734"), 3, "cardiac_qt", 0.92),   # Amiodarone + Risperidone
            (("DB01118", "DB00502"), 3, "cardiac_qt", 0.93),   # Amiodarone + Haloperidol
            # Warfarin interactions
            (("DB00682", "DB01118"), 2, "cyp450_metabolic", 0.92),  # Warfarin + Amiodarone
            (("DB00682", "DB00196"), 2, "cyp450_metabolic", 0.93),  # Warfarin + Fluconazole
            (("DB00682", "DB00564"), 2, "cyp450_metabolic", 0.90),  # Warfarin + Carbamazepine
            (("DB00682", "DB01026"), 2, "cyp450_metabolic", 0.91),  # Warfarin + Ketoconazole
            (("DB00682", "DB00945"), 2, "bleeding", 0.90),          # Warfarin + Aspirin-like
            # Statin + CYP3A4 inhibitors
            (("DB00641", "DB01118"), 2, "cyp450_metabolic", 0.90),  # Simvastatin + Amiodarone
            (("DB00641", "DB01026"), 3, "cyp450_metabolic", 0.94),  # Simvastatin + Ketoconazole
            # MAO interactions
            (("DB00601", "DB00472"), 3, "serotonin_syndrome", 0.95),  # Linezolid + Fluoxetine
        ]
        added_curated = 0
        for pair_info in HIGH_RISK_PAIRS:
            (id1, id2), sev, mech, conf = pair_info
            key = tuple(sorted([id1, id2]))
            if key in state.interactions_lookup:
                # Only upgrade — never downgrade known entries
                ex = state.interactions_lookup[key]
                state.interactions_lookup[key].update({
                    "severity": max(ex["severity"], sev),
                    "confidence": max(ex["confidence"], conf),
                    "mechanism_type": mech if ex.get("mechanism_type") == "unknown" else ex["mechanism_type"],
                    "description": ex["description"] or f"Known {mech.replace('_', ' ')} risk",
                })
            else:
                state.interactions_lookup[key] = {
                    "severity": sev,
                    "mechanism_type": mech,
                    "description": f"Known {mech.replace('_', ' ')} risk (curated)",
                    "confidence": conf,
                    "support_count": 5,
                }
                added_curated += 1
        logger.info(f"Curated high-risk pairs: {added_curated} added, total: {len(state.interactions_lookup):,}")

        # ── Augment with TWOSIDES severity (overwrites with real severity data) ─
        twosides_path = ROOT / cfg.paths.data_processed / "twosides.parquet"
        if twosides_path.exists():
            tdf = pd.read_parquet(twosides_path)
            augmented = 0
            for _, row in tdf.iterrows():
                id1 = row.get("drug1_drugbank_id")
                id2 = row.get("drug2_drugbank_id")
                if not id1 or not id2 or str(id1) == "None" or str(id2) == "None":
                    continue
                key = tuple(sorted([str(id1), str(id2)]))
                # TWOSIDES severity proxy: 0=no signal, 1=mild, 2=moderate, 3=serious
                ts_sev = int(row.get("twosides_severity_proxy", 0))
                # num_side_effects → confidence (more SE signals = more confident)
                n_se = int(row.get("num_side_effects", 1))
                conf = min(0.98, 0.6 + 0.02 * min(n_se, 20))  # 0.6–0.98 range
                if key in state.interactions_lookup:
                    # Keep max severity between DrugBank and TWOSIDES
                    existing = state.interactions_lookup[key]
                    merged_sev = max(existing["severity"], ts_sev)
                    merged_conf = max(existing["confidence"], conf)
                    state.interactions_lookup[key].update({
                        "severity": merged_sev,
                        "confidence": merged_conf,
                        "support_count": existing["support_count"] + 1,
                    })
                elif ts_sev > 0:
                    # New pair only from TWOSIDES — add it
                    state.interactions_lookup[key] = {
                        "severity": ts_sev,
                        "mechanism_type": "pharmacodynamic",
                        "description": f"TWOSIDES: {n_se} co-reported side effects",
                        "confidence": conf,
                        "support_count": 1,
                    }
                    augmented += 1
            logger.info(
                f"TWOSIDES augmentation: {augmented:,} new pairs added, "
                f"total lookup: {len(state.interactions_lookup):,} pairs"
            )

        # ── Load DDI graph ─────────────────────────────────────────────────────
        graph_path = ROOT / cfg.paths.data_graphs / "ddi_hetero_graph.pt"
        if graph_path.exists():
            state.ddi_graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            state.drug_to_idx = getattr(state.ddi_graph, "drug_to_idx", {})
            state.idx_to_drug = {v: k for k, v in state.drug_to_idx.items()}
            logger.info(f"DDI graph loaded: {state.ddi_graph['drug'].num_nodes:,} drugs")

        # ── Load R-GCN model ───────────────────────────────────────────────────
        best_ckpt = ROOT / cfg.paths.checkpoints / "rgcn_finetune" / "rgcn_best.pt"
        if best_ckpt.exists() and state.ddi_graph is not None:
            from models.rgcn_predictor import build_rgcn_predictor

            drug_feature_dim = state.ddi_graph["drug"].x.shape[1]
            model = build_rgcn_predictor(cfg, drug_feature_dim)
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            state.rgcn_model = model
            state.model_loaded = True
            logger.info("R-GCN model loaded ✓")

            # Precompute drug embeddings once so pairwise/polypharmacy
            # endpoints can call prediction_head directly (fast) instead
            # of re-running the full graph forward pass on every request.
            try:
                from training.finetune_rgcn import build_combined_edge_index
                # torch is already imported at module level — do NOT re-import here
                # (re-importing inside a function makes it a local variable, causing
                # UnboundLocalError on every torch.* call that appears before this line)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device).eval()
                drug_x = state.ddi_graph["drug"].x.to(device)
                edge_index, edge_type = build_combined_edge_index(state.ddi_graph, device)
                with torch.no_grad():
                    state.rgcn_drug_embeddings = model.get_drug_embeddings(
                        drug_x, edge_index, edge_type
                    ).cpu()
                logger.info(
                    f"R-GCN drug embeddings precomputed: "
                    f"{state.rgcn_drug_embeddings.shape}"
                )
            except Exception as _e:
                logger.warning(f"Could not precompute R-GCN embeddings: {_e}")
        else:
            logger.warning(
                "R-GCN checkpoint not found. API will use knowledge-graph-only mode. "
                "Run: python train.py to train the model."
            )

        # ── Load drug embeddings + recommender ─────────────────────────────────
        emb_path = ROOT / cfg.paths.data_embeddings / "drug_embeddings.pt"
        emb_idx_path = ROOT / cfg.paths.data_embeddings / "embedding_drug_ids.parquet"
        if emb_path.exists() and emb_idx_path.exists() and state.drugs_df is not None:
            from models.alternative_recommender import DrugAlternativeRecommender

            state.drug_embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
            emb_ids_df = pd.read_parquet(emb_idx_path)
            emb_drug_ids = emb_ids_df["drug_id"].tolist()

            state.recommender = DrugAlternativeRecommender(
                embeddings=state.drug_embeddings,
                drug_ids=emb_drug_ids,
                drug_metadata=state.drugs_df,
                interactions_lookup=state.interactions_lookup,
                rgcn_predictor=state.rgcn_model,
                ddi_graph=state.ddi_graph,
                drug_to_idx=state.drug_to_idx,
                top_k_candidates=cfg.recommender.top_k_candidates,
                top_k_return=cfg.recommender.top_k_return,
            )
            logger.info("Drug alternative recommender loaded ✓")

    except Exception as e:
        logger.exception(f"Resource loading error: {e}")


def _load_fallback_drug_list() -> None:
    """Load a minimal known drug list when DrugBank is not available."""
    common_drugs = [
        ("Warfarin", "DB00682"), ("Metformin", "DB00331"), ("Lisinopril", "DB00722"),
        ("Atorvastatin", "DB01076"), ("Simvastatin", "DB00641"), ("Amlodipine", "DB00381"),
        ("Omeprazole", "DB00338"), ("Ibuprofen", "DB01050"), ("Amoxicillin", "DB01060"),
        ("Ciprofloxacin", "DB00537"), ("Metoprolol", "DB00264"), ("Furosemide", "DB00695"),
        ("Losartan", "DB00678"), ("Gabapentin", "DB00996"), ("Sertraline", "DB01104"),
        ("Escitalopram", "DB01175"), ("Alprazolam", "DB00404"), ("Lorazepam", "DB00186"),
        ("Tramadol", "DB00193"), ("Amiodarone", "DB01118"), ("Clopidogrel", "DB01234"),
        ("Fluoxetine", "DB00472"), ("Carbamazepine", "DB00564"), ("Fluconazole", "DB00196"),
        ("Linezolid", "DB00601"), ("Digoxin", "DB00390"), ("Cyclosporine", "DB00091"),
    ]
    state.drug_names = [n for n, _ in common_drugs]
    state.drug_name_to_id = {}
    for name, did in common_drugs:
        state.drug_name_to_id[name.lower()] = did
        state.drug_name_to_id[name] = did
    logger.info(f"Using fallback drug list: {len(common_drugs)} drugs")




# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedSafe API",
    description=(
        "GNN-powered polypharmacy drug interaction safety analysis. "
        "⚠️ For educational purposes only. Not a substitute for medical advice."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React frontend
cfg_temp = None
try:
    cfg_temp = load_config()
    cors_origins = cfg_temp.api.cors_origins
except Exception:
    cors_origins = ["http://localhost:5173", "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper: Resolve Drug Name → ID ──────────────────────────────────────────

def resolve_drug_name(name: str) -> tuple[str, str]:
    """
    Resolve a drug name (possibly misspelled) to (canonical_name, drugbank_id).

    Resolution order:
      1. Exact case-insensitive match in name→ID map (covers names + synonyms)
      2. Fuzzy match at configured threshold (default 70)
      3. Broad fuzzy match at threshold 50 (catches more misspellings)
      4. Return as-is (name will not match any lookup key — no interaction found)

    The returned drug_id is ALWAYS a real DrugBank ID when the drug is known.
    """
    threshold = state.cfg.api.fuzzy_match_threshold if state.cfg else 70
    name_stripped = name.strip()
    name_lower = name_stripped.lower()

    # ── 1. Exact match (covers primary names + synonyms) ─────────────────────
    if name_lower in state.drug_name_to_id:
        drug_id = state.drug_name_to_id[name_lower]
        # Get proper canonical name from drugs list
        canonical = next(
            (n for n in state.drug_names if n.lower() == name_lower),
            name_stripped,
        )
        return canonical, drug_id

    # ── 2. Fuzzy match at configured threshold ────────────────────────────────
    if state.drug_names:
        match_result = process.extractOne(
            name_stripped,
            state.drug_names,
            scorer=fuzz.WRatio,
            score_cutoff=threshold,
        )
        if match_result:
            matched_name, score, _ = match_result
            drug_id = state.drug_name_to_id.get(matched_name.lower(), "")
            if drug_id:
                logger.debug(f"Fuzzy matched '{name}' -> '{matched_name}' (score={score})")
                return matched_name, drug_id

        # ── 3. Broad fuzzy at lower threshold (catches misspellings) ──────────
        match_result_broad = process.extractOne(
            name_stripped,
            state.drug_names,
            scorer=fuzz.WRatio,
            score_cutoff=50,
        )
        if match_result_broad:
            matched_name, score, _ = match_result_broad
            drug_id = state.drug_name_to_id.get(matched_name.lower(), "")
            if drug_id:
                logger.info(f"Broad fuzzy matched '{name}' -> '{matched_name}' (score={score})")
                return matched_name, drug_id

    # ── 4. Unknown drug — return name as-is (will be a lookup miss) ──────────
    logger.warning(f"Drug not found in database: '{name}' — interactions may not be detected")
    return name_stripped, name_lower  # Use lowercase name as ID (consistent format)


def _build_interaction_matrix(
    drug_names: list[str],
    all_interactions: dict,
) -> dict[str, dict[str, int]]:
    """Build drug×drug severity matrix for heatmap rendering."""
    matrix: dict[str, dict[str, int]] = {d: {} for d in drug_names}
    for drug_a, drug_b in __import__("itertools").combinations(drug_names, 2):
        pair = all_interactions.get((drug_a, drug_b))
        sev = pair.severity if pair else 0
        matrix[drug_a][drug_b] = sev
        matrix[drug_b][drug_a] = sev
    return matrix


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "model_loaded": state.model_loaded,
        "drugs_count": len(state.drug_names),
        "interactions_count": len(state.interactions_lookup),
        "disclaimer": (
            "This API is for educational purposes only. "
            "Not a substitute for professional medical advice."
        ),
    }


@app.get("/drugs/search", response_model=DrugSearchResponse, tags=["Drugs"])
async def search_drugs(
    q: str = Query(..., description="Drug name search query", min_length=1),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results to return"),
):
    """
    Fuzzy drug name search for autocomplete.

    Returns ranked results matching the query string.
    """
    if not q.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty")

    if not state.drug_names:
        return DrugSearchResponse(query=q, results=[], total_found=0)

    matches = process.extract(
        q.strip(),
        state.drug_names,
        scorer=fuzz.WRatio,
        limit=limit * 2,  # Over-fetch then filter
    )

    results = []
    seen = set()
    for name, score, _ in matches:
        if score < 40 or name in seen:
            continue
        seen.add(name)
        drug_id = state.drug_name_to_id.get(name.lower(), "")

        atc = None
        categories = None
        if state.drugs_df is not None:
            row = state.drugs_df[state.drugs_df["name"] == name]
            if len(row) > 0:
                atc = row.iloc[0].get("atc_level1", None)
                categories = row.iloc[0].get("categories", None)
                if categories:
                    categories = str(categories).split("|")[0] if "|" in str(categories) else str(categories)

        results.append(DrugSearchResult(
            name=name,
            drugbank_id=drug_id or None,
            atc_class=atc,
            categories=categories,
            match_score=score / 100.0,
        ))

        if len(results) >= limit:
            break

    return DrugSearchResponse(query=q, results=results, total_found=len(results))


@app.post("/analyze/polypharmacy", response_model=SafetyReportResponse, tags=["Analysis"])
async def analyze_polypharmacy(request: DrugListRequest):
    """
    Full polypharmacy safety analysis for a patient's medication list.

    Returns a complete safety report including:
    - Overall risk score (0–100)
    - All pairwise interactions with severity and explanation
    - Special flags (QT, CNS, bleeding, Warfarin)
    - Risk culprit identification
    - Shapley value attribution

    ⚠️ For educational purposes only. Consult your pharmacist or doctor.
    """
    drugs = request.drugs

    # Resolve drug names
    resolved: list[tuple[str, str]] = []
    for name in drugs:
        canonical, drug_id = resolve_drug_name(name)
        resolved.append((canonical, drug_id))

    drug_names = [r[0] for r in resolved]
    drug_id_map = {r[0].lower(): r[1] for r in resolved}

    # Compute safety report
    from scoring.polypharmacy_score import compute_polypharmacy_score

    report = compute_polypharmacy_score(
        drug_names=drug_names,
        drug_id_map=drug_id_map,
        interactions_lookup=state.interactions_lookup,
        rgcn=state.rgcn_model,
        ddi_graph=state.ddi_graph,
        drug_to_idx=state.drug_to_idx,
        include_shapley=True,
        precomputed_embeddings=state.rgcn_drug_embeddings,
    )

    # Convert to response format
    flagged_cards = [
        InteractionCard(
            drug_a=p.drug_a,
            drug_b=p.drug_b,
            severity=p.severity,
            severity_label=p.severity_label,
            interaction_prob=p.interaction_prob,
            confidence=p.confidence,
            mechanism_type=p.mechanism_type,
            plain_english=p.plain_english,
            clinical_implication="Consult your pharmacist or doctor for personalized guidance.",
            cyp_enzymes=p.cyp_enzymes,
            is_special_flag=p.is_special_flag,
            support_count=p.support_count,
            low_data_warning=p.low_data_warning,
            faers_score=p.faers_score,
        )
        for p in report.flagged_interactions
    ]

    special_flag_models = [
        SpecialFlagModel(
            flag_type=f.flag_type,
            severity=f.severity,
            drugs_involved=f.drugs_involved,
            message=f.message,
            color=f.color,
        )
        for f in report.special_flags
    ]

    interaction_matrix = _build_interaction_matrix(drug_names, report.all_interactions)

    return SafetyReportResponse(
        drug_list=report.drug_list,
        overall_risk_score=report.overall_risk_score,
        risk_tier=report.risk_tier,
        risk_tier_label=report.risk_tier_label,
        risk_tier_color=report.risk_tier_color,
        summary=report.summary,
        flagged_interactions=flagged_cards,
        special_flags=special_flag_models,
        warfarin_warning=report.warfarin_warning,
        risk_culprit=report.risk_culprit,
        risk_culprit_explanation=report.risk_culprit_explanation,
        shapley_values=report.shapley_values,
        drug_interaction_counts=report.drug_interaction_counts,
        num_flagged=report.num_flagged,
        num_pairs_checked=report.num_pairs_checked,
        interaction_matrix=interaction_matrix,
    )


@app.post("/analyze/pairwise", response_model=PairwiseResponse, tags=["Analysis"])
async def analyze_pairwise(request: PairRequest):
    """
    Single drug pair interaction analysis with full explanation.

    Returns severity, mechanism, GNNExplainer attribution, and MC Dropout
    confidence intervals.

    ⚠️ For educational purposes only. Consult your pharmacist or doctor.
    """
    drug_a_name, drug_a_id = resolve_drug_name(request.drug_a)
    drug_b_name, drug_b_id = resolve_drug_name(request.drug_b)

    if drug_a_name.lower() == drug_b_name.lower():
        raise HTTPException(status_code=422, detail="Drug A and Drug B must be different")

    # Lookup interaction
    from scoring.polypharmacy_score import compute_pair_severity
    from explainability.mechanism_templates import generate_mechanism_explanation, detect_special_scenario
    from explainability.monte_carlo_dropout import mc_dropout_predict

    severity, confidence = compute_pair_severity(
        drug_a_id=drug_a_id,
        drug_b_id=drug_b_id,
        rgcn=state.rgcn_model,
        ddi_graph=state.ddi_graph,
        drug_to_idx=state.drug_to_idx,
        interactions_lookup=state.interactions_lookup,
        precomputed_embeddings=state.rgcn_drug_embeddings,
    )

    key = tuple(sorted([drug_a_id, drug_b_id]))
    record = state.interactions_lookup.get(key, {})
    mechanism_type = record.get("mechanism_type", "unknown")
    description = record.get("description", "")
    support_count = record.get("support_count", 1)

    # Check special scenarios
    special = detect_special_scenario(drug_a_name, drug_b_name)
    if special:
        mechanism_type = special

    explanation = generate_mechanism_explanation(
        drug_a=drug_a_name,
        drug_b=drug_b_name,
        mechanism_type=mechanism_type,
        severity=severity,
        drugbank_description=description,
        support_count=support_count,
    )

    # MC Dropout uncertainty — uses precomputed embeddings (fast)
    uncertainty = None
    severity_dist = None
    severity_dist_std = None
    warning_msg = ""
    gnn_explanation = None  # GNNExplainer disabled (too slow for real-time use)

    if state.rgcn_model is not None and state.rgcn_drug_embeddings is not None:
        idx_a = state.drug_to_idx.get(drug_a_id)
        idx_b = state.drug_to_idx.get(drug_b_id)
        if idx_a is not None and idx_b is not None and idx_a < len(state.rgcn_drug_embeddings) and idx_b < len(state.rgcn_drug_embeddings):
            try:
                import torch
                import numpy as _np
                device = next(state.rgcn_model.parameters()).device
                h_a = state.rgcn_drug_embeddings[idx_a].unsqueeze(0).to(device)
                h_b = state.rgcn_drug_embeddings[idx_b].unsqueeze(0).to(device)

                # Skip if embeddings contain NaN (model may have degenerate weights)
                if torch.isnan(h_a).any() or torch.isnan(h_b).any():
                    logger.debug("NaN in precomputed embeddings — skipping MC Dropout")
                else:
                    n_samples = max(5, min(10, state.cfg.scoring.mc_dropout_samples if state.cfg else 10))
                    sev_counts = [0, 0, 0, 0]
                    probs = []
                    state.rgcn_model.train()
                    try:
                        with torch.no_grad():
                            for _ in range(n_samples):
                                bl, sl, _, _ = state.rgcn_model.prediction_head(h_a, h_b)
                                p = float(torch.sigmoid(bl).item())
                                if not _np.isnan(p):
                                    probs.append(p)
                                    sev_counts[int(sl.argmax(-1).item())] += 1
                    finally:
                        state.rgcn_model.eval()  # always restore eval mode

                    if probs:
                        mean_prob = float(_np.mean(probs))
                        sigma = float(_np.std(probs))
                        n_valid = len(probs)
                        severity_dist = [c / n_valid for c in sev_counts]
                        severity_dist_std = [sigma] * 4
                        confidence = min(1.0, mean_prob * 1.5)
                        warning_msg = f"MC uncertainty \u03c3={sigma:.3f} ({n_valid} samples; lower = more certain)"
            except Exception as e:
                logger.debug(f"MC Dropout failed: {e}")
                try:
                    state.rgcn_model.eval()
                except Exception:
                    pass

    # Get SMILES for molecular viewer
    drug_a_smiles = state.drug_id_to_smiles.get(drug_a_id)
    drug_b_smiles = state.drug_id_to_smiles.get(drug_b_id)

    conf_level = "high" if confidence > 0.75 else "medium" if confidence > 0.5 else "low"
    if support_count < 5:
        conf_level = "low"
        warning_msg = warning_msg or f"Limited data ({support_count} records). Consult a pharmacist."

    return PairwiseResponse(
        drug_a=drug_a_name,
        drug_b=drug_b_name,
        interaction_detected=(severity > 0 or confidence > 0.5),
        severity=explanation.mechanism_type != "unknown" and severity or severity,
        severity_label=explanation.severity_label,
        interaction_prob=confidence,
        confidence=confidence,
        confidence_level=conf_level,
        mechanism_type=explanation.mechanism_type,
        plain_english=explanation.plain_english,
        clinical_implication=explanation.clinical_implication,
        cyp_enzymes=explanation.affected_cyp_enzymes,
        is_special_flag=explanation.is_special_flag,
        support_count=support_count,
        low_data_warning=(support_count < 5),
        warning_message=warning_msg,
        gnnexplainer=gnn_explanation,
        severity_distribution=severity_dist,
        severity_distribution_std=severity_dist_std,
        drug_a_smiles=drug_a_smiles,
        drug_b_smiles=drug_b_smiles,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /check/add-drug
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/check/add-drug", response_model=SafeAddResponse, tags=["Analysis"])
async def check_add_drug(request: SafeAddRequest):
    """
    Check whether a new drug can be safely added to an existing regimen.

    Returns a verdict (safe / monitor / caution / avoid) and a breakdown of
    every interaction between the candidate drug and each current medication.

    ⚠️ For educational purposes only.
    """
    from scoring.polypharmacy_score import compute_pair_severity
    from explainability.mechanism_templates import generate_mechanism_explanation

    cand_name, cand_id = resolve_drug_name(request.candidate_drug)

    current_resolved: list[tuple[str, str]] = []
    for d in request.current_drugs:
        n, did = resolve_drug_name(d)
        if n.lower() != cand_name.lower():
            current_resolved.append((n, did))

    # Severity label map
    _sev_labels = {0: "none", 1: "minor", 2: "moderate", 3: "major", 4: "contraindicated"}

    interactions: list[SafeAddInteraction] = []
    max_severity = 0

    for cur_name, cur_id in current_resolved:
        severity, confidence = compute_pair_severity(
            drug_a_id=cand_id,
            drug_b_id=cur_id,
            rgcn=state.rgcn_model,
            ddi_graph=state.ddi_graph,
            drug_to_idx=state.drug_to_idx,
            interactions_lookup=state.interactions_lookup,
            precomputed_embeddings=state.rgcn_drug_embeddings,
        )
        max_severity = max(max_severity, severity)

        # Lookup extra details from interaction record
        key = tuple(sorted([cand_id, cur_id]))
        record = state.interactions_lookup.get(key, {})
        support_count = int(record.get("support_count", 1))
        mechanism_type = str(record.get("mechanism_type", "unknown"))
        cyp_enzymes = record.get("cyp_enzymes", []) or []

        plain_english = generate_mechanism_explanation(
            cand_name, cur_name, mechanism_type, severity
        ) if severity > 0 else f"No known clinically significant interaction between {cand_name} and {cur_name}."

        interactions.append(SafeAddInteraction(
            current_drug=cur_name,
            severity=severity,
            severity_label=_sev_labels.get(severity, "unknown"),
            interaction_prob=min(1.0, confidence * 1.2),
            confidence=confidence,
            mechanism_type=mechanism_type,
            plain_english=plain_english,
            cyp_enzymes=list(cyp_enzymes),
            support_count=support_count,
            low_data_warning=(support_count < 3),
        ))

    # Sort: worst interactions first
    interactions.sort(key=lambda x: (-x.severity, -x.interaction_prob))

    num_flagged = sum(1 for i in interactions if i.severity > 0)

    # Verdict thresholds
    if max_severity == 0:
        verdict, label, color, emoji = "safe", "Safe to Add", "#22c55e", "✅"
    elif max_severity == 1:
        verdict, label, color, emoji = "monitor", "Generally Safe — Monitor", "#84cc16", "🟡"
    elif max_severity == 2:
        verdict, label, color, emoji = "caution", "Use With Caution", "#f59e0b", "⚠️"
    else:
        verdict, label, color, emoji = "avoid", "Avoid Adding", "#ef4444", "🚫"

    if num_flagged == 0:
        summary = f"{cand_name} has no known interactions with your current medications."
    else:
        worst = max(interactions, key=lambda x: x.severity)
        summary = (
            f"Adding {cand_name} flags {num_flagged} of {len(interactions)} drug pair(s). "
            f"Most serious: {cand_name} + {worst.current_drug} ({worst.severity_label})."
        )

    return SafeAddResponse(
        candidate_drug=cand_name,
        current_drugs=[n for n, _ in current_resolved],
        verdict=verdict,
        verdict_label=label,
        verdict_color=color,
        verdict_emoji=emoji,
        max_severity=max_severity,
        num_interactions=num_flagged,
        num_pairs_checked=len(interactions),
        interactions=interactions,
        summary=summary,
    )


@app.post("/recommend/alternative", response_model=AlternativeResponse, tags=["Recommendations"])
async def recommend_alternative(request: AlternativeRequest):
    """
    Find safer drug alternatives for a high-risk medication.

    Given a drug to replace and the patient's other medications,
    returns up to 3 structurally similar alternatives with lower
    interaction risk.

    ⚠️ For educational purposes only. Never change medications without consulting your doctor.
    """
    drug_name, drug_id = resolve_drug_name(request.drug_to_replace)

    current_names = []
    current_ids = []
    for name in request.current_drugs:
        n, i = resolve_drug_name(name)
        current_names.append(n)
        current_ids.append(i)

    if state.recommender is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Alternative recommender not loaded. "
                "Train the model first: python train.py"
            ),
        )

    from scoring.polypharmacy_score import compute_polypharmacy_score

    original_report = compute_polypharmacy_score(
        drug_names=[drug_name] + current_names,
        interactions_lookup=state.interactions_lookup,
        include_shapley=False,
        precomputed_embeddings=state.rgcn_drug_embeddings,
    )

    try:
        recommendations = state.recommender.recommend(
            drug_to_replace_id=drug_id,
            patient_drug_ids=current_ids,
            original_risk=original_report.overall_risk_score,
        )
    except Exception as e:
        logger.exception(f"Recommender failed: {e}")
        raise HTTPException(status_code=503, detail=f"Recommendation failed: {str(e)}")

    if not recommendations:
        return AlternativeResponse(
            drug_to_replace=drug_name,
            current_drugs=current_names,
            original_risk_score=original_report.overall_risk_score,
            alternatives=[],
            explanation=(
                f"No suitable alternatives found for {drug_name}. "
                "This may indicate the drug is highly unique in its molecular class "
                "or that insufficient embedding data is available. "
                "Consult your pharmacist for manual alternatives."
            ),
        )

    alt_models = [
        AlternativeRecommendationModel(
            drug_name=r.drug_name,
            drug_id=r.drug_id,
            similarity_score=r.similarity_score,
            risk_reduction_pct=r.risk_reduction_pct,
            total_risk_with_patient=r.total_risk_with_patient,
            atc_class_match=r.atc_class_match,
            mechanism_explanation=r.mechanism_explanation,
            shared_cyp_enzymes=r.shared_cyp_enzymes,
            confidence=r.confidence,
        )
        for r in recommendations
    ]

    best = recommendations[0] if recommendations else None
    explanation = (
        f"Found {len(recommendations)} safer alternatives to {drug_name}. "
        + (
            f"Best option: {best.drug_name} with {best.similarity_score:.0%} molecular similarity "
            f"and {best.risk_reduction_pct:.0f}% lower interaction risk with your current medications."
            if best else ""
        )
    )

    return AlternativeResponse(
        drug_to_replace=drug_name,
        current_drugs=current_names,
        original_risk_score=original_report.overall_risk_score,
        alternatives=alt_models,
        explanation=explanation,
    )


@app.get("/drugs/{drug_name}", tags=["Drugs"])
async def get_drug_info(drug_name: str):
    """Get detailed information about a specific drug."""
    canonical, drug_id = resolve_drug_name(drug_name)

    if state.drugs_df is not None:
        rows = state.drugs_df[state.drugs_df["name"] == canonical]
        if len(rows) > 0:
            row = rows.iloc[0].to_dict()
            # Convert non-serializable types
            result = {
                k: (str(v) if v is not None and not isinstance(v, (str, int, float, bool)) else v)
                for k, v in row.items()
            }
            return {"drug": result, "found": True}

    return {"drug": {"name": canonical, "drugbank_id": drug_id}, "found": False}


@app.get("/drugs/{drug_name}/neighbors", tags=["Drugs"])
async def get_drug_neighbors(drug_name: str, top_k: int = 5):
    """
    Find the top-k most structurally similar drugs using GNN embedding cosine similarity.

    Uses the pre-computed R-GCN drug embedding matrix loaded at startup.
    Returns an empty list if embeddings are not available (graceful degradation).
    """
    canonical, drug_id = resolve_drug_name(drug_name)

    # Graceful degradation — embeddings may not be loaded
    if (
        state.recommender is None
        or state.recommender._emb_normalized is None
        or drug_id not in state.recommender.emb_id_to_idx
    ):
        logger.warning(f"Neighbor lookup unavailable for '{canonical}' (embeddings not loaded or drug not in index)")
        return {"query": canonical, "neighbors": []}

    try:
        import torch

        query_idx = state.recommender.emb_id_to_idx[drug_id]
        query_emb = state.recommender._emb_normalized[query_idx]  # [dim]

        # Cosine similarities against all drugs — O(n) matmul, ~1ms
        sims = torch.matmul(state.recommender._emb_normalized, query_emb)  # [n]
        sims[query_idx] = -1.0  # Exclude self

        top_indices = sims.argsort(descending=True)[: top_k + 5].tolist()  # Over-fetch, filter below

        neighbors = []
        seen_ids = {drug_id}
        for idx in top_indices:
            if len(neighbors) >= top_k:
                break
            neighbor_id = state.recommender.drug_ids[idx]
            if neighbor_id in seen_ids:
                continue
            seen_ids.add(neighbor_id)

            similarity = float(sims[idx].item())
            if similarity < 0.0:
                continue

            # Fetch metadata
            meta = state.recommender._get_meta(neighbor_id)
            name_val = meta.get("name", neighbor_id)
            smiles_val = state.drug_id_to_smiles.get(neighbor_id) or meta.get("smiles")
            mw = meta.get("molecular_weight")
            cats = meta.get("categories")
            atc = meta.get("atc_level1")

            # Clean up serialization
            if smiles_val and str(smiles_val) in ("nan", "None", ""):
                smiles_val = None
            if mw is not None:
                try:
                    mw = float(mw)
                except (ValueError, TypeError):
                    mw = None
            if cats and str(cats) not in ("nan", "None"):
                cats = str(cats).split("|")[0].strip()
            else:
                cats = None
            if atc and str(atc) in ("nan", "None"):
                atc = None

            neighbors.append({
                "name": str(name_val),
                "drugbank_id": neighbor_id,
                "similarity": round(similarity, 4),
                "smiles": smiles_val,
                "molecular_weight": mw,
                "categories": cats,
                "atc_level1": str(atc) if atc else None,
            })

        return {"query": canonical, "neighbors": neighbors}

    except Exception as e:
        logger.exception(f"Neighbor lookup failed for '{canonical}': {e}")
        return {"query": canonical, "neighbors": []}




if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    uvicorn.run(
        "serving.api:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
        workers=1,
        loop="asyncio",  # Required for Ctrl+C to work on Windows
    )
