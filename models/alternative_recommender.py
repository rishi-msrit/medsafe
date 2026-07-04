"""
models/alternative_recommender.py
===================================
Drug alternative recommendation using molecular embedding similarity.

Constructor (as called by serving/api.py):
  DrugAlternativeRecommender(
      embeddings, drug_ids, drug_metadata, interactions_lookup,
      rgcn_predictor, ddi_graph, drug_to_idx,
      top_k_candidates, top_k_return,
  )

recommend() returns list of AlternativeResult objects (attribute access via r.drug_name etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch


# ── Result object (api.py accesses these as attributes, NOT dict keys) ─────────

@dataclass
class AlternativeResult:
    drug_name: str
    drug_id: str
    similarity_score: float          # 0.0–1.0 molecular cosine similarity
    risk_reduction_pct: float        # 0–100, how much lower than original_risk
    total_risk_with_patient: float   # estimated total risk if substituted
    atc_class_match: bool
    mechanism_explanation: str
    shared_cyp_enzymes: List[str] = field(default_factory=list)
    confidence: float = 0.5          # 0.0–1.0


# ── Curated drug class overrides ──────────────────────────────────────────────
# The ATC data in drugs.parquet is largely empty/unreliable for class matching.
# These hardcoded sets cover the most clinically relevant drug families and ensure
# "Simvastatin → Atorvastatin" (not "Simvastatin → Milrinone").
# DrugBank IDs are used as canonical identifiers.

DRUG_CLASS_OVERRIDES: dict[str, set[str]] = {
    # HMG-CoA reductase inhibitors (statins)
    "statins": {
        "DB00641",  # Simvastatin
        "DB01076",  # Atorvastatin
        "DB00175",  # Pravastatin
        "DB01098",  # Rosuvastatin
        "DB00227",  # Lovastatin
        "DB01095",  # Fluvastatin
        "DB08860",  # Pitavastatin
        "DB01124",  # Cerivastatin
    },
    # Azole antifungals
    "azole_antifungals": {
        "DB00196",  # Fluconazole
        "DB01167",  # Itraconazole
        "DB00582",  # Voriconazole
        "DB00956",  # Posaconazole
        "DB01026",  # Ketoconazole
        "DB00257",  # Clotrimazole
        "DB00061",  # Miconazole
    },
    # NSAIDs (non-selective COX inhibitors + selective COX-2)
    "nsaids": {
        "DB01050",  # Ibuprofen
        "DB00788",  # Naproxen
        "DB00586",  # Diclofenac
        "DB00482",  # Celecoxib
        "DB00328",  # Indomethacin
        "DB00554",  # Piroxicam
        "DB00814",  # Meloxicam
        "DB00224",  # Diflunisal
        "DB00991",  # Oxaprozin
    },
    # SSRIs / SNRIs
    "ssris": {
        "DB00472",  # Fluoxetine
        "DB01104",  # Sertraline
        "DB00715",  # Paroxetine
        "DB01175",  # Escitalopram
        "DB00215",  # Citalopram
        "DB01332",  # Fluvoxamine
        "DB00656",  # Trazodone
    },
    # ACE inhibitors
    "ace_inhibitors": {
        "DB00691",  # Moexipril
        "DB01342",  # Enalapril
        "DB00722",  # Lisinopril
        "DB00833",  # Captopril
        "DB00878",  # Ramipril
        "DB01345",  # Quinapril
        "DB13166",  # Zofenopril
    },
    # ARBs (Angiotensin receptor blockers)
    "arbs": {
        "DB00966",  # Telmisartan
        "DB00177",  # Valsartan
        "DB00678",  # Losartan
        "DB01345",  # Irbesartan
        "DB00795",  # Olmesartan
    },
    # Anticoagulants
    "anticoagulants": {
        "DB00682",  # Warfarin
        "DB09075",  # Apixaban
        "DB06292",  # Rivaroxaban
        "DB06695",  # Dabigatran
        "DB00001",  # Lepirudin
    },
    # Beta-blockers
    "beta_blockers": {
        "DB00335",  # Atenolol
        "DB00612",  # Bisoprolol
        "DB00264",  # Metoprolol
        "DB01297",  # Nebivolol
        "DB00571",  # Propranolol
        "DB00491",  # Carvedilol
    },
    # Proton pump inhibitors
    "ppis": {
        "DB00338",  # Omeprazole
        "DB00448",  # Lansoprazole
        "DB00939",  # Pantoprazole
        "DB00736",  # Esomeprazole
        "DB00432",  # Rabeprazole
    },
    # Biguanides (diabetes)
    "biguanides": {
        "DB00331",  # Metformin
        "DB00653",  # Phenformin
    },
    # Antidepressants - TCAs
    "tricyclic_antidepressants": {
        "DB00456",  # Clomipramine
        "DB00776",  # Amitriptyline
        "DB01151",  # Nortriptyline
        "DB01053",  # Desipramine
    },
    # Macrolide antibiotics
    "macrolides": {
        "DB00199",  # Erythromycin
        "DB01211",  # Clarithromycin
        "DB00207",  # Azithromycin
    },
}

# Build reverse map: drugbank_id -> class name
_ID_TO_CLASS: dict[str, str] = {}
for _cls, _ids in DRUG_CLASS_OVERRIDES.items():
    for _did in _ids:
        _ID_TO_CLASS[_did] = _cls


# ── Recommender ───────────────────────────────────────────────────────────────

class DrugAlternativeRecommender:
    """
    Recommends safer drug alternatives using:
    1. GIN molecular embedding cosine similarity
    2. DrugBank interaction severity with patient's regimen
    3. ATC class matching
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        drug_ids: list,
        drug_metadata: pd.DataFrame,
        interactions_lookup: dict,
        rgcn_predictor=None,
        ddi_graph=None,
        drug_to_idx: Optional[dict] = None,
        top_k_candidates: int = 20,
        top_k_return: int = 3,
    ):
        self.drug_ids = list(drug_ids)               # 11,584 IDs (from embedding_drug_ids.parquet)
        self.drug_metadata = drug_metadata.copy() if drug_metadata is not None else pd.DataFrame()
        self.interactions_lookup = interactions_lookup or {}
        self.rgcn_predictor = rgcn_predictor
        self.ddi_graph = ddi_graph
        self.drug_to_idx = drug_to_idx or {}
        self.top_k_candidates = top_k_candidates
        self.top_k_return = top_k_return

        # IMPORTANT: embeddings may have MORE rows than drug_ids (12,227 vs 11,584)
        # because inject_embeddings.py saved a full graph-ordered tensor.
        # Truncate to len(drug_ids) so indices stay in range.
        n = len(self.drug_ids)
        if embeddings is not None and len(embeddings) > 0:
            emb = embeddings[:n].cpu().float()
            norms = emb.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self._emb_normalized = emb / norms    # [n, 64]
        else:
            self._emb_normalized = None

        # drug_id → embedding row index
        self.emb_id_to_idx: dict[str, int] = {
            did: i for i, did in enumerate(self.drug_ids)
        }

        # Detect metadata column name: drugbank_id or drug_id
        id_col = None
        if not self.drug_metadata.empty:
            if "drugbank_id" in self.drug_metadata.columns:
                id_col = "drugbank_id"
            elif "drug_id" in self.drug_metadata.columns:
                id_col = "drug_id"

        if id_col:
            self.meta_lookup: dict[str, dict] = {
                str(row[id_col]): row.to_dict()
                for _, row in self.drug_metadata.iterrows()
            }
            # Only suggest approved drugs as alternatives
            self.approved_ids: set[str] = {
                str(row[id_col])
                for _, row in self.drug_metadata.iterrows()
                if "approved" in str(row.get("groups", "") or "").lower()
            }
        else:
            self.meta_lookup = {}
            self.approved_ids: set[str] = set()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_meta(self, drug_id: str) -> dict:
        return self.meta_lookup.get(drug_id, {})

    def _get_atc(self, drug_id: str) -> str:
        return str(self._get_meta(drug_id).get("atc_level1", "") or "")

    def _get_atc_prefix(self, drug_id: str, n: int = 4) -> str:
        """Get first n chars of ATC code (e.g. 'C10A' = statins at level 3)."""
        meta = self._get_meta(drug_id)
        atc_codes = str(meta.get("atc_codes", "") or "")
        if atc_codes and atc_codes != "nan":
            # Take first ATC code, return its prefix
            first = atc_codes.split("|")[0].strip()
            return first[:n] if len(first) >= n else first
        # Fall back to atc_level1
        return self._get_atc(drug_id)[:n]

    def _get_name(self, drug_id: str) -> str:
        return str(self._get_meta(drug_id).get("name", drug_id) or drug_id)

    def _get_cyp_enzymes(self, drug_id: str) -> list[str]:
        meta = self._get_meta(drug_id)
        enzymes = []
        for e in ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]:
            for role in ["substrate", "inhibitor", "inducer"]:
                if meta.get(f"{e}_{role}"):
                    enzymes.append(f"{e} {role}")
        return enzymes

    def _max_severity(self, cand_id: str, patient_ids: list[str]) -> int:
        max_sev = 0
        for pid in patient_ids:
            key = tuple(sorted([cand_id, pid]))
            rec = self.interactions_lookup.get(key, {})
            max_sev = max(max_sev, rec.get("severity", 0))
        return max_sev

    def _estimate_risk(self, cand_id: str, patient_ids: list[str]) -> float:
        """Rough total risk score (0–100) if this candidate replaced the original drug."""
        sev = self._max_severity(cand_id, patient_ids)
        sev_weights = [0, 15, 35, 65, 90]
        base = sev_weights[min(sev, 4)]
        # Penalise for number of CYP overlaps
        cand_cyp = set(self._get_cyp_enzymes(cand_id))
        cyp_pen = sum(
            len(cand_cyp & set(self._get_cyp_enzymes(pid))) * 3
            for pid in patient_ids
        )
        return min(100.0, base + cyp_pen)

    def _build_explanation(self, max_sev: int, sim: float, atc_match: bool,
                           shared_cyp: list[str]) -> str:
        sev_labels = [
            "no known interactions",
            "only minor interactions",
            "moderate interactions",
            "major interactions",
            "contraindicated interactions",
        ]
        sev_str = sev_labels[min(max_sev, 4)]
        sim_str = "high" if sim > 0.7 else "moderate" if sim > 0.4 else "lower"
        atc_str = "same therapeutic class" if atc_match else "different therapeutic class"
        cyp_str = f"; shared CYP: {', '.join(shared_cyp[:2])}" if shared_cyp else ""
        return f"{atc_str}; {sev_str} with regimen; {sim_str} molecular similarity{cyp_str}"

    # ── Main API ──────────────────────────────────────────────────────────────

    def recommend(
        self,
        drug_to_replace_id: str,
        patient_drug_ids: list[str],
        original_risk: float = 0.0,
    ) -> list[AlternativeResult]:
        """
        Two-pass recommendation strategy:
          Pass 1 — same ATC class (level 3 prefix e.g. 'C10A' = statins): these are genuine
                   clinical alternatives.
          Pass 2 — fill remaining slots with best approved drugs by embedding similarity.

        This prevents cross-class suggestions (e.g., antibiotics instead of statins).
        """
        if self._emb_normalized is None or drug_to_replace_id not in self.emb_id_to_idx:
            return self._fallback(drug_to_replace_id, patient_drug_ids, original_risk)

        src_idx = self.emb_id_to_idx[drug_to_replace_id]
        src_emb = self._emb_normalized[src_idx]        # [64]
        src_atc1 = self._get_atc(drug_to_replace_id)   # e.g. 'C'
        src_atc3 = self._get_atc_prefix(drug_to_replace_id, 4)  # e.g. 'C10A'
        src_atc2 = self._get_atc_prefix(drug_to_replace_id, 3)  # e.g. 'C10'
        exclude = set(patient_drug_ids) | {drug_to_replace_id}

        # Cosine similarities against all known drugs
        sims = (self._emb_normalized @ src_emb).numpy()  # [n]
        for exc in exclude:
            idx = self.emb_id_to_idx.get(exc)
            if idx is not None:
                sims[idx] = -2.0

        top_indices = np.argsort(sims)[::-1][:self.top_k_candidates * 8]

        def _make_result(cand_id: str, idx: int, atc_match: bool) -> AlternativeResult:
            sim = float(sims[idx])
            max_sev = self._max_severity(cand_id, patient_drug_ids)
            shared_cyp = self._get_cyp_enzymes(cand_id)
            total_risk = self._estimate_risk(cand_id, patient_drug_ids)
            risk_reduction = max(0.0, original_risk - total_risk)
            # ATC match contributes 45% to safety score (clinically relevant)
            sev_pen = max_sev / 4.0
            confidence = min(1.0, sim * 0.5 + (0.5 if atc_match else 0.1))
            return AlternativeResult(
                drug_name=self._get_name(cand_id),
                drug_id=cand_id,
                similarity_score=round(sim, 4),
                risk_reduction_pct=round(risk_reduction, 1),
                total_risk_with_patient=round(total_risk, 1),
                atc_class_match=atc_match,
                mechanism_explanation=self._build_explanation(max_sev, sim, atc_match, shared_cyp),
                shared_cyp_enzymes=shared_cyp[:4],
                confidence=round(confidence, 3),
            )

        same_class: list[AlternativeResult] = []
        other_class: list[AlternativeResult] = []
        seen: set[str] = set()

        src_curated_class = _ID_TO_CLASS.get(drug_to_replace_id)
        src_has_specific_atc = bool(src_atc3 and src_atc3 not in ("", "nan", "C")) or \
                               bool(src_atc2 and src_atc2 not in ("", "nan", "C"))

        # ── Pass 1: same-class candidates ────────────────────────────────────
        # For curated drugs: directly iterate over every class member in the
        # embedding index — do NOT rely on cosine similarity rank here, because
        # GIN embeddings capture interaction profiles, not therapeutic class.
        # Atorvastatin may sit far from Simvastatin in embedding space even though
        # they are both HMG-CoA reductase inhibitors.
        if src_curated_class is not None:
            class_members = DRUG_CLASS_OVERRIDES.get(src_curated_class, set())
            for cand_id in class_members:
                if cand_id in exclude:
                    continue
                if self.approved_ids and cand_id not in self.approved_ids:
                    continue
                cand_idx = self.emb_id_to_idx.get(cand_id)
                if cand_idx is None:
                    continue
                seen.add(cand_id)
                result = _make_result(cand_id, cand_idx, atc_match=True)
                same_class.append(result)

        # ── Pass 2: fill remaining slots from embedding-similarity ranking ────
        # Used for cross-class options (or when src not in curated dict).
        top_indices = np.argsort(sims)[::-1][:self.top_k_candidates * 4]
        for idx in top_indices:
            if len(same_class) >= self.top_k_return and len(other_class) >= self.top_k_return:
                break
            if idx >= len(self.drug_ids):
                continue
            cand_id = self.drug_ids[idx]
            if cand_id in exclude or cand_id in seen:
                continue
            if self.approved_ids and cand_id not in self.approved_ids:
                continue

            seen.add(cand_id)

            if src_curated_class is not None:
                # Already collected same-class in pass 1; any remainder is other_class
                atc_match = False
            else:
                cand_atc3 = self._get_atc_prefix(cand_id, 4)
                cand_atc2 = self._get_atc_prefix(cand_id, 3)
                cand_atc1 = self._get_atc(cand_id)
                atc_match = (
                    (src_atc3 and cand_atc3 and cand_atc3 == src_atc3
                     and src_atc3 not in ("", "nan")) or
                    (src_atc2 and cand_atc2 and cand_atc2 == src_atc2
                     and src_atc2 not in ("", "nan")) or
                    (not src_has_specific_atc and src_atc1 and cand_atc1
                     and cand_atc1 == src_atc1 and src_atc1 not in ("", "nan"))
                )

            result = _make_result(cand_id, idx, atc_match)
            if atc_match and len(same_class) < self.top_k_return:
                same_class.append(result)
            elif not atc_match and len(other_class) < self.top_k_return:
                other_class.append(result)

        combined = same_class + other_class
        combined.sort(key=lambda r: (
            not r.atc_class_match,
            r.total_risk_with_patient,
            -r.similarity_score,
        ))
        return combined[:self.top_k_return]

    def _fallback(
        self,
        drug_id: str,
        patient_ids: list[str],
        original_risk: float,
    ) -> list[AlternativeResult]:
        """ATC-class fallback when no embedding available."""
        src_atc = self._get_atc(drug_id)
        if not src_atc or self.drug_metadata.empty:
            return []

        id_col = "drugbank_id" if "drugbank_id" in self.drug_metadata.columns else "drug_id"
        exclude = set(patient_ids) | {drug_id}
        atc_col = "atc_level1"
        if atc_col not in self.drug_metadata.columns:
            return []

        same_class = self.drug_metadata[
            (self.drug_metadata[atc_col] == src_atc) &
            (~self.drug_metadata[id_col].isin(exclude))
        ]

        results = []
        for _, row in same_class.head(self.top_k_candidates).iterrows():
            cid = str(row[id_col])
            max_sev = self._max_severity(cid, patient_ids)
            total_risk = self._estimate_risk(cid, patient_ids)
            risk_reduction = max(0.0, original_risk - total_risk)
            results.append(AlternativeResult(
                drug_name=str(row.get("name", cid)),
                drug_id=cid,
                similarity_score=0.0,
                risk_reduction_pct=round(risk_reduction, 1),
                total_risk_with_patient=round(total_risk, 1),
                atc_class_match=True,
                mechanism_explanation=(
                    f"Same ATC class ({src_atc}); "
                    "no molecular similarity available (no SMILES)"
                ),
                shared_cyp_enzymes=[],
                confidence=0.3,
            ))

        results.sort(key=lambda r: (-r.risk_reduction_pct, r.total_risk_with_patient))
        return results[:self.top_k_return]
