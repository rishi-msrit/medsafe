"""
serving/schemas.py
==================
Pydantic v2 schemas for all FastAPI request and response models.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─── Request Schemas ──────────────────────────────────────────────────────────

class DrugListRequest(BaseModel):
    """Request for polypharmacy analysis."""
    drugs: list[str] = Field(
        ...,
        min_length=1,
        max_length=15,
        description="List of drug names (1–15 drugs)",
        examples=[["Warfarin", "Aspirin", "Metformin", "Lisinopril"]],
    )

    @field_validator("drugs")
    @classmethod
    def validate_drugs(cls, v: list[str]) -> list[str]:
        cleaned = [d.strip() for d in v if d.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty drug name is required")
        if len(cleaned) > 15:
            raise ValueError("Maximum 15 drugs per request")
        return cleaned


class PairRequest(BaseModel):
    """Request for pairwise interaction lookup."""
    drug_a: str = Field(..., description="First drug name", min_length=1)
    drug_b: str = Field(..., description="Second drug name", min_length=1)

    @field_validator("drug_a", "drug_b")
    @classmethod
    def validate_drug_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Drug name cannot be empty")
        return cleaned


class AlternativeRequest(BaseModel):
    """Request for safer alternative recommendations."""
    drug_to_replace: str = Field(..., description="Drug to find alternatives for")
    current_drugs: list[str] = Field(
        default=[],
        max_length=14,
        description="Other drugs the patient is currently taking",
    )


# ─── Response Schemas ─────────────────────────────────────────────────────────

class InteractionCard(BaseModel):
    """A single pairwise drug interaction detail."""
    drug_a: str
    drug_b: str
    severity: int                          # 0–3
    severity_label: str                    # minor/moderate/major/contraindicated
    interaction_prob: float                # 0–1
    confidence: float                      # 0–1
    mechanism_type: str
    plain_english: str
    clinical_implication: str
    cyp_enzymes: list[str]
    is_special_flag: bool
    support_count: int
    low_data_warning: bool
    faers_score: Optional[float] = None
    severity_probs: Optional[list[float]] = None  # [p0, p1, p2, p3] from MC Dropout
    severity_probs_std: Optional[list[float]] = None


class SpecialFlagModel(BaseModel):
    """A system-level special interaction flag."""
    flag_type: str
    severity: str
    drugs_involved: list[str]
    message: str
    color: str


class SafetyReportResponse(BaseModel):
    """Full polypharmacy safety report response."""
    drug_list: list[str]
    overall_risk_score: float
    risk_tier: str
    risk_tier_label: str
    risk_tier_color: str
    summary: str
    flagged_interactions: list[InteractionCard]
    special_flags: list[SpecialFlagModel]
    warfarin_warning: bool
    risk_culprit: Optional[str]
    risk_culprit_explanation: str
    shapley_values: dict[str, float]
    drug_interaction_counts: dict[str, int]
    num_flagged: int
    num_pairs_checked: int
    interaction_matrix: dict[str, dict[str, int]]  # drug_a → drug_b → severity


class PairwiseResponse(BaseModel):
    """Response for a pairwise drug interaction query."""
    drug_a: str
    drug_b: str
    interaction_detected: bool
    severity: int
    severity_label: str
    interaction_prob: float
    confidence: float
    confidence_level: str              # high/medium/low
    mechanism_type: str
    plain_english: str
    clinical_implication: str
    cyp_enzymes: list[str]
    is_special_flag: bool
    support_count: int
    low_data_warning: bool
    warning_message: str
    gnnexplainer: Optional[dict] = None       # GNNExplainer attribution
    severity_distribution: Optional[list[float]] = None  # [p0, p1, p2, p3]
    severity_distribution_std: Optional[list[float]] = None
    drug_a_smiles: Optional[str] = None       # For molecular viewer
    drug_b_smiles: Optional[str] = None


class AlternativeRecommendationModel(BaseModel):
    """A single safer drug alternative recommendation."""
    drug_name: str
    drug_id: str
    similarity_score: float
    risk_reduction_pct: float
    total_risk_with_patient: float
    atc_class_match: bool
    mechanism_explanation: str
    shared_cyp_enzymes: list[str]
    confidence: float


class AlternativeResponse(BaseModel):
    """Response for safer drug alternative recommendations."""
    drug_to_replace: str
    current_drugs: list[str]
    original_risk_score: float
    alternatives: list[AlternativeRecommendationModel]
    explanation: str


class DrugSearchResult(BaseModel):
    """A single drug search result."""
    name: str
    drugbank_id: Optional[str] = None
    atc_class: Optional[str] = None
    categories: Optional[str] = None
    match_score: float = 1.0


class DrugSearchResponse(BaseModel):
    """Response for drug name search/autocomplete."""
    query: str
    results: list[DrugSearchResult]
    total_found: int


# ─── Safe-Add Schemas ─────────────────────────────────────────────────────────

class SafeAddRequest(BaseModel):
    """Request for 'can I safely add this drug?' check."""
    candidate_drug: str = Field(..., description="Drug the patient wants to add", min_length=1)
    current_drugs: list[str] = Field(
        default=[],
        max_length=14,
        description="Drugs the patient is already taking",
    )

    @field_validator("candidate_drug")
    @classmethod
    def validate_candidate(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("candidate_drug cannot be empty")
        return cleaned


class SafeAddInteraction(BaseModel):
    """One pairwise interaction between the candidate and a current drug."""
    current_drug: str
    severity: int                  # 0=none 1=minor 2=moderate 3=major 4=contraindicated
    severity_label: str
    interaction_prob: float
    confidence: float
    mechanism_type: str
    plain_english: str
    cyp_enzymes: list[str]
    support_count: int
    low_data_warning: bool


class SafeAddResponse(BaseModel):
    """Response for the safe-add drug check."""
    candidate_drug: str
    current_drugs: list[str]
    verdict: str                   # safe | monitor | caution | avoid
    verdict_label: str             # human-readable label
    verdict_color: str             # hex color
    verdict_emoji: str
    max_severity: int
    num_interactions: int          # pairs with severity > 0
    num_pairs_checked: int
    interactions: list[SafeAddInteraction]
    summary: str
