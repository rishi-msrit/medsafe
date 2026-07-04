
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from explainability.mechanism_templates import (  # noqa: E402
    detect_special_scenario,
    generate_mechanism_explanation,
)

# ─── Constants ────────────────────────────────────────────────────────────────

SEVERITY_WEIGHTS = {
    3: 2.0,  # contraindicated
    2: 1.5,  # major
    1: 1.0,  # moderate
    0: 0.3,  # minor
}

RISK_TIERS = [
    (0, 30, "safe", "Generally safe", "#22c55e"),
    (31, 60, "review", "Review recommended", "#f59e0b"),
    (61, 80, "high", "High risk", "#f97316"),
    (81, 100, "critical", "Critical — consult doctor immediately", "#ef4444"),
]

QT_DRUGS = {
    "amiodarone", "sotalol", "dofetilide", "ibutilide", "quinidine",
    "procainamide", "disopyramide", "haloperidol", "thioridazine",
    "chlorpromazine", "quetiapine", "ziprasidone", "droperidol",
    "methadone", "clarithromycin", "erythromycin", "azithromycin",
    "moxifloxacin", "levofloxacin", "ondansetron", "domperidone",
    "hydroxychloroquine", "chloroquine", "pentamidine",
}

CNS_DEPRESSANT_CATEGORIES = {
    "benzodiazepine", "opioid", "barbiturate", "sedative", "hypnotic",
    "antihistamine", "muscle relaxant", "general anesthetic",
}

CNS_DRUG_NAMES = {
    "morphine", "oxycodone", "hydrocodone", "fentanyl", "tramadol",
    "codeine", "diazepam", "lorazepam", "alprazolam", "clonazepam",
    "zolpidem", "eszopiclone", "diphenhydramine", "hydroxyzine",
}

NSAID_DRUGS = {
    "ibuprofen", "naproxen", "aspirin", "diclofenac", "celecoxib",
    "indomethacin", "meloxicam", "ketoprofen", "piroxicam", "ketorolac",
}

ANTICOAGULANT_DRUGS = {
    "warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran",
    "edoxaban", "enoxaparin", "fondaparinux", "clopidogrel",
    "ticagrelor", "prasugrel",
}

WARFARIN_NAMES = {"warfarin", "coumadin"}


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class PairInteraction:
    """A pairwise drug interaction with severity and confidence."""
    drug_a: str
    drug_b: str
    severity: int                    # 0–3
    severity_label: str              # minor/moderate/major/contraindicated
    confidence: float                # 0–1
    interaction_prob: float          # Raw model probability
    mechanism_type: str
    plain_english: str
    cyp_enzymes: list[str] = field(default_factory=list)
    is_special_flag: bool = False
    support_count: int = 1
    faers_score: float = 0.0
    low_data_warning: bool = False


@dataclass
class SpecialFlag:
    """A system-level special interaction flag (not pairwise)."""
    flag_type: str          # "qt_prolongation", "cns_depression", "nsaid_anticoag", "warfarin"
    severity: str           # "critical", "high", "moderate"
    drugs_involved: list[str]
    message: str
    color: str              # UI color code


@dataclass
class SafetyReport:
    """Full safety report for a patient's medication list."""
    drug_list: list[str]
    overall_risk_score: float        # 0–100
    risk_tier: str                   # safe/review/high/critical
    risk_tier_label: str             # Human-readable tier name
    risk_tier_color: str             # Hex color
    summary: str                     # Plain English summary
    flagged_interactions: list[PairInteraction]
    all_interactions: dict[tuple[str, str], PairInteraction]  # All pairs (incl green)
    special_flags: list[SpecialFlag]
    warfarin_warning: bool
    risk_culprit: Optional[str]      # Drug with highest interaction burden
    risk_culprit_explanation: str
    shapley_values: dict[str, float] # Drug → Shapley value
    drug_interaction_counts: dict[str, int]  # Drug → number of flagged interactions
    num_flagged: int
    num_pairs_checked: int


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _is_qt_drug(name: str) -> bool:
    return name.lower().strip() in QT_DRUGS


def _is_cns_drug(name: str, categories: str = "") -> bool:
    n = name.lower().strip()
    return n in CNS_DRUG_NAMES or any(c in categories.lower() for c in CNS_DEPRESSANT_CATEGORIES)


def _is_nsaid(name: str) -> bool:
    return name.lower().strip() in NSAID_DRUGS


def _is_anticoagulant(name: str) -> bool:
    return name.lower().strip() in ANTICOAGULANT_DRUGS


def _is_warfarin(name: str) -> bool:
    return name.lower().strip() in WARFARIN_NAMES


def get_risk_tier(score: float) -> tuple[str, str, str]:
    """Return (tier_key, tier_label, color) for a risk score."""
    for lo, hi, key, label, color in RISK_TIERS:
        if lo <= score <= hi:
            return key, label, color
    return "critical", "Critical", "#ef4444"


# ─── Core Pair Severity Lookup ────────────────────────────────────────────────

def compute_pair_severity(
    drug_a_id: str,
    drug_b_id: str,
    rgcn: Optional[object] = None,
    ddi_graph: Optional[object] = None,
    drug_to_idx: Optional[dict] = None,
    interactions_lookup: Optional[dict] = None,
    precomputed_embeddings=None,  # precomputed [N_drug, hidden] tensor (fast path)
) -> tuple[int, float]:
    """
    compute_pair_severity priority:
    1. KG lookup (fastest), 2. precomputed embeddings, 3. default (0, 0.0)
    NOTE: no full graph forward pass here — too slow for real-time use
    """
    # ── Step 1: Knowledge graph lookup ────────────────────────────────────────
    if interactions_lookup is not None:
        candidates = [
            tuple(sorted([drug_a_id, drug_b_id])),
        ]
        if drug_a_id.upper() != drug_a_id or drug_b_id.upper() != drug_b_id:
            candidates.append(tuple(sorted([drug_a_id.upper(), drug_b_id.upper()])))

        for key in candidates:
            if key in interactions_lookup:
                record = interactions_lookup[key]
                sev = int(record.get("severity", 1))
                conf = float(record.get("confidence", 0.85))
                return sev, conf

    # ── Step 2: Fast prediction from precomputed embeddings ───────────────────
    # Uses prediction_head only — no graph forward pass needed.
    if rgcn is not None and precomputed_embeddings is not None and drug_to_idx is not None:
        try:
            import torch
            idx_a = drug_to_idx.get(drug_a_id)
            idx_b = drug_to_idx.get(drug_b_id)
            if idx_a is not None and idx_b is not None and \
                    idx_a < len(precomputed_embeddings) and idx_b < len(precomputed_embeddings):
                device = next(rgcn.parameters()).device
                h_a = precomputed_embeddings[idx_a].unsqueeze(0).to(device)
                h_b = precomputed_embeddings[idx_b].unsqueeze(0).to(device)
                with torch.no_grad():
                    bl, sl, _, _ = rgcn.prediction_head(h_a, h_b)
                    prob = float(torch.sigmoid(bl).item())
                    sev = int(sl.argmax(-1).item())
                return sev, min(1.0, prob * 2)
        except Exception as e:
            logger.debug(f"Embedding-based prediction failed for ({drug_a_id}, {drug_b_id}): {e}")

    return 0, 0.0


# ─── Special Flags Detection ──────────────────────────────────────────────────

def detect_special_flags(drug_names: list[str], drug_metadata: Optional[dict] = None) -> list[SpecialFlag]:
    """Detect special system-level flags: QT, CNS, NSAID+anticoag, Warfarin."""
    flags = []
    names_lower = [n.lower().strip() for n in drug_names]

    # QT Prolongation
    qt_drugs_in_list = [n for n in drug_names if _is_qt_drug(n)]
    if len(qt_drugs_in_list) >= 2:
        flags.append(SpecialFlag(
            flag_type="qt_prolongation",
            severity="critical",
            drugs_involved=qt_drugs_in_list,
            message=(
                f"⚡ QT Prolongation Risk: {', '.join(qt_drugs_in_list[:3])} "
                f"{'and others' if len(qt_drugs_in_list) > 3 else ''} can all prolong the QT interval. "
                f"Combined risk of fatal cardiac arrhythmia (Torsades de Pointes) is significantly elevated. "
                f"Urgent cardiology/pharmacy review required."
            ),
            color="#ef4444",
        ))

    # CNS Depression
    cns_drugs = [n for n in drug_names if _is_cns_drug(n)]
    if len(cns_drugs) >= 2:
        flags.append(SpecialFlag(
            flag_type="cns_depression",
            severity="high",
            drugs_involved=cns_drugs,
            message=(
                f"😴 CNS Depression Risk: {', '.join(cns_drugs[:3])} all depress the central nervous system. "
                f"Combined effect can cause excessive sedation, respiratory depression, "
                f"and in severe cases, coma. Avoid driving or operating heavy machinery."
            ),
            color="#f97316",
        ))

    # NSAID + Anticoagulant
    nsaids = [n for n in drug_names if _is_nsaid(n)]
    anticoags = [n for n in drug_names if _is_anticoagulant(n)]
    if nsaids and anticoags:
        flags.append(SpecialFlag(
            flag_type="nsaid_anticoagulant",
            severity="high",
            drugs_involved=nsaids + anticoags,
            message=(
                f"🩸 Bleeding Risk: {', '.join(nsaids)} (NSAID) + {', '.join(anticoags)} (anticoagulant) "
                f"is one of the most dangerous combinations in medicine. "
                f"NSAIDs impair platelet function while anticoagulants reduce clotting. "
                f"Serious or fatal bleeding events are significantly more likely. "
                f"Discuss safer alternatives with your doctor."
            ),
            color="#f97316",
        ))

    # Warfarin
    warfarin_drugs = [n for n in drug_names if _is_warfarin(n)]
    if warfarin_drugs:
        flags.append(SpecialFlag(
            flag_type="warfarin_alert",
            severity="critical",
            drugs_involved=warfarin_drugs,
            message=(
                f"⚠️  Warfarin Alert: Warfarin has the narrowest therapeutic window of any commonly "
                f"prescribed drug and interacts with 200+ medications, foods, and supplements. "
                f"ANY change to this medication list requires immediate pharmacist or physician review "
                f"and likely INR (blood clotting) monitoring."
            ),
            color="#ef4444",
        ))

    return flags


# ─── Main Scoring Function ────────────────────────────────────────────────────

def compute_polypharmacy_score(
    drug_names: list[str],
    drug_id_map: Optional[dict[str, str]] = None,
    interactions_lookup: Optional[dict] = None,
    rgcn: Optional[object] = None,
    ddi_graph: Optional[object] = None,
    drug_to_idx: Optional[dict] = None,
    drug_metadata: Optional[dict] = None,
    include_shapley: bool = True,
    mc_samples: int = 0,
    precomputed_embeddings=None,  # precomputed R-GCN embeddings for fast pair scoring
) -> SafetyReport:
    """Compute full polypharmacy safety report for a patient drug list."""
    import itertools

    n = len(drug_names)

    # Edge case
    if n == 0:
        return SafetyReport(
            drug_list=[], overall_risk_score=0.0,
            risk_tier="safe", risk_tier_label="No drugs",
            risk_tier_color="#22c55e", summary="No drugs provided.",
            flagged_interactions=[], all_interactions={},
            special_flags=[], warfarin_warning=False,
            risk_culprit=None, risk_culprit_explanation="",
            shapley_values={}, drug_interaction_counts={},
            num_flagged=0, num_pairs_checked=0,
        )

    if n == 1:
        return SafetyReport(
            drug_list=drug_names, overall_risk_score=0.0,
            risk_tier="safe", risk_tier_label="Generally safe",
            risk_tier_color="#22c55e",
            summary=f"Only one drug ({drug_names[0]}) — no interactions possible.",
            flagged_interactions=[], all_interactions={},
            special_flags=detect_special_flags(drug_names),
            warfarin_warning=any(_is_warfarin(d) for d in drug_names),
            risk_culprit=None, risk_culprit_explanation="",
            shapley_values={drug_names[0]: 0.0}, drug_interaction_counts={drug_names[0]: 0},
            num_flagged=0, num_pairs_checked=0,
        )

    # Map names to DrugBank IDs
    drug_ids = {}
    if drug_id_map:
        for name in drug_names:
            did = drug_id_map.get(name.lower(), drug_id_map.get(name, None))
            if did:
                drug_ids[name] = did

    # Compute all pairwise interactions
    all_interactions: dict[tuple[str, str], PairInteraction] = {}
    flagged: list[PairInteraction] = []
    drug_interaction_counts = {d: 0 for d in drug_names}

    total_raw_risk = 0.0
    max_possible_risk = 0.0

    for drug_a, drug_b in itertools.combinations(drug_names, 2):
        drug_a_id = drug_ids.get(drug_a, drug_a.lower())
        drug_b_id = drug_ids.get(drug_b, drug_b.lower())

        # Get severity and confidence
        severity, confidence = compute_pair_severity(
            drug_a_id=drug_a_id,
            drug_b_id=drug_b_id,
            rgcn=rgcn,
            ddi_graph=ddi_graph,
            drug_to_idx=drug_to_idx,
            interactions_lookup=interactions_lookup,
            precomputed_embeddings=precomputed_embeddings,
        )

        interaction_prob = confidence  # Proxy: confidence ≈ interaction probability

        # Support count (for low-data warning)
        support_count = 5  # Default: assume sufficient data
        if interactions_lookup:
            key = tuple(sorted([drug_a_id, drug_b_id]))
            if key in interactions_lookup:
                support_count = interactions_lookup[key].get("support_count", 5)

        # Mechanism and explanation
        # Check special scenarios first
        scenario = detect_special_scenario(drug_a, drug_b)
        mechanism_type = scenario or (interactions_lookup or {}).get(
            tuple(sorted([drug_a_id, drug_b_id])), {}
        ).get("mechanism_type", "unknown")

        explanation = generate_mechanism_explanation(
            drug_a=drug_a,
            drug_b=drug_b,
            mechanism_type=mechanism_type,
            severity=severity,
            support_count=support_count,
        )

        severity_label_map = {0: "minor", 1: "moderate", 2: "major", 3: "contraindicated"}

        pair = PairInteraction(
            drug_a=drug_a,
            drug_b=drug_b,
            severity=severity,
            severity_label=severity_label_map[severity],
            confidence=confidence,
            interaction_prob=interaction_prob,
            mechanism_type=explanation.mechanism_type,
            plain_english=explanation.plain_english,
            cyp_enzymes=explanation.affected_cyp_enzymes,
            is_special_flag=explanation.is_special_flag,
            support_count=support_count,
            low_data_warning=(support_count < 5),
        )

        all_interactions[(drug_a, drug_b)] = pair

        # Accumulate risk
        w = SEVERITY_WEIGHTS[severity]
        weighted_contribution = severity * w * confidence
        total_raw_risk += weighted_contribution
        max_possible_risk += 3 * SEVERITY_WEIGHTS[3]  # Max: all contraindicated with 100% confidence

        # Track flagged interactions (severity > 0 or special flag)
        if severity > 0 or explanation.is_special_flag:
            flagged.append(pair)
            drug_interaction_counts[drug_a] += 1
            drug_interaction_counts[drug_b] += 1

    # Normalize to 0–100
    # Each pair contributes at most: 3 * 2.0 * 1.0 = 6.0 (contraindicated, full confidence)
    # Average contribution per pair, then scale to 0–100
    num_pairs = len(list(all_interactions))
    if num_pairs > 0 and max_possible_risk > 0:
        # Per-pair average risk (0–1 range)
        per_pair_max = 3 * SEVERITY_WEIGHTS[3] * 1.0  # = 6.0
        avg_risk_ratio = total_raw_risk / (num_pairs * per_pair_max)

        # Apply a non-linear amplifier so even 1 bad pair in a small list scores high.
        # Use sqrt to compress high-density combos and amplify sparse but severe ones.
        amplified = avg_risk_ratio ** 0.5

        # Scale so that:
        #  - 1 major pair (sev=2, conf=0.8) out of 1 pair → score ~65
        #  - 1 contraindicated pair (sev=3, conf=1.0) → score ~100
        #  - 3 moderate pairs (sev=1, conf=0.7) → score ~40–50
        overall_score = float(np.clip(amplified * 100, 0.0, 100.0))
    else:
        overall_score = 0.0
    overall_score = float(np.clip(overall_score, 0.0, 100.0))

    # Get risk tier
    tier_key, tier_label, tier_color = get_risk_tier(overall_score)

    # Special flags
    special_flags = detect_special_flags(drug_names)

    # Boost score for special flags
    if any(f.flag_type == "qt_prolongation" for f in special_flags):
        overall_score = min(100.0, overall_score + 15.0)
    if any(f.flag_type == "nsaid_anticoagulant" for f in special_flags):
        overall_score = min(100.0, overall_score + 10.0)

    # Re-get tier after boost
    tier_key, tier_label, tier_color = get_risk_tier(overall_score)

    # Warfarin warning
    warfarin_warning = any(_is_warfarin(d) for d in drug_names)

    # Summary
    n_flagged = len(flagged)
    n_severe = sum(1 for p in flagged if p.severity >= 2)
    n_moderate = sum(1 for p in flagged if p.severity == 1)
    n_special = len(special_flags)

    if n_flagged == 0:
        summary = f"No significant interactions detected among your {n} medications. Risk score: {overall_score:.0f}/100."
    else:
        parts = []
        if n_severe > 0:
            parts.append(f"{n_severe} high-risk interaction{'s' if n_severe > 1 else ''}")
        if n_moderate > 0:
            parts.append(f"{n_moderate} moderate interaction{'s' if n_moderate > 1 else ''}")
        if n_special > 0:
            parts.append(f"{n_special} special safety flag{'s' if n_special > 1 else ''}")
        summary = (
            f"Your {n} medications have {', and '.join(parts)} detected. "
            f"Risk score: {overall_score:.0f}/100. {tier_label}."
        )

    # Risk culprit (drug with most flagged interactions)
    risk_culprit = max(drug_interaction_counts, key=drug_interaction_counts.get) if drug_interaction_counts else None
    culprit_count = drug_interaction_counts.get(risk_culprit, 0) if risk_culprit else 0
    culprit_explanation = (
        f"{risk_culprit} is involved in {culprit_count} of your {n_flagged} flagged interactions. "
        f"Discuss with your doctor whether this medication can be reviewed or substituted."
        if risk_culprit and culprit_count > 0
        else "No single drug stands out as a primary risk contributor."
    )

    # Shapley values
    shapley_values = {}
    if include_shapley and n >= 2:
        try:
            from explainability.shapley_attribution import compute_drug_attribution

            def risk_fn(drugs: list[str]) -> float:
                """Partial risk score for a drug subset."""
                if len(drugs) < 2:
                    return 0.0
                subset_score = 0.0
                for a, b in itertools.combinations(drugs, 2):
                    pair = all_interactions.get((a, b)) or all_interactions.get((b, a))
                    if pair:
                        subset_score += pair.severity * SEVERITY_WEIGHTS[pair.severity] * pair.confidence
                return subset_score

            shapley_result = compute_drug_attribution(drug_names, risk_fn, n_samples=100)
            shapley_values = dict(zip(shapley_result.drug_names, shapley_result.shapley_values))
            if shapley_result.risk_culprit:
                risk_culprit = shapley_result.risk_culprit
                culprit_explanation = shapley_result.explanation
        except Exception as e:
            logger.debug(f"Shapley computation failed: {e}")

    # Sort flagged interactions by severity
    flagged.sort(key=lambda p: (p.severity, p.is_special_flag), reverse=True)

    return SafetyReport(
        drug_list=drug_names,
        overall_risk_score=overall_score,
        risk_tier=tier_key,
        risk_tier_label=tier_label,
        risk_tier_color=tier_color,
        summary=summary,
        flagged_interactions=flagged,
        all_interactions=all_interactions,
        special_flags=special_flags,
        warfarin_warning=warfarin_warning,
        risk_culprit=risk_culprit,
        risk_culprit_explanation=culprit_explanation,
        shapley_values=shapley_values,
        drug_interaction_counts=drug_interaction_counts,
        num_flagged=n_flagged,
        num_pairs_checked=len(all_interactions),
    )


def compute_risk_delta(
    current_drugs: list[str],
    new_drug: str,
    current_report: Optional[SafetyReport] = None,
    **kwargs,
) -> dict:
    """Compute risk delta when adding new_drug to current_drugs."""
    old_score = current_report.overall_risk_score if current_report else 0.0
    if not current_report:
        old_report = compute_polypharmacy_score(current_drugs, include_shapley=False, **kwargs)
        old_score = old_report.overall_risk_score

    new_report = compute_polypharmacy_score(
        current_drugs + [new_drug],
        include_shapley=False,
        **kwargs,
    )
    new_score = new_report.overall_risk_score

    # New interactions introduced by the new drug
    new_interactions = [
        pair for pair in new_report.flagged_interactions
        if new_drug in (pair.drug_a, pair.drug_b)
    ]

    return {
        "old_score": old_score,
        "new_score": new_score,
        "delta_score": new_score - old_score,
        "new_drug": new_drug,
        "new_interactions": new_interactions,
        "risk_increased": new_score > old_score,
    }


if __name__ == "__main__":
    # Quick test
    test_drugs = ["Warfarin", "Aspirin", "Ibuprofen", "Metformin", "Lisinopril"]
    report = compute_polypharmacy_score(test_drugs, include_shapley=True)
    print(f"Risk Score: {report.overall_risk_score:.1f}/100 [{report.risk_tier}]")
    print(f"Summary: {report.summary}")
    print(f"Special Flags: {len(report.special_flags)}")
    print(f"Flagged Pairs: {report.num_flagged}")
    print(f"Risk Culprit: {report.risk_culprit}")
