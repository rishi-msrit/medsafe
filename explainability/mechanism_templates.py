"""
explainability/mechanism_templates.py
======================================
Template-based mechanism explanation for drug-drug interactions.

Given the dominant edge type from GNNExplainer attribution,
generates human-readable mechanism explanations covering:
  - CYP450 metabolic competition
  - Pharmacodynamic synergy/antagonism
  - Absorption interference
  - Special clinical scenarios (QT prolongation, CNS depression, bleeding risk)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class MechanismExplanation:
    """Structured mechanism explanation for a drug-drug interaction."""
    mechanism_type: str           # Category: cyp450_metabolic, pharmacodynamic, etc.
    severity_label: str           # contraindicated / major / moderate / minor
    plain_english: str            # Human-readable explanation
    clinical_implication: str     # What the patient/doctor should do
    evidence_basis: str           # "DrugBank curated", "FAERS pharmacovigilance", etc.
    affected_cyp_enzymes: list[str]  # CYP enzymes involved (if applicable)
    is_special_flag: bool         # QT, CNS, bleeding, Warfarin — requires banner


SEVERITY_LABELS = {0: "minor", 1: "moderate", 2: "major", 3: "contraindicated"}

SEVERITY_CLINICAL = {
    3: "These drugs should NEVER be taken together. Consult your doctor immediately.",
    2: "Avoid combining unless benefit clearly outweighs risk. Close medical supervision required.",
    1: "This combination requires monitoring. Report any unusual symptoms to your doctor or pharmacist.",
    0: "This interaction is generally manageable. Note any new symptoms and report to your pharmacist.",
}

# CYP enzyme → affected pharmacological effect
CYP_EFFECTS = {
    "CYP1A2": "metabolism of caffeine, theophylline, and certain antidepressants",
    "CYP2C9": "metabolism of warfarin and NSAIDs",
    "CYP2C19": "metabolism of proton pump inhibitors and certain antidepressants",
    "CYP2D6": "metabolism of codeine, tramadol, and many antidepressants",
    "CYP3A4": "metabolism of approximately 50% of all prescription drugs",
}

# Mechanism type → explanation template
MECHANISM_TEMPLATES = {
    "cyp450_metabolic": (
        "Metabolic Competition via {cyp_list}: "
        "Both drugs are metabolized by the same liver enzyme(s). "
        "{drug_b} levels may rise to potentially toxic concentrations because "
        "{drug_a} slows its breakdown. "
        "CYP enzymes affected: {cyp_list}, which handles {cyp_effect}."
    ),
    "metabolic": (
        "Metabolic Interaction: "
        "{drug_a} affects how your liver processes {drug_b}, "
        "potentially altering its blood levels and effectiveness."
    ),
    "pharmacodynamic": (
        "Pharmacodynamic Interaction: "
        "{drug_a} and {drug_b} both act on the same biological system. "
        "Their combined effect may be stronger (additive or synergistic) "
        "or weaker (antagonistic) than either drug alone."
    ),
    "absorption": (
        "Absorption Interaction: "
        "{drug_a} affects how {drug_b} is absorbed from your digestive tract, "
        "potentially reducing or enhancing its effectiveness."
    ),
    "transporter": (
        "Drug Transporter Interaction: "
        "Both drugs interact with protein pumps (e.g., P-glycoprotein) "
        "that control how drugs move in and out of cells, "
        "potentially altering the distribution of {drug_b}."
    ),
    "cardiac_qt": (
        "Cardiac QT Prolongation Risk: "
        "CRITICAL WARNING — Both {drug_a} and {drug_b} can prolong the QT interval "
        "(an electrical measurement of heart rhythm). "
        "Combining them significantly increases the risk of a potentially fatal "
        "heart rhythm disorder called Torsades de Pointes."
    ),
    "bleeding": (
        "Bleeding Risk: "
        "{drug_a} and {drug_b} together significantly increase your risk of serious bleeding. "
        "NSAIDs (like ibuprofen and aspirin) impair platelet function while "
        "anticoagulants (like warfarin) reduce clotting protein activity — "
        "their combination is one of the most dangerous in all of medicine."
    ),
    "cns_depression": (
        "Additive CNS (Brain/Nervous System) Depression: "
        "Both {drug_a} and {drug_b} slow down the central nervous system. "
        "Combined, they can cause excessive sedation, impaired breathing, "
        "confusion, and in severe cases, coma or death."
    ),
    "serotonin_syndrome": (
        "Serotonin Syndrome Risk: "
        "{drug_a} and {drug_b} both increase serotonin levels in the brain. "
        "Combined, this can cause a dangerous condition called serotonin syndrome: "
        "agitation, rapid heart rate, high body temperature, and muscle rigidity."
    ),
    "renal": (
        "Kidney Function Interaction: "
        "{drug_a} and {drug_b} may have additive effects on the kidneys, "
        "potentially reducing kidney function or altering how drugs are cleared from your body."
    ),
    "unknown": (
        "Drug Interaction Detected: "
        "A potential interaction between {drug_a} and {drug_b} has been identified in clinical databases. "
        "The exact mechanism is not fully characterized. "
        "Please consult your pharmacist or doctor for personalized guidance."
    ),
}


# Special scenario detectors
_QT_KEYWORDS = {"amiodarone", "sotalol", "dofetilide", "quinidine", "haloperidol", "methadone",
                 "clarithromycin", "azithromycin", "moxifloxacin", "ondansetron", "hydroxychloroquine"}
_CNS_KEYWORDS = {"benzodiazepine", "opioid", "morphine", "oxycodone", "fentanyl", "diazepam",
                 "lorazepam", "alprazolam", "zolpidem", "diphenhydramine", "alcohol"}
_NSAID_KEYWORDS = {"ibuprofen", "naproxen", "aspirin", "diclofenac", "celecoxib", "indomethacin"}
_ANTICOAG_KEYWORDS = {"warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran", "enoxaparin"}
_WARFARIN_KEYWORDS = {"warfarin"}


def _normalize(name: str) -> str:
    return name.lower().strip()


def detect_special_scenario(
    drug_a: str,
    drug_b: str,
    drug_a_categories: str = "",
    drug_b_categories: str = "",
) -> Optional[str]:
    """
    Detect high-priority special interaction scenarios.

    Returns the mechanism type if a special scenario is detected, else None.
    """
    na = _normalize(drug_a)
    nb = _normalize(drug_b)
    ca = drug_a_categories.lower() if drug_a_categories else ""
    cb = drug_b_categories.lower() if drug_b_categories else ""

    # Warfarin (any combination → always flag regardless)
    if na in _WARFARIN_KEYWORDS or nb in _WARFARIN_KEYWORDS:
        return "warfarin_special"

    # QT prolongation (both drugs in QT list)
    if (na in _QT_KEYWORDS or nb in _QT_KEYWORDS):
        if (na in _QT_KEYWORDS and nb in _QT_KEYWORDS):
            return "cardiac_qt"

    # Bleeding (NSAID + anticoagulant)
    nsaid_a = na in _NSAID_KEYWORDS or "nsaid" in ca
    nsaid_b = nb in _NSAID_KEYWORDS or "nsaid" in cb
    anticoag_a = na in _ANTICOAG_KEYWORDS or "anticoagulant" in ca
    anticoag_b = nb in _ANTICOAG_KEYWORDS or "anticoagulant" in cb

    if (nsaid_a and anticoag_b) or (nsaid_b and anticoag_a):
        return "bleeding"

    # CNS depression (both are CNS depressants)
    cns_a = na in _CNS_KEYWORDS or any(kw in ca for kw in ["opioid", "benzodiazepine", "sedative"])
    cns_b = nb in _CNS_KEYWORDS or any(kw in cb for kw in ["opioid", "benzodiazepine", "sedative"])
    if cns_a and cns_b:
        return "cns_depression"

    return None


def extract_cyp_enzymes_from_description(description: str) -> list[str]:
    """Extract mentioned CYP enzyme names from an interaction description."""
    cyp_pattern = re.compile(r"CYP[0-9][A-Z][0-9]+", re.IGNORECASE)
    found = cyp_pattern.findall(description)
    # Normalize: CYP3a4 → CYP3A4
    normalized = []
    for cyp in found:
        cyp_upper = cyp.upper()
        if cyp_upper not in normalized:
            normalized.append(cyp_upper)
    return normalized


def generate_mechanism_explanation(
    drug_a: str,
    drug_b: str,
    mechanism_type: str,
    severity: int,
    gnn_explainer_edges: Optional[list[dict]] = None,
    drugbank_description: Optional[str] = None,
    drug_a_categories: str = "",
    drug_b_categories: str = "",
    evidence_sources: Optional[list[str]] = None,
    support_count: int = 1,
) -> MechanismExplanation:
    """
    Generate a complete mechanism explanation for a drug-drug interaction.

    Args:
        drug_a:                Name of drug A
        drug_b:                Name of drug B
        mechanism_type:        Mechanism category (from pipeline classification)
        severity:              Integer severity 0–3
        gnn_explainer_edges:   Edge importance dicts from GNNExplainer
        drugbank_description:  Raw DrugBank description text (for CYP extraction)
        drug_a_categories:     Drug A categories string
        drug_b_categories:     Drug B categories string
        evidence_sources:      List of data sources (DrugBank, FAERS, TWOSIDES)
        support_count:         Number of data sources supporting this interaction

    Returns:
        MechanismExplanation dataclass
    """
    # Extract CYP enzymes from description or GNNExplainer edges
    cyp_enzymes = []
    if drugbank_description:
        cyp_enzymes = extract_cyp_enzymes_from_description(drugbank_description)

    if gnn_explainer_edges:
        for edge in gnn_explainer_edges:
            edge_type = edge.get("edge_type", "")
            if edge_type == "shares_cyp_enzyme":
                cyp_id = edge.get("cyp_enzyme_id", 0)
                cyp_names = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
                if 0 <= cyp_id < len(cyp_names):
                    cyp = cyp_names[cyp_id]
                    if cyp not in cyp_enzymes:
                        cyp_enzymes.append(cyp)

    # Check for special scenarios (override mechanism type)
    special = detect_special_scenario(drug_a, drug_b, drug_a_categories, drug_b_categories)
    if special:
        mechanism_type = special
        if special == "cardiac_qt":
            severity = max(severity, 2)  # At least major
        elif special in ("bleeding", "cns_depression", "warfarin_special"):
            severity = max(severity, 1)

    # Cap severity at 3
    severity = min(max(severity, 0), 3)
    severity_label = SEVERITY_LABELS[severity]

    # Format CYP list string
    cyp_list_str = ", ".join(cyp_enzymes) if cyp_enzymes else "CYP450 enzymes"
    cyp_effect = CYP_EFFECTS.get(cyp_enzymes[0], "drug metabolism") if cyp_enzymes else "drug metabolism"

    # Select and fill template
    template_key = mechanism_type
    if template_key not in MECHANISM_TEMPLATES:
        template_key = "unknown"

    # Special Warfarin handling
    if mechanism_type == "warfarin_special":
        plain = (
            f"⚠️  WARFARIN ALERT: Warfarin has an extremely narrow therapeutic window "
            f"and interacts with over 200 drugs. Any change to your medications requires "
            f"pharmacist or physician review and likely INR (blood clotting) monitoring. "
            f"The interaction between {drug_a} and {drug_b} may alter warfarin's anticoagulant effect."
        )
        is_special = True
    else:
        template = MECHANISM_TEMPLATES[template_key]
        try:
            plain = template.format(
                drug_a=drug_a,
                drug_b=drug_b,
                cyp_list=cyp_list_str,
                cyp_effect=cyp_effect,
            )
        except KeyError:
            plain = template.format(drug_a=drug_a, drug_b=drug_b)
        is_special = mechanism_type in ("cardiac_qt", "bleeding", "cns_depression", "warfarin_special")

    # Clinical implication
    clinical = SEVERITY_CLINICAL[severity]

    # Evidence basis
    if support_count < 5:
        clinical = (
            f"⚠️  Limited data ({support_count} source(s)). Interaction status uncertain. "
            f"Consult a pharmacist for personalized advice. " + clinical
        )

    sources = evidence_sources or ["DrugBank"]
    evidence_basis = f"Sources: {', '.join(sources)} ({support_count} supporting records)"

    return MechanismExplanation(
        mechanism_type=mechanism_type,
        severity_label=severity_label,
        plain_english=plain,
        clinical_implication=clinical,
        evidence_basis=evidence_basis,
        affected_cyp_enzymes=cyp_enzymes,
        is_special_flag=is_special,
    )


def explain_multiple_interactions(
    drug_list: list[str],
    interaction_pairs: list[dict],
) -> list[dict]:
    """
    Generate explanations for all flagged interactions in a polypharmacy analysis.

    Args:
        drug_list:          All drugs in the patient's medication list
        interaction_pairs:  List of dicts with keys: drug_a, drug_b, severity,
                           mechanism_type, description, support_count, etc.

    Returns:
        List of explanation dicts, sorted by severity (descending)
    """
    explanations = []

    for pair in interaction_pairs:
        drug_a = pair.get("drug_a", "Drug A")
        drug_b = pair.get("drug_b", "Drug B")
        severity = int(pair.get("severity", 1))
        mtype = pair.get("mechanism_type", "unknown")
        desc = pair.get("description", "")
        support = int(pair.get("support_count", 1))
        sources = pair.get("evidence_sources", ["DrugBank"])

        explanation = generate_mechanism_explanation(
            drug_a=drug_a,
            drug_b=drug_b,
            mechanism_type=mtype,
            severity=severity,
            drugbank_description=desc,
            evidence_sources=sources,
            support_count=support,
        )

        explanations.append({
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": severity,
            "severity_label": explanation.severity_label,
            "mechanism_type": explanation.mechanism_type,
            "plain_english": explanation.plain_english,
            "clinical_implication": explanation.clinical_implication,
            "evidence_basis": explanation.evidence_basis,
            "cyp_enzymes": explanation.affected_cyp_enzymes,
            "is_special_flag": explanation.is_special_flag,
        })

    # Sort by severity descending, then by is_special_flag
    explanations.sort(key=lambda x: (x["severity"], x["is_special_flag"]), reverse=True)

    return explanations


if __name__ == "__main__":
    # Test examples
    print("=== CYP3A4 Metabolic Interaction ===")
    result = generate_mechanism_explanation(
        drug_a="Clarithromycin",
        drug_b="Simvastatin",
        mechanism_type="cyp450_metabolic",
        severity=2,
        drugbank_description="Clarithromycin is a strong inhibitor of CYP3A4.",
    )
    print(f"Mechanism: {result.mechanism_type}")
    print(f"Severity: {result.severity_label}")
    print(f"Explanation: {result.plain_english}")
    print()

    print("=== Warfarin + NSAIDs ===")
    result2 = generate_mechanism_explanation(
        drug_a="Warfarin",
        drug_b="Ibuprofen",
        mechanism_type="bleeding",
        severity=2,
    )
    print(f"Special Flag: {result2.is_special_flag}")
    print(f"Explanation: {result2.plain_english}")
    print()

    print("=== Detect Special Scenario ===")
    scenario = detect_special_scenario("Morphine", "Lorazepam")
    print(f"Morphine + Lorazepam: {scenario}")
