from __future__ import annotations

from pathlib import Path

from src.data.schemas import PredictionResult

DISCLAIMER = """RESEARCH USE ONLY\n\nPredictions are computational hypotheses and are not clinical treatment recommendations. Model outputs should not be interpreted as proof of efficacy, safety, clinical benefit, or patient eligibility."""


def generate_report(path: str | Path, patient_id: str, cancer_type: str, pathway_activity: dict[str, float], results: list[PredictionResult], synthetic: bool = True) -> Path:
    """Generate an executive markdown report with research disclaimers.

    Purpose: persist digital-twin simulation outputs and limitations.
    Inputs: path, patient metadata, pathway mapping, prediction results. Shapes: scalar metadata and (n_drugs,) results.
    Outputs: written Path. Output shape: one file.
    Data types: strings, dict, dataclasses.
    Exceptions: OSError if path cannot be written.
    Assumptions: reported predictions are research-only computational hypotheses.
    """
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Precision Oncology Virtual Tumor Digital Twin Report", "", DISCLAIMER, ""]
    if synthetic:
        lines += ["**SYNTHETIC DATA — DEMONSTRATION ONLY — NOT CLINICAL EVIDENCE**", ""]
    lines += [f"## Patient/Tumor Overview", f"- Patient: {patient_id}", f"- Cancer type: {cancer_type}", "", "## Pathway Activity"]
    lines += [f"- {k}: {v:.3f}" for k, v in pathway_activity.items()]
    lines += ["", "## Drug Predictions"]
    for idx, r in enumerate(results[:20], start=1):
        lines += [f"### {idx}. {r.drug_name}", f"- Computational efficacy: {r.efficacy_score:.3f}", f"- Predicted toxicity: {r.toxicity_risk:.3f}", f"- Resistance risk: {r.resistance_risk:.3f}", f"- Uncertainty: {r.uncertainty:.3f}", f"- Regulatory status: {r.regulatory_status}", f"- Affordability: {r.affordability_category}", f"- Overall research ranking score: {r.composite_research_score:.3f}", ""]
    lines += ["## Limitations", "- Demonstration data and model outputs are not validated clinical evidence.", "- Attribution, when available, describes model behavior and does not establish biological causation.", "", DISCLAIMER]
    p.write_text("\n".join(lines))
    return p
