from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionResult:
    patient_id: str
    drug_id: str
    drug_name: str
    predicted_pIC50: float
    predicted_logIC50: float
    predicted_IC50: float
    efficacy_score: float
    toxicity_risk: float
    resistance_risk: float
    clinical_response_probability: float
    biological_plausibility: float
    uncertainty: float
    OOD_score: float
    confidence_category: str
    evidence_level: str = "DATA_UNAVAILABLE"
    regulatory_status: str = "UNKNOWN"
    availability_status: str = "UNKNOWN"
    affordability_category: str = "PRICE_DATA_UNAVAILABLE"
    generic_available: bool | None = None
    biosimilar_available: bool | None = None
    clinical_trial_count: int = 0
    composite_research_score: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)
