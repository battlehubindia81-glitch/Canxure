from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from src.chemistry.fingerprints import DrugRecord, smiles_to_fingerprint
from src.data.schemas import PredictionResult
from src.real_world.feasibility import RealWorldFeasibilityEngine


class VirtualSimulationChamber:
    def __init__(self, patient, drug_library: list[DrugRecord], model=None, genes: list[str] | None = None):
        self.patient = patient
        self.drug_library = drug_library
        self.model = model
        self.genes = genes or list(patient.expression_profile)[:500]
        self.feasibility = RealWorldFeasibilityEngine()

    def run(self, ranking_strategy: str = "balanced") -> list[PredictionResult]:
        """Batch-evaluate a patient against a drug library without reloading the model.

        Purpose: virtual tumor-drug simulation for research hypotheses.
        Inputs: ranking strategy scalar. Input shape: one string.
        Outputs: list of PredictionResult. Output shape: (n_drugs,).
        Data types: dataclass list.
        Exceptions: ValueError for invalid drug SMILES from fingerprinting.
        Assumptions: demo fallback scores are deterministic hypotheses, not evidence.
        """
        fps = np.vstack([smiles_to_fingerprint(d.smiles) for d in self.drug_library])
        expr = self.patient.get_expression_vector(self.genes)
        pathway = self.patient.get_pathway_activity()
        biology_score = float(np.tanh(sum(pathway.values()) if pathway else 0.0))
        base = float(np.tanh(np.mean(expr)))
        results = []
        for i, drug in enumerate(self.drug_library):
            fp_score = float(fps[i].mean())
            pIC50 = 5.0 + base + fp_score + biology_score * 0.2
            logIC50 = -pIC50
            ic50 = 10 ** logIC50
            efficacy = float(1 / (1 + math.exp(-(pIC50 - 5))))
            toxicity = float(min(1.0, max(0.0, 0.25 + fp_score)))
            resistance = float(min(1.0, max(0.0, 0.35 - biology_score * 0.1)))
            uncertainty = 0.15 + (0.15 if not self.model else 0.05)
            confidence = "LOW" if uncertainty > 0.25 else "MODERATE"
            feas = self.feasibility.assess(drug.drug_name)
            composite = self._score(efficacy, toxicity, resistance, abs(biology_score), uncertainty, ranking_strategy)
            results.append(PredictionResult(self.patient.patient_id, drug.drug_id, drug.drug_name, pIC50, logIC50, ic50, efficacy, toxicity, resistance, efficacy * (1 - uncertainty), abs(biology_score), uncertainty, 0.0, confidence, composite_research_score=composite, **feas))
        return sorted(results, key=lambda r: r.composite_research_score, reverse=True)

    @staticmethod
    def _score(efficacy: float, toxicity: float, resistance: float, biology: float, uncertainty: float, strategy: str) -> float:
        weights = {
            "efficacy-first": (0.65, 0.1, 0.1, 0.1, 0.05),
            "balanced": (0.4, 0.2, 0.15, 0.1, 0.15),
            "evidence-first": (0.3, 0.15, 0.15, 0.1, 0.3),
            "access-aware": (0.35, 0.2, 0.15, 0.1, 0.2),
        }.get(strategy, (0.4, 0.2, 0.15, 0.1, 0.15))
        return weights[0] * efficacy + weights[1] * (1 - toxicity) + weights[2] * (1 - resistance) + weights[3] * biology + weights[4] * (1 - uncertainty)


def results_to_dicts(results: list[PredictionResult]) -> list[dict[str, object]]:
    return [asdict(r) for r in results]
