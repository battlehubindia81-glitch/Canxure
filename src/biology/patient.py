from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.networks.biological_network import BiologicalNetwork


@dataclass
class PatientTumor:
    patient_id: str
    cancer_type: str
    expression_profile: dict[str, float]
    mutation_profile: dict[str, dict[str, Any]] = field(default_factory=dict)
    copy_number: dict[str, float] = field(default_factory=dict)
    clinical_context: dict[str, Any] = field(default_factory=dict)
    biological_network: BiologicalNetwork = field(default_factory=BiologicalNetwork.demo)

    def __post_init__(self) -> None:
        self._baseline_mutations = deepcopy(self.mutation_profile)

    def get_expression_vector(self, genes: list[str]) -> np.ndarray:
        return np.array([self.expression_profile.get(g, 0.0) for g in genes], dtype=np.float32)

    def get_mutation_state(self) -> dict[str, float]:
        return {gene: float(meta.get("allele_frequency", 1.0)) for gene, meta in self.mutation_profile.items()}

    def get_network_state(self) -> dict[str, float]:
        return self.biological_network.propagate(self.get_mutation_state())

    def get_pathway_activity(self) -> dict[str, float]:
        return self.biological_network.pathway_activity(self.get_network_state())

    def simulate_mutation(self, gene: str, alteration: str, allele_frequency: float = 1.0) -> None:
        """Apply a mutation and update downstream network/pathway state indirectly.

        Purpose: support interactive virtual perturbations without retraining.
        Inputs: gene, alteration, allele frequency. Input shape: scalar strings/float.
        Outputs: None; internal mutation state changes. Output shape: not applicable.
        Data types: str, float.
        Exceptions: ValueError for invalid gene/allele_frequency.
        Assumptions: propagation is a hypothesis-generation abstraction.
        """
        if not gene:
            raise ValueError("gene is required")
        if not 0 <= allele_frequency <= 1:
            raise ValueError("allele_frequency must be in [0, 1]")
        self.mutation_profile[gene] = {"variant": alteration, "variant_type": "simulated", "allele_frequency": allele_frequency}

    def reset_state(self) -> None:
        self.mutation_profile = deepcopy(self._baseline_mutations)
