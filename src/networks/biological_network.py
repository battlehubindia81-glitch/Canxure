from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class BiologicalNetwork:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    pathways: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def demo(cls) -> "BiologicalNetwork":
        net = cls(pathways={"MAPK": ["EGFR", "RAS", "RAF", "MEK", "ERK"], "PI3K_AKT": ["PIK3CA", "PI3K", "AKT", "MTOR"], "DDR": ["TP53", "BRCA1", "BRCA2", "ATM"]})
        for chain in net.pathways.values():
            for a, b in zip(chain, chain[1:]):
                net.graph.add_edge(a, b, weight=1.0, sign=1.0)
        net.graph.add_edge("EGFR", "PI3K", weight=0.7, sign=1.0)
        return net

    def propagate(self, mutations: dict[str, float], depth: int = 3, decay: float = 0.65) -> dict[str, float]:
        """Propagate mutation perturbations through directed biological edges.

        Purpose: convert mutation state into network/pathway perturbation signals.
        Inputs: gene-to-strength mapping. Input shape: sparse mapping of genes.
        Outputs: gene-to-activity mapping. Output shape: sparse mapping of genes.
        Data types: dict[str, float].
        Exceptions: none; unseen genes remain as mutation seeds.
        Assumptions: demo edge signs/weights are illustrative, not clinical evidence.
        """
        state = dict(mutations)
        frontier = dict(mutations)
        for step in range(depth):
            nxt: dict[str, float] = {}
            for gene, value in frontier.items():
                for _, nbr, attrs in self.graph.out_edges(gene, data=True):
                    delta = value * float(attrs.get("weight", 1.0)) * float(attrs.get("sign", 1.0)) * (decay ** (step + 1))
                    nxt[nbr] = nxt.get(nbr, 0.0) + delta
            for gene, value in nxt.items():
                state[gene] = state.get(gene, 0.0) + value
            frontier = nxt
        return state

    def pathway_activity(self, network_state: dict[str, float]) -> dict[str, float]:
        return {name: float(np.mean([network_state.get(g, 0.0) for g in genes])) for name, genes in self.pathways.items()}
