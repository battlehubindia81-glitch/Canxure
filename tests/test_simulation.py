from src.biology.patient import PatientTumor
from src.data.demo import DEMO_DRUGS, synthetic_expression
from src.data.preprocessing import preprocess_expression
from src.simulation.virtual_chamber import VirtualSimulationChamber


def test_simulation_twenty_drugs():
    expr, _ = preprocess_expression(synthetic_expression(2))
    genes = [c for c in expr.columns if c != "sample_id"]
    p = PatientTumor("P1", "DEMO", dict(zip(genes, expr.iloc[0][genes].astype(float))))
    p.simulate_mutation("EGFR", "L858R")
    results = VirtualSimulationChamber(p, DEMO_DRUGS, genes=genes).run()
    assert len(results) == 20
    assert results[0].composite_research_score >= results[-1].composite_research_score
