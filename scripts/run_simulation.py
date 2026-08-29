#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.biology.patient import PatientTumor
from src.data.demo import DEMO_DRUGS, synthetic_expression
from src.data.preprocessing import preprocess_expression
from src.reporting.report import generate_report
from src.simulation.virtual_chamber import VirtualSimulationChamber, results_to_dicts
from src.utils.environment import print_environment_report
from src.utils.reproducibility import set_random_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run synthetic demonstration workflow")
    args = parser.parse_args()
    if not args.demo:
        raise SystemExit("Only --demo is supported without external datasets")
    set_random_seed()
    env = print_environment_report()
    expr_raw = synthetic_expression(n_samples=8)
    expr, meta = preprocess_expression(expr_raw, artifact_dir="data/metadata")
    genes = [c for c in expr.columns if c != "sample_id"]
    profile = dict(zip(genes, expr.iloc[0][genes].astype(float)))
    patient = PatientTumor("DEMO-001", "DEMO", profile, {"EGFR": {"variant": "L858R", "variant_type": "SNV", "allele_frequency": 0.8}})
    chamber = VirtualSimulationChamber(patient, DEMO_DRUGS, genes=genes)
    results = chamber.run()
    out_pred = Path("outputs/predictions/demo_predictions.csv"); out_pred.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results_to_dicts(results)).to_csv(out_pred, index=False)
    report = generate_report("outputs/reports/digital_twin_report.md", patient.patient_id, patient.cancer_type, patient.get_pathway_activity(), results)
    model_summary = {"model": "demo deterministic simulation", "parameters": 0, "dataset_size": len(expr), "drugs": len(DEMO_DRUGS), "hardware": env.__dict__, "preprocessing": meta}
    Path("outputs/model").mkdir(parents=True, exist_ok=True)
    Path("outputs/model/model_summary.json").write_text(json.dumps(model_summary, indent=2))
    Path("outputs/model/model_summary.txt").write_text(json.dumps(model_summary, indent=2))
    print("=" * 60)
    print("PRECISION ONCOLOGY VIRTUAL TUMOR DIGITAL TWIN")
    print("Dataset\n-------")
    print(f"Tumors:                   {len(expr)}")
    print(f"Genes:                    {len(genes)}")
    print(f"Drugs:                    {len(DEMO_DRUGS)}")
    print("Simulation\n----------")
    print(f"Patient:                  {patient.patient_id}")
    print(f"Cancer:                   {patient.cancer_type}")
    print("Top computational candidates")
    for i, r in enumerate(results[:3], 1):
        print(f"{i}. {r.drug_name} Efficacy={r.efficacy_score:.3f} Toxicity={r.toxicity_risk:.3f} Resistance={r.resistance_risk:.3f} Uncertainty={r.uncertainty:.3f}")
    print(f"Artifacts: {out_pred}, {report}")
    print("RESEARCH USE ONLY - NOT A CLINICAL TREATMENT RECOMMENDATION")


if __name__ == "__main__":
    main()
