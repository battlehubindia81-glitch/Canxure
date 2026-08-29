from __future__ import annotations

import pandas as pd
import streamlit as st

from src.biology.patient import PatientTumor
from src.data.demo import DEMO_DRUGS, synthetic_expression
from src.data.preprocessing import preprocess_expression
from src.simulation.virtual_chamber import VirtualSimulationChamber, results_to_dicts

st.set_page_config(page_title="Research Virtual Tumor Digital Twin", layout="wide")
st.title("Precision Oncology Virtual Tumor Digital Twin")
st.error("RESEARCH USE ONLY — Predictions are computational hypotheses and are not clinical treatment recommendations.")
cancer_type = st.sidebar.selectbox("Cancer type", ["DEMO", "Lung", "Colorectal", "Breast"])
gene = st.sidebar.selectbox("Mutation gene", ["EGFR", "KRAS", "BRAF", "PIK3CA", "TP53"])
variant = st.sidebar.text_input("Variant", "L858R")
strength = st.sidebar.slider("Perturbation strength", 0.0, 1.0, 0.8)
strategy = st.sidebar.selectbox("Ranking strategy", ["balanced", "efficacy-first", "evidence-first", "access-aware"])
expr, _ = preprocess_expression(synthetic_expression(4))
genes = [c for c in expr.columns if c != "sample_id"]
patient = PatientTumor("DEMO-001", cancer_type, dict(zip(genes, expr.iloc[0][genes].astype(float))))
patient.simulate_mutation(gene, variant, strength)
results = VirtualSimulationChamber(patient, DEMO_DRUGS, genes=genes).run(strategy)
st.subheader("Biology: pathway activity")
st.json(patient.get_pathway_activity())
st.subheader("Drug response research ranking")
st.dataframe(pd.DataFrame(results_to_dicts(results))[['drug_name','efficacy_score','toxicity_risk','resistance_risk','confidence_category','regulatory_status','affordability_category','composite_research_score']])
st.caption("Every field is synthetic/offline unless externally configured with provenance. Do not use for patient eligibility or treatment decisions.")
