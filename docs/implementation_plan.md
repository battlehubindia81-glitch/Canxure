# Implementation Plan

## Codespaces responsibilities

- Source-code development, unit tests, static checks, lightweight synthetic demos, Git commits, and PRs.
- CPU-safe scripts: `scripts/run_simulation.py --demo`, `scripts/train_baseline.py`, and smoke tests for neural model construction when PyTorch is installed.

## Google Colab responsibilities

- Runtime synchronization using `scripts/sync_colab.py`.
- Optional Google Drive mounting for large datasets, checkpoints, cached fingerprints, and experiment artifacts.
- CPU/GPU neural training using `scripts/train_research.py --device auto` with resumable checkpoints.

## Package interfaces

- `src.data`: schemas, synthetic demo data, expression preprocessing, provenance artifacts.
- `src.chemistry`: SMILES validation and 1024-bit ECFP-style fingerprint generation with RDKit fallback.
- `src.networks` and `src.biology`: network propagation, pathway activity, and `PatientTumor` mutation simulation.
- `src.models`: Lite and Research PyTorch architectures plus actual parameter/memory/latency utilities.
- `src.training`: masked losses and checkpoint save/load.
- `src.simulation`: `VirtualSimulationChamber` batch drug inference and transparent ranking weights.
- `src.real_world`: feasibility placeholders that preserve provenance and avoid fabricated claims.
- `src.reporting`: markdown executive reports with research-only disclaimers.

## Data schemas

- Expression: `sample_id,gene1,...,geneN`.
- Drug: `drug_id,drug_name,SMILES` plus `fp_0..fp_1023`.
- Response: `sample_id,drug_id,response` with explicit endpoint configuration.
- Prediction: structured `PredictionResult` dataclass with efficacy, toxicity, resistance, uncertainty, OOD, feasibility, and provenance fields.

## Research model parameter estimate

The default Research model has approximately 2.5M trainable parameters from expression, mutation, biology, drug, fusion, and multitask-head modules. The code computes the actual count at runtime with `count_parameters`; the implementation intentionally does not inflate the network solely to satisfy a headline number, and it remains below the requested 10M upper bound for CPU/free-Colab viability.

## Dependency conflict notes

- RDKit and DeepChem installation is often more reliable in Colab/conda-compatible environments than minimal Codespaces images; core tests use an offline fingerprint fallback if RDKit is unavailable.
- PyTorch Geometric must match the installed PyTorch and CUDA wheels; it is optional and not required for baseline demos.
- PyTorch is required for neural training and checkpointing, but baseline/simulation code remains separable.
- LightGBM may be unavailable on minimal Python runtimes; `scripts/train_baseline.py` falls back to scikit-learn gradient boosting while documenting that LightGBM is the intended baseline when installed.
