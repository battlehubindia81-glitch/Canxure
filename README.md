# Precision Oncology Virtual Tumor Digital Twin

**RESEARCH USE ONLY**

Predictions are computational hypotheses and are not clinical treatment recommendations. Model outputs must not be interpreted as proof of efficacy, safety, clinical benefit, or patient eligibility.

This repository implements a modular, Codespaces-compatible and Colab-orchestrated research framework for multimodal virtual-tumor simulations. Core logic lives in `src/` and `scripts/`; notebooks only orchestrate package calls.

## Workflow

Claude Code/Codex → GitHub Codespace CPU development → Git commit/push → GitHub → Google Colab clone/pull → optional CPU/GPU training → Google Drive artifacts.

## Quick start

```bash
pip install -r requirements-dev.txt
pytest
python scripts/run_simulation.py --demo
python scripts/train_baseline.py
python scripts/train_research.py --device auto
```

## Research model parameter estimate

The Research model uses expression, mutation, biology, and drug encoders plus multimodal fusion and multitask heads. Actual counts are computed by `src.models.metrics`; do not report parameter counts from documentation alone.

## Dependency notes

- RDKit/DeepChem are easiest to install in Colab/conda-like environments; this repo provides offline demo fingerprint fallback for tests.
- PyTorch is required for neural training, while PyTorch Geometric is optional and should match the active Torch/CUDA wheel.
- LightGBM is a classical baseline and is never described as neural trainable parameters.
