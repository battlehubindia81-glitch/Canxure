#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.models.metrics import benchmark_inference, checkpoint_size_mb, count_non_trainable_parameters, count_parameters, count_trainable_parameters, estimate_model_memory, parameters_by_module
from src.models.research_model import ResearchDigitalTwin
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.utils.environment import print_environment_report
from src.utils.reproducibility import set_random_seed


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable: requested cuda but CUDA is not available")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume")
    args = parser.parse_args()
    set_random_seed()
    env = print_environment_report()
    device = resolve_device(args.device)
    model = ResearchDigitalTwin().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if args.resume:
        load_checkpoint(args.resume, model, opt)
    x = (torch.randn(16, 500, device=device), torch.randn(16, 64, device=device), torch.randn(16, 64, device=device), torch.randn(16, 1024, device=device))
    y = torch.randn(16, 1, device=device)
    start = time.perf_counter()
    model.train(); opt.zero_grad(); loss = torch.nn.functional.mse_loss(model(*x)["efficacy"], y); loss.backward(); opt.step()
    training_time = time.perf_counter() - start
    ckpt = Path("models/research/checkpoints/latest.pt")
    save_checkpoint(ckpt, model, opt, 1, float(loss.item()), {"device": str(device)})
    summary = {"model_version": "research-v001", "architecture": "multimodal MLP with network fallback interface", "total_parameters": count_parameters(model), "trainable_parameters": count_trainable_parameters(model), "non_trainable_parameters": count_non_trainable_parameters(model), "parameters_by_module": parameters_by_module(model), "memory": estimate_model_memory(model), "checkpoint_size_mb": checkpoint_size_mb(model), "inference": benchmark_inference(model, x, runs=3), "training_time_seconds": training_time, "loss": float(loss.item()), "device": str(device), "hardware": env.__dict__}
    Path("outputs/model").mkdir(parents=True, exist_ok=True)
    Path("outputs/model/model_summary.json").write_text(json.dumps(summary, indent=2))
    Path("outputs/model/model_summary.txt").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
