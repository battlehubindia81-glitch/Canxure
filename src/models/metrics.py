from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any


def count_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_non_trainable_parameters(model: Any) -> int:
    return count_parameters(model) - count_trainable_parameters(model)


def parameters_by_module(model: Any) -> dict[str, int]:
    return {name: sum(p.numel() for p in module.parameters(recurse=False)) for name, module in model.named_modules() if name}


def estimate_model_memory(model: Any, bytes_per_parameter: int = 4) -> dict[str, float]:
    total = count_parameters(model) * bytes_per_parameter
    return {"parameters_mb": round(total / 1024**2, 3), "training_estimate_mb": round(total * 4 / 1024**2, 3)}


def checkpoint_size_mb(model: Any) -> float:
    import torch

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = Path(tmp.name)
    torch.save(model.state_dict(), path)
    size = path.stat().st_size / 1024**2
    path.unlink(missing_ok=True)
    return round(size, 3)


def benchmark_inference(model: Any, sample_inputs: tuple[Any, ...], runs: int = 10) -> dict[str, float]:
    import torch

    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(runs):
            model(*sample_inputs)
        elapsed = time.perf_counter() - start
    batch = int(sample_inputs[0].shape[0])
    return {"latency_ms": round((elapsed / runs) * 1000, 3), "throughput_per_second": round((batch * runs) / elapsed, 3)}
