from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, model: Any, optimizer: Any | None, epoch: int, metric: float, config: dict[str, Any], seed: int = 42) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer else None, "validation_metric": metric, "configuration": config, "random_seed": seed}, p)


def load_checkpoint(path: str | Path, model: Any, optimizer: Any | None = None) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and ckpt.get("optimizer_state"):
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
