from __future__ import annotations

import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if valid.sum() == 0:
        return pred.sum() * 0.0
    return torch.mean((pred[valid] - target[valid]) ** 2)
