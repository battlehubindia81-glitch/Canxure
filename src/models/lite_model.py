from __future__ import annotations

import torch
from torch import nn


class LiteDigitalTwin(nn.Module):
    def __init__(self, input_dim: int = 1524):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
