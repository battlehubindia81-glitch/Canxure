from __future__ import annotations

import os
import random

import numpy as np

RANDOM_SEED = 42


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducible synthetic experiments.

    Purpose: synchronize Python, NumPy, and optional PyTorch randomness.
    Inputs: seed scalar integer. Input shape: ().
    Outputs: None. Output shape: not applicable.
    Data types: int.
    Exceptions: none; optional PyTorch is used only if installed.
    Assumptions: deterministic GPU kernels may still require runtime-specific settings.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
