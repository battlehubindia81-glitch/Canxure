from __future__ import annotations

import numpy as np


def ensemble_uncertainty(predictions: list[float]) -> dict[str, float | str]:
    arr = np.asarray(predictions, dtype=float)
    std = float(arr.std())
    return {"prediction": float(arr.mean()), "uncertainty": std, "confidence_category": "LOW" if std > 0.3 else "MODERATE" if std > 0.1 else "HIGH"}
