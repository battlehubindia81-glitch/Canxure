from __future__ import annotations

import numpy as np


def gradient_feature_attribution(features: np.ndarray, gradients: np.ndarray, names: list[str], top_k: int = 10) -> list[dict[str, float | str]]:
    scores = np.abs(features * gradients)
    idx = np.argsort(scores)[-top_k:][::-1]
    return [{"feature": names[i], "attribution": float(scores[i])} for i in idx]
