from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)), "R2": float(r2_score(y_true, y_pred)), "Pearson": float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan"), "Spearman": float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")}
