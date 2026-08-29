#!/usr/bin/env python
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.chemistry.fingerprints import smiles_to_fingerprint
from src.data.demo import DEMO_DRUGS, synthetic_expression, synthetic_response
from src.data.preprocessing import preprocess_expression
from src.evaluation.metrics import regression_metrics
from src.utils.reproducibility import set_random_seed


def main() -> None:
    set_random_seed()
    expr, _ = preprocess_expression(synthetic_expression(30))
    genes = [c for c in expr.columns if c != "sample_id"]
    responses = synthetic_response(expr.sample_id.tolist(), [d.drug_id for d in DEMO_DRUGS])
    drug_map = {d.drug_id: smiles_to_fingerprint(d.smiles) for d in DEMO_DRUGS}
    rows = []
    y = []
    expr_map = expr.set_index("sample_id")[genes].to_dict("index")
    for row in responses.itertuples(index=False):
        rows.append(np.concatenate([np.fromiter(expr_map[row.sample_id].values(), dtype=float), drug_map[row.drug_id]]))
        y.append(row.response)
    X = np.vstack(rows); y = np.asarray(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    start = time.perf_counter()
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42)
    except ImportError:
        model = HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, random_state=42)
    model.fit(Xtr, ytr)
    train_time = time.perf_counter() - start
    pred = model.predict(Xte)
    metrics = regression_metrics(yte, pred)
    summary = {"baseline": type(model).__name__, "features": int(X.shape[1]), "trees": 300 if type(model).__name__ == "LGBMRegressor" else "fallback", "training_time_seconds": train_time, "metrics": metrics}
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/baseline_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
