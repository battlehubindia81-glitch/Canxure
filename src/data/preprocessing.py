from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PRIORITY_GENES = ["TP53", "EGFR", "KRAS", "BRAF", "PIK3CA", "ALK", "ROS1", "MET", "ERBB2", "RET", "NTRK1", "NTRK2", "NTRK3"]


def preprocess_expression(df: pd.DataFrame, top_n: int = 500, mode: str = "variance_plus_priority", artifact_dir: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, object]]:
    """Clean, variance-filter, and MinMax-scale expression data.

    Purpose: convert sample-by-gene expression table to fixed model input.
    Inputs: DataFrame with sample_id and genes. Input shape: (n_samples, n_genes+1).
    Outputs: processed DataFrame and metadata. Output shape: (n_samples, top_n+1).
    Data types: pandas DataFrame, dict.
    Exceptions: ValueError for missing sample_id or invalid mode.
    Assumptions: gene columns are numeric expression-like values.
    """
    if "sample_id" not in df.columns:
        raise ValueError("Expression table requires sample_id column")
    if mode not in {"strict_variance", "variance_plus_priority"}:
        raise ValueError("mode must be strict_variance or variance_plus_priority")
    clean = df.drop_duplicates(subset=["sample_id"]).copy()
    gene_cols = [c for c in clean.columns if c != "sample_id"]
    clean[gene_cols] = clean[gene_cols].apply(pd.to_numeric, errors="coerce")
    missingness = clean[gene_cols].isna().mean().to_dict()
    clean[gene_cols] = clean[gene_cols].fillna(clean[gene_cols].median(numeric_only=True)).fillna(0.0)
    variances = clean[gene_cols].var().sort_values(ascending=False)
    selected = list(variances.head(top_n).index)
    if mode == "variance_plus_priority":
        for gene in PRIORITY_GENES:
            if gene in gene_cols and gene not in selected:
                selected.append(gene)
        selected = selected[:top_n]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(clean[selected].to_numpy(dtype=float))
    out = pd.DataFrame(scaled, columns=selected)
    out.insert(0, "sample_id", clean["sample_id"].to_numpy())
    metadata = {"selected_genes": selected, "missingness": missingness, "shape": list(out.shape), "mode": mode}
    if artifact_dir:
        path = Path(artifact_dir); path.mkdir(parents=True, exist_ok=True)
        (path / "selected_genes.json").write_text(json.dumps(selected, indent=2))
        (path / "expression_preprocessing.json").write_text(json.dumps(metadata, indent=2))
    return out, metadata
