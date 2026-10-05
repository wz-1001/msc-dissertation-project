"""
column_analysis.py

Target-leakage detection: scores how strongly every feature column is
statistically associated with the target. Cramer's V for classification
(categorical), Pearson |r| for regression (continuous).
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr

DEFAULT_ASSOCIATION_THRESHOLD = 0.95


def _cramers_v(feature: pd.Series, target: pd.Series) -> float:
    valid = feature.notna() & target.notna()
    feature, target = feature[valid], target[valid]
    if feature.nunique() < 2 or len(feature) < 10:
        return 0.0

    table = pd.crosstab(feature, target)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0

    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    denom = n * (min(r, k) - 1)

    return float((chi2 / denom) ** 0.5) if denom > 0 else 0.0


def compute_column_associations(df: pd.DataFrame, target_col: str,
                                 task_type: str = "classification",
                                 threshold: float = DEFAULT_ASSOCIATION_THRESHOLD,
                                 max_unique_ratio: float = 0.5,
                                 max_bins: int = 15) -> list:
    target = df[target_col]
    n = len(df)
    scores = []

    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]

        # skip anything that's basically an ID column
        if series.nunique() / n > max_unique_ratio:
            continue

        valid = series.notna() & target.notna()
        f_clean, t_clean = series[valid], target[valid]

        if len(f_clean) < 10:
            continue

        score = 0.0
        metric_name = "Cramer's V"

        if task_type == "regression" and pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(target):
            r_val, _ = pearsonr(f_clean, t_clean)
            score = float(abs(r_val)) if not np.isnan(r_val) else 0.0
            metric_name = "Pearson |r|"

        else:
            # cramer's v needs categories, not raw numbers - qcut buckets a continuous column into bins first
            if pd.api.types.is_numeric_dtype(t_clean) and t_clean.nunique() > max_bins:
                try:
                    t_test = pd.qcut(t_clean, q=max_bins, duplicates="drop")
                except ValueError:
                    t_test = t_clean
            else:
                t_test = t_clean

            if pd.api.types.is_numeric_dtype(f_clean) and f_clean.nunique() > max_bins:
                try:
                    f_test = pd.qcut(f_clean, q=max_bins, duplicates="drop")
                except ValueError:
                    f_test = f_clean
            else:
                f_test = f_clean

            score = _cramers_v(f_test, t_test)

        scores.append({
            "column": str(col),
            "association": round(float(score), 4),
            "metric": metric_name,
            "suggested_drop": bool(score >= threshold),
        })

    scores.sort(key=lambda r: r["association"], reverse=True)  # highest-risk columns first
    return scores