"""Composition-feature-space applicability-domain (AD) score (Phase 2.6B).

A pure, model-free nearest-neighbor distance in a standardized composition
feature space. `Family` is never used as an input here (this phase's task
brief: "Do NOT use Family label as a predictive input" for the AD score).
Kept independent of training/model code, mirroring
`evaluation.family_robustness`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist

APPLICABILITY_DOMAIN_VERSION = "1.0"


def fit_standardization(x_ref: pd.DataFrame) -> dict[str, Any]:
    """Column mean/std fit on a reference (training) feature frame only.

    A zero-variance column is standardized with scale 1.0 (its raw,
    already-degenerate values pass through unchanged) rather than dividing
    by zero.
    """
    values = x_ref.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    std_safe = np.where(std > 0, std, 1.0)
    return {"columns": list(x_ref.columns), "mean": mean, "std": std_safe}


def _standardize(x: pd.DataFrame, stats: dict[str, Any]) -> np.ndarray:
    cols = list(stats["columns"])
    if list(x.columns) != cols:
        raise ValueError("applicability_domain: feature columns do not match fitted stats")
    standardized = (x.to_numpy(dtype=float) - stats["mean"]) / stats["std"]
    return np.asarray(standardized, dtype=float)


def nearest_neighbor_ad_score(
    x_ref: pd.DataFrame, x_query: pd.DataFrame, stats: dict[str, Any] | None = None
) -> np.ndarray:
    """Euclidean distance from each `x_query` row to its nearest `x_ref` row,
    in the feature space standardized on `x_ref` alone (train-only fit; no
    validation/test statistic ever enters `stats`).

    Assumes `x_ref` and `x_query` are disjoint row sets, which holds for
    every caller in this phase (a fold's/LOMFO exclusion's training pool
    versus its own held-out rows) -- a query row is never its own nearest
    neighbor by construction, so no explicit self-exclusion is applied.
    """
    if stats is None:
        stats = fit_standardization(x_ref)
    z_ref = _standardize(x_ref, stats)
    z_query = _standardize(x_query, stats)
    dist = cdist(z_query, z_ref, metric="euclidean")
    return np.asarray(dist.min(axis=1), dtype=float)


def ad_score_error_correlation(ad_scores: np.ndarray, abs_residual: np.ndarray) -> dict[str, float]:
    ad_scores = np.asarray(ad_scores, dtype=float)
    abs_residual = np.asarray(abs_residual, dtype=float)
    if ad_scores.shape != abs_residual.shape:
        raise ValueError("ad_score_error_correlation: shape mismatch")
    if np.allclose(ad_scores, ad_scores[0]):
        return {"spearman_r": float("nan"), "p_value": float("nan"), "n": int(len(ad_scores))}
    rho, p = scipy_stats.spearmanr(ad_scores, abs_residual)
    return {"spearman_r": float(rho), "p_value": float(p), "n": int(len(ad_scores))}


def ad_score_quantile_error_table(
    ad_scores: np.ndarray, abs_residual: np.ndarray, n_bins: int = 4
) -> list[dict[str, Any]]:
    """Quantile-binned mean error by AD score (lowest-AD bin first) -- "does
    higher OOD score correlate with larger errors" as a table, not only a
    single correlation coefficient. Deterministic ascending-AD-score sort
    (stable, so ties break by original row order)."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")
    ad_scores = np.asarray(ad_scores, dtype=float)
    abs_residual = np.asarray(abs_residual, dtype=float)
    if ad_scores.shape != abs_residual.shape:
        raise ValueError("ad_score_quantile_error_table: shape mismatch")
    order = np.argsort(ad_scores, kind="stable")
    sorted_ad = ad_scores[order]
    sorted_err = abs_residual[order]
    n = len(ad_scores)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    out: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        out.append(
            {
                "bin": i,
                "n": int(hi - lo),
                "ad_score_min": float(sorted_ad[lo]),
                "ad_score_max": float(sorted_ad[hi - 1]),
                "mean_abs_residual": float(np.mean(sorted_err[lo:hi])),
            }
        )
    return out
