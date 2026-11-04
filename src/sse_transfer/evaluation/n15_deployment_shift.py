"""N15 analysis 2 -- deployment-shift alignment (post-review sensitivity
analysis, NOT part of the N5 pretest lock).

Pure summary/comparison helpers over already-computed nearest-neighbor
composition-feature distances (`uncertainty.applicability_domain.
fit_standardization` / `nearest_neighbor_ad_score`, reused unmodified by the
N15 driver script for every regime's actual distance computation). This
module never reads a target value `y`/`log_target`; it operates on distance
arrays and family-label arrays only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

N15_DEPLOYMENT_SHIFT_VERSION = "1.0"


def distance_distribution_summary(distances: np.ndarray) -> dict[str, float]:
    """n, median, IQR (q25/q75), and q90 of a 1-D distance array."""
    arr = np.asarray(distances, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("distance_distribution_summary: expected a non-empty 1-D array")
    q25, median, q75, q90 = np.quantile(arr, [0.25, 0.5, 0.75, 0.90])
    return {
        "n": int(arr.size),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "q90": float(q90),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def wasserstein_distance_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Thin, documented wrapper around `scipy.stats.wasserstein_distance`
    (1-D earth-mover's distance) -- a single import point for this metric so
    every N15 cross-regime comparison uses the identical function/version."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return float(scipy_stats.wasserstein_distance(a_arr, b_arr))


def family_novelty_fraction(
    held_out_family: pd.Series | np.ndarray, train_family: pd.Series | np.ndarray
) -> dict[str, Any]:
    """Fraction of held-out rows whose family label does not appear anywhere
    in the paired training partition -- purely descriptive, using only the
    existing deterministic `family_filled` label already produced by
    `evaluation.internal_validation.canonicalize_and_fill_family`. No new
    family grouping/labeling is introduced here."""
    held = pd.Series(held_out_family).astype(str).to_numpy()
    train_families = set(pd.Series(train_family).astype(str).unique().tolist())
    if held.size == 0:
        raise ValueError("family_novelty_fraction: empty held_out_family")
    novel_mask = np.array([fam not in train_families for fam in held])
    return {
        "n": int(held.size),
        "n_novel": int(novel_mask.sum()),
        "fraction_novel": float(novel_mask.mean()),
    }
