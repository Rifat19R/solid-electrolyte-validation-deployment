"""Regression metrics and bootstrap confidence intervals for Phase 1.

Kept independent of training/model code (CLAUDE.md "Keep evaluation
independent from training"). EXPERIMENT_PROTOCOL.md "Primary endpoint" /
"Secondary metrics" / "Seeds and statistics".
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

METRIC_KEYS: tuple[str, ...] = ("mae", "rmse", "r2", "spearman_rho", "median_absolute_error")


def _is_constant(x: np.ndarray) -> bool:
    """True if every entry of `x` is identical (Spearman rank correlation
    is mathematically undefined for a constant vector: with zero rank
    variance, `scipy.stats.spearmanr` would divide by zero and emit
    `ConstantInputWarning` while returning NaN -- this short-circuits to
    the same NaN result without triggering the warning or the division)."""
    return bool(x.size > 0 and np.all(x == x[0]))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} vs {y_pred.shape}"
        )
    if y_true.ndim != 1:
        raise ValueError("regression_metrics expects 1-D arrays")

    n = int(len(y_true))
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    medae = float(np.median(np.abs(errors)))
    r2 = float(r2_score(y_true, y_pred)) if n > 1 else float("nan")
    spearman_defined = n > 1 and not _is_constant(y_true) and not _is_constant(y_pred)
    spearman_rho = (
        float(stats.spearmanr(y_true, y_pred).statistic) if spearman_defined else float("nan")
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman_rho": spearman_rho,
        "median_absolute_error": medae,
        "n": n,
    }


def relative_improvement_pct(baseline_mae: float, candidate_mae: float) -> float:
    """Percent MAE improvement of `candidate` over `baseline`: positive means
    candidate is better (lower MAE). Callers must pass full-precision metric
    values (e.g. the unrounded `mae_mean` stored in a results JSON), never a
    value already rounded for display -- rounding before this division
    silently changes a reported GO/NO-GO percentage.
    """
    if baseline_mae == 0:
        return float("nan")
    return 100.0 * (baseline_mae - candidate_mae) / baseline_mae


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_key: str,
    *,
    seed: int,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> dict[str, float]:
    """Bootstrap CI for one metric via row-level resampling with replacement."""
    if metric_key not in METRIC_KEYS:
        raise ValueError(f"Unknown metric_key {metric_key!r}; expected one of {METRIC_KEYS}")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        values[i] = regression_metrics(y_true[idx], y_pred[idx])[metric_key]

    # A resample can be a constant (or n<=1) subsample -- e.g. drawing only
    # duplicate-labeled rows -- for which `spearman_rho`/`r2` are
    # mathematically undefined NaN (see `regression_metrics`). `np.quantile`
    # does not skip NaN, so an unfiltered resample can silently corrupt the
    # reported CI bound; filter to finite draws first, matching the
    # established pattern in `regime_utility.bootstrap_regime_pair_delta_ci`.
    finite = values[np.isfinite(values)]
    alpha = (1.0 - ci) / 2.0
    if len(finite) == 0:
        lo = hi = float("nan")
    else:
        lo, hi = np.quantile(finite, [alpha, 1.0 - alpha])
    return {
        "point": regression_metrics(y_true, y_pred)[metric_key],
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "n_boot": n_boot,
        "n_finite": int(len(finite)),
        "seed": seed,
    }
