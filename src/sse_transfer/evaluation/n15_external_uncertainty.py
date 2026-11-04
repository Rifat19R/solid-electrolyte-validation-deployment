"""N15 analysis 4 -- external uncertainty via row-paired bootstrap
(post-review sensitivity analysis, NOT part of the N5 pretest lock).

Unlike `evaluation.metrics.bootstrap_metric_ci` (which resamples
independently per call/model), every model here is scored on the SAME
resampled row-index vector within a given bootstrap iteration, so a
per-iteration model ranking is well-defined and rank-frequency/pairwise
win-rate statistics can be computed. Descriptive only -- no significance
test is performed or implied.
"""

from __future__ import annotations

from typing import Any

import numpy as np

N15_EXTERNAL_UNCERTAINTY_VERSION = "1.0"


def row_paired_bootstrap_mae_ci_and_ranking(
    residuals_by_model: dict[str, np.ndarray],
    *,
    seed: int,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Row-paired bootstrap over external residuals (`y_pred - y_true`).

    Returns per-model point MAE + percentile CI, a rank-frequency table
    (1-indexed, rank 1 = lowest/best MAE), the probability the point-estimate
    best model stays rank 1 under resampling, and a pairwise win-rate matrix
    (fraction of resamples where model A's resampled MAE < model B's).
    """
    models = sorted(residuals_by_model)
    if not models:
        raise ValueError("row_paired_bootstrap_mae_ci_and_ranking: no models given")
    n_rows = len(next(iter(residuals_by_model.values())))
    if n_rows == 0:
        raise ValueError("row_paired_bootstrap_mae_ci_and_ranking: empty residual arrays")
    residual_matrix = np.empty((len(models), n_rows), dtype=float)
    for i, m in enumerate(models):
        arr = np.asarray(residuals_by_model[m], dtype=float)
        if arr.shape != (n_rows,):
            raise ValueError(
                f"row_paired_bootstrap_mae_ci_and_ranking: residual length mismatch for {m!r}"
            )
        residual_matrix[i] = arr

    point_mae = np.abs(residual_matrix).mean(axis=1)  # (n_models,)
    best_model_idx = int(np.argmin(point_mae))

    rng = np.random.default_rng(seed)
    boot_mae = np.empty((n_boot, len(models)), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        boot_mae[b] = np.abs(residual_matrix[:, idx]).mean(axis=1)

    # rank 1 = lowest MAE; argsort ascending, then invert to get each model's rank.
    # order: (n_boot, n_models), model index at each rank.
    order = np.argsort(boot_mae, axis=1, kind="stable")
    ranks = np.empty_like(order)
    for b in range(n_boot):
        ranks[b, order[b]] = np.arange(1, len(models) + 1)

    alpha = (1.0 - ci) / 2.0
    per_model: dict[str, Any] = {}
    for i, m in enumerate(models):
        lo, hi = np.quantile(boot_mae[:, i], [alpha, 1.0 - alpha])
        rank_counts = {int(r): int(np.sum(ranks[:, i] == r)) for r in range(1, len(models) + 1)}
        per_model[m] = {
            "point_mae": float(point_mae[i]),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "ci_level": ci,
            "rank_frequency": {str(r): count / n_boot for r, count in sorted(rank_counts.items())},
            "probability_rank_1": rank_counts[1] / n_boot,
        }

    pairwise_win_rate: dict[str, dict[str, float]] = {m: {} for m in models}
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                continue
            pairwise_win_rate[m1][m2] = float(np.mean(boot_mae[:, i] < boot_mae[:, j]))

    return {
        "models": models,
        "n_rows": n_rows,
        "seed": seed,
        "n_boot": n_boot,
        "ci_level": ci,
        "best_model_point_estimate": models[best_model_idx],
        "probability_best_model_stays_rank_1": (
            per_model[models[best_model_idx]]["probability_rank_1"]
        ),
        "per_model": per_model,
        "pairwise_win_rate_descriptive_only": pairwise_win_rate,
        "note": (
            "pairwise_win_rate_descriptive_only[a][b] is the fraction of row-paired bootstrap "
            "resamples in which model a's resampled MAE was strictly lower than model b's. "
            "Descriptive only -- not a p-value, not a claim of statistical significance."
        ),
    }
