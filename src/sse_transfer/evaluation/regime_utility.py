"""N7 — does an internal validation regime (A/B/C) predict real external
transfer behavior? Pure functions over already-computed N4 (internal) and
N6 (external) results (CLAUDE.md "Keep evaluation independent from
training"): nothing here fits a model or reads a raw dataset file.

Per artifacts/frozen/external_transfer_lock.json's `predeclared_n7_protocol`, frozen
before N6 opened any Hargreaves target: four statistics per regime
(median abs MAE gap, Spearman rank agreement, Kendall tau, top-model
selection regret), each with a row-level bootstrap CI over the external
cohort. Internal (N4) numbers are never resampled -- only which external
rows are drawn varies across bootstrap iterations, matching the lock's
literal wording.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

REGIME_UTILITY_VERSION = "1.0"


def _assert_finite(label: str, values_by_model: dict[str, float]) -> None:
    """Raise rather than silently propagate a NaN/inf MAE into a rank or
    gap statistic (reviewer remediation: a latent gap an adversarial audit
    flagged -- np.allclose(vals, vals[0]) is False whenever NaN is present,
    which would bypass rank_agreement's constant-input guard and let NaN
    reach scipy's rank-correlation functions silently). No current frozen
    artifact contains a non-finite MAE (checked); this guard exists so a
    future one fails loudly instead of producing a silently-wrong number."""
    bad = {m: v for m, v in values_by_model.items() if not np.isfinite(v)}
    if bad:
        raise ValueError(f"{label}: non-finite value(s) in input: {bad}")


def median_abs_mae_gap(
    internal_by_model: dict[str, float], external_by_model: dict[str, float]
) -> float:
    """Median over models of |internal pooled MAE - external MAE|."""
    models = sorted(internal_by_model)
    if models != sorted(external_by_model):
        raise ValueError("median_abs_mae_gap: internal and external must cover the same model set")
    _assert_finite("median_abs_mae_gap.internal", internal_by_model)
    _assert_finite("median_abs_mae_gap.external", external_by_model)
    gaps = [abs(internal_by_model[m] - external_by_model[m]) for m in models]
    return float(np.median(gaps))


def rank_agreement(
    internal_by_model: dict[str, float], external_by_model: dict[str, float]
) -> dict[str, float]:
    """Spearman rho and Kendall tau between the internal-MAE ranking and the
    external-MAE ranking (both ascending = better), over the same model set.
    A model absent from a ranking (e.g. undefined LOMFO coverage) must be
    excluded from BOTH arrays by the caller before calling this function --
    it raises rather than silently dropping mismatched models."""
    models = sorted(internal_by_model)
    if models != sorted(external_by_model):
        raise ValueError("rank_agreement: internal and external must cover the same model set")
    _assert_finite("rank_agreement.internal", internal_by_model)
    _assert_finite("rank_agreement.external", external_by_model)
    internal_vals = np.array([internal_by_model[m] for m in models])
    external_vals = np.array([external_by_model[m] for m in models])
    if np.allclose(internal_vals, internal_vals[0]) or np.allclose(external_vals, external_vals[0]):
        return {"spearman_rho": float("nan"), "kendall_tau": float("nan"), "n_models": len(models)}
    rho, _ = scipy_stats.spearmanr(internal_vals, external_vals)
    tau, _ = scipy_stats.kendalltau(internal_vals, external_vals)
    return {"spearman_rho": float(rho), "kendall_tau": float(tau), "n_models": len(models)}


def top_model_selection_regret(
    internal_by_model: dict[str, float], external_by_model: dict[str, float]
) -> float:
    """external_MAE(regime's argmin-internal-MAE model) minus
    external_MAE(true argmin-external-MAE model). Zero iff the regime's #1
    pick is also the truly best external model; always >= 0."""
    models = sorted(internal_by_model)
    if models != sorted(external_by_model):
        raise ValueError(
            "top_model_selection_regret: internal and external must cover the same model set"
        )
    _assert_finite("top_model_selection_regret.internal", internal_by_model)
    _assert_finite("top_model_selection_regret.external", external_by_model)
    picked = min(models, key=lambda m: internal_by_model[m])
    true_best = min(models, key=lambda m: external_by_model[m])
    regret = external_by_model[picked] - external_by_model[true_best]
    if regret < -1e-9:
        raise RuntimeError(
            "top_model_selection_regret: negative regret is impossible by definition"
        )
    return float(max(regret, 0.0))


def regime_summary(
    internal_by_model: dict[str, float], external_by_model: dict[str, float]
) -> dict[str, Any]:
    """All four N7 statistics for one regime, as a single dict."""
    agreement = rank_agreement(internal_by_model, external_by_model)
    return {
        "median_abs_mae_gap": median_abs_mae_gap(internal_by_model, external_by_model),
        "spearman_rho": agreement["spearman_rho"],
        "kendall_tau": agreement["kendall_tau"],
        "top_model_selection_regret": top_model_selection_regret(
            internal_by_model, external_by_model
        ),
        "n_models": agreement["n_models"],
    }


def macro_family_mae_excluding(
    per_family_mae: dict[str, dict[str, float]], excluded_family: str
) -> float:
    """Unweighted mean of `mae` across every family in `per_family_mae`
    except `excluded_family` (N9 sensitivity B: does the `__MISSING__`
    family's presence in a family-level average change the picture?).
    Raises if `excluded_family` is the only family present (nothing left to
    average) rather than silently returning NaN."""
    remaining = {k: v for k, v in per_family_mae.items() if k != excluded_family}
    if not remaining:
        raise ValueError(
            f"macro_family_mae_excluding: no families remain after excluding {excluded_family!r}"
        )
    return float(np.mean([v["mae"] for v in remaining.values()]))


def bootstrap_regime_pair_delta_ci(
    internal_by_model_a: dict[str, float],
    internal_by_model_b: dict[str, float],
    external_residuals_by_model: dict[str, np.ndarray],
    *,
    statistic: str,
    seed: int,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap CI for statistic(regime B) - statistic(regime A), resampling
    external ROWS with replacement each iteration (internal/N4 numbers are
    never resampled -- they are a fixed reference per the lock). `statistic`
    is one of "median_abs_mae_gap", "spearman_rho", "kendall_tau",
    "top_model_selection_regret". `external_residuals_by_model[model]` is a
    1-D array of that model's (y_pred - y_true) residuals on the strict
    external cohort, all arrays the same length and row-aligned."""
    if statistic not in (
        "median_abs_mae_gap",
        "spearman_rho",
        "kendall_tau",
        "top_model_selection_regret",
    ):
        raise ValueError(f"Unknown statistic {statistic!r}")
    models = sorted(internal_by_model_a)
    if models != sorted(internal_by_model_b) or models != sorted(external_residuals_by_model):
        raise ValueError("bootstrap_regime_pair_delta_ci: model sets must match across all inputs")
    n_rows = len(next(iter(external_residuals_by_model.values())))
    for m, arr in external_residuals_by_model.items():
        if len(arr) != n_rows:
            raise ValueError(f"bootstrap_regime_pair_delta_ci: residual length mismatch for {m!r}")

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n_rows, size=n_rows)
        ext_mae = {
            m: float(np.mean(np.abs(arr[idx]))) for m, arr in external_residuals_by_model.items()
        }
        if statistic == "median_abs_mae_gap":
            val_a = median_abs_mae_gap(internal_by_model_a, ext_mae)
            val_b = median_abs_mae_gap(internal_by_model_b, ext_mae)
        elif statistic == "top_model_selection_regret":
            val_a = top_model_selection_regret(internal_by_model_a, ext_mae)
            val_b = top_model_selection_regret(internal_by_model_b, ext_mae)
        else:
            key = "spearman_rho" if statistic == "spearman_rho" else "kendall_tau"
            val_a = rank_agreement(internal_by_model_a, ext_mae)[key]
            val_b = rank_agreement(internal_by_model_b, ext_mae)[key]
        deltas[i] = val_b - val_a

    finite = deltas[np.isfinite(deltas)]
    if len(finite) == 0:
        return {
            "statistic": statistic,
            "point_delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_level": ci,
            "n_boot": n_boot,
            "n_finite": 0,
            "seed": seed,
        }
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(finite, [alpha, 1.0 - alpha])
    point_a = None
    point_b = None
    ext_mae_point = {
        m: float(np.mean(np.abs(arr))) for m, arr in external_residuals_by_model.items()
    }
    if statistic == "median_abs_mae_gap":
        point_a = median_abs_mae_gap(internal_by_model_a, ext_mae_point)
        point_b = median_abs_mae_gap(internal_by_model_b, ext_mae_point)
    elif statistic == "top_model_selection_regret":
        point_a = top_model_selection_regret(internal_by_model_a, ext_mae_point)
        point_b = top_model_selection_regret(internal_by_model_b, ext_mae_point)
    else:
        key = "spearman_rho" if statistic == "spearman_rho" else "kendall_tau"
        point_a = rank_agreement(internal_by_model_a, ext_mae_point)[key]
        point_b = rank_agreement(internal_by_model_b, ext_mae_point)[key]
    return {
        "statistic": statistic,
        "point_delta": float(point_b - point_a),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "n_boot": n_boot,
        "n_finite": int(len(finite)),
        "seed": seed,
    }


def cluster_bootstrap_regime_pair_delta_ci(
    internal_by_model_a: dict[str, float],
    internal_by_model_b: dict[str, float],
    external_residuals_by_model: dict[str, np.ndarray],
    row_cluster_id: np.ndarray,
    *,
    statistic: str,
    seed: int,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Reviewer remediation Issue 06: source/DOI-cluster bootstrap variant of
    `bootstrap_regime_pair_delta_ci`. Identical statistic logic and identical
    fixed internal (never-resampled) reference; the only difference is the
    resampling UNIT -- clusters of external rows (e.g. rows/formulas sharing
    a literature-source DOI, possibly transitively) are resampled with
    replacement instead of individual rows, so rows within a resampled
    cluster always appear or are absent together. `row_cluster_id[i]` is the
    cluster label for `external_residuals_by_model[model][i]`, identical
    length and row order for every model. This is a separate function, not a
    modification of `bootstrap_regime_pair_delta_ci` -- the row-level
    function and every existing caller (N7b's primary analysis) are
    unchanged."""
    if statistic not in (
        "median_abs_mae_gap",
        "spearman_rho",
        "kendall_tau",
        "top_model_selection_regret",
    ):
        raise ValueError(f"Unknown statistic {statistic!r}")
    models = sorted(internal_by_model_a)
    if models != sorted(internal_by_model_b) or models != sorted(external_residuals_by_model):
        raise ValueError(
            "cluster_bootstrap_regime_pair_delta_ci: model sets must match across all inputs"
        )
    n_rows = len(next(iter(external_residuals_by_model.values())))
    for m, arr in external_residuals_by_model.items():
        if len(arr) != n_rows:
            raise ValueError(
                f"cluster_bootstrap_regime_pair_delta_ci: residual length mismatch for {m!r}"
            )
    if len(row_cluster_id) != n_rows:
        raise ValueError("cluster_bootstrap_regime_pair_delta_ci: row_cluster_id length mismatch")

    unique_clusters = np.unique(row_cluster_id)
    cluster_row_idx = {c: np.where(row_cluster_id == c)[0] for c in unique_clusters}
    n_clusters = len(unique_clusters)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_row_idx[c] for c in sampled_clusters])
        ext_mae = {
            m: float(np.mean(np.abs(arr[idx]))) for m, arr in external_residuals_by_model.items()
        }
        if statistic == "median_abs_mae_gap":
            val_a = median_abs_mae_gap(internal_by_model_a, ext_mae)
            val_b = median_abs_mae_gap(internal_by_model_b, ext_mae)
        elif statistic == "top_model_selection_regret":
            val_a = top_model_selection_regret(internal_by_model_a, ext_mae)
            val_b = top_model_selection_regret(internal_by_model_b, ext_mae)
        else:
            key = "spearman_rho" if statistic == "spearman_rho" else "kendall_tau"
            val_a = rank_agreement(internal_by_model_a, ext_mae)[key]
            val_b = rank_agreement(internal_by_model_b, ext_mae)[key]
        deltas[i] = val_b - val_a

    finite = deltas[np.isfinite(deltas)]
    ext_mae_point = {
        m: float(np.mean(np.abs(arr))) for m, arr in external_residuals_by_model.items()
    }
    if statistic == "median_abs_mae_gap":
        point_a = median_abs_mae_gap(internal_by_model_a, ext_mae_point)
        point_b = median_abs_mae_gap(internal_by_model_b, ext_mae_point)
    elif statistic == "top_model_selection_regret":
        point_a = top_model_selection_regret(internal_by_model_a, ext_mae_point)
        point_b = top_model_selection_regret(internal_by_model_b, ext_mae_point)
    else:
        key = "spearman_rho" if statistic == "spearman_rho" else "kendall_tau"
        point_a = rank_agreement(internal_by_model_a, ext_mae_point)[key]
        point_b = rank_agreement(internal_by_model_b, ext_mae_point)[key]

    if len(finite) == 0:
        return {
            "statistic": statistic,
            "point_delta": float(point_b - point_a),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_level": ci,
            "n_boot": n_boot,
            "n_finite": 0,
            "n_clusters": n_clusters,
            "seed": seed,
        }
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(finite, [alpha, 1.0 - alpha])
    return {
        "statistic": statistic,
        "point_delta": float(point_b - point_a),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "n_boot": n_boot,
        "n_finite": int(len(finite)),
        "n_clusters": n_clusters,
        "seed": seed,
    }


def formula_cluster_ids(formulas: Sequence[str]) -> np.ndarray:
    """Reviewer remediation Issue 10: map each reduced-formula string to a
    dense integer cluster id, grouping exact-duplicate formulas together
    (first-appearance order, so the mapping is deterministic for a fixed
    input order). Feeding the result into
    `cluster_bootstrap_regime_pair_delta_ci` as `row_cluster_id` implements a
    formula-cluster bootstrap: formulas with more than one member row are
    resampled as a unit. When every formula is unique this degenerates to
    one row per cluster, i.e. plain row-level resampling with a different
    (but distributionally equivalent) draw sequence."""
    index: dict[str, int] = {}
    ids = np.empty(len(formulas), dtype=int)
    for i, f in enumerate(formulas):
        if f not in index:
            index[f] = len(index)
        ids[i] = index[f]
    return ids


def formula_balanced_mae(formulas: Sequence[str], abs_residuals: np.ndarray) -> float:
    """Reviewer remediation Issue 10: equal-weight-per-unique-formula MAE,
    MAE_formula = mean_f( mean_{i in f} |residual_i| ), giving every unique
    reduced formula equal total influence regardless of how many measurement
    rows it contributes. Implemented as an explicit group-then-average (not
    a shortcut assuming any particular group size), so it is a genuine test
    of the grouping logic rather than a relabeled row mean; it equals the
    ordinary row-weighted mean |residual| exactly when every formula is
    unique (each group has exactly one member)."""
    formulas = list(formulas)
    abs_residuals = np.asarray(abs_residuals, dtype=float)
    if len(formulas) != len(abs_residuals):
        raise ValueError("formula_balanced_mae: formulas/abs_residuals length mismatch")
    if len(formulas) == 0:
        raise ValueError("formula_balanced_mae: empty input")
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for f, r in zip(formulas, abs_residuals, strict=True):
        sums[f] = sums.get(f, 0.0) + float(r)
        counts[f] = counts.get(f, 0) + 1
    per_formula_mean = [sums[f] / counts[f] for f in sums]
    return float(np.mean(per_formula_mean))
