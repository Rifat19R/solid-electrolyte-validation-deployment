"""Family-level (group) robustness metrics for OOD diagnostics (Phase 2.2).

Kept independent of training/model code (CLAUDE.md "Keep evaluation
independent from training"), mirroring `evaluation.metrics`. Every function
here operates on already-computed predictions/errors and a group label
(canonical `Family`) -- it never fits a model or touches raw data.

Motivation: `evaluation.metrics.regression_metrics` reports a single pooled
(sample-weighted) MAE, which can hide the failure mode Phase 2.0V/2.1
diagnosed -- a model that looks good on average because it fits a few large
families well, while doing badly on most of the many small ones. These
functions give every family equal say (macro-family) rather than every row
equal say (sample-weighted/pooled), so "improvement" cannot be manufactured
by fitting the largest family better.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from sse_transfer.evaluation.metrics import relative_improvement_pct

FAMILY_ROBUSTNESS_METRICS_VERSION = "1.2"

WEIGHT_SCHEMES: tuple[str, ...] = ("uniform", "inverse_family_size", "sqrt_inverse_family_size")

#: Fixed-point iteration count for `family_sample_weights_by_alpha`'s `cap`
#: option (clip-then-renormalize does not commute, so a single pass can
#: violate the bound; this many iterations converges well within float
#: precision for any realistic family-count distribution).
CAP_FIXED_POINT_ITERS = 50


def family_sample_weights(families: pd.Series, scheme: str) -> np.ndarray:
    """Per-row training weight from a group-size-aware scheme.

    - "uniform": every row weight 1.0 (the implicit scheme behind an
      ordinary/"sample-weighted" pooled MAE).
    - "inverse_family_size": weight `1 / n_family`, so each family
      contributes the same *total* weight regardless of its row count.
    - "sqrt_inverse_family_size": weight `1 / sqrt(n_family)`, a softer
      correction between uniform and full inverse-size weighting.

    In both non-uniform schemes, weights are rescaled so the mean weight
    over the input rows is exactly 1.0 (keeps regularization/learning-rate
    hyperparameters comparable in scale across schemes and against the
    unweighted baseline). Missing-family rows are treated as their own
    group (`fillna` is the caller's responsibility -- this function raises
    on unlabeled entries rather than silently grouping them together
    unless the caller has already filled them).
    """
    if scheme not in WEIGHT_SCHEMES:
        raise ValueError(f"Unknown weight scheme {scheme!r}; expected one of {WEIGHT_SCHEMES}")
    if families.isna().any():
        raise ValueError(
            "family_sample_weights received unlabeled (NaN) family values; "
            "callers must fill missing-family rows to an explicit sentinel first."
        )

    counts = families.value_counts()
    n_family = families.map(counts).to_numpy(dtype=float)

    if scheme == "uniform":
        weights = np.ones(len(families), dtype=float)
    elif scheme == "inverse_family_size":
        weights = 1.0 / n_family
    else:  # sqrt_inverse_family_size
        weights = 1.0 / np.sqrt(n_family)

    mean_weight = weights.mean()
    if mean_weight <= 0 or not math.isfinite(mean_weight):
        raise RuntimeError(f"family_sample_weights: degenerate mean weight {mean_weight}")
    return weights / mean_weight


def family_sample_weights_by_alpha(
    families: pd.Series, alpha: float, *, cap: float | None = None
) -> np.ndarray:
    """Per-row weight `w_i \\propto 1 / n_family_i^alpha` (Phase 2.3 weight-
    strength sweep), continuously interpolating between uniform weighting
    (`alpha=0`) and full inverse-family-size weighting (`alpha=1`).

    Generalizes `family_sample_weights`: at `alpha=0.0`, `alpha=0.5`, and
    `alpha=1.0` this reproduces (bit-for-bit) `family_sample_weights(...,
    "uniform" | "sqrt_inverse_family_size" | "inverse_family_size")`
    respectively -- both are kept (rather than rewriting the Phase-2.2
    scheme-name function on top of this one) so Phase 2.2's frozen results
    stay reproducible from unchanged code.

    Weights are computed from `families` alone, so calling this on a single
    fold's/candidate's training partition uses only that partition's family
    counts (fold-only weighting; a global count would leak validation/test
    family sizes into training weights).

    `cap`, if given, must be `> 1.0`: weights are bounded to `[1/cap, cap]`
    while keeping the mean at 1 via fixed-point clip-then-renormalize
    iteration (a single clip-then-renormalize pass can push the clipped
    extreme back outside `[1/cap, cap]`, since renormalizing shifts the
    mean) -- so no row's weight ends up more than `cap` times the mean
    purely from the family-size term (Phase 2.3 Part B weight-cap
    sensitivity). `cap=None` (the default) applies no clipping.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0; got {alpha}")
    if cap is not None and cap <= 1.0:
        raise ValueError(
            f"cap must be > 1.0 (it bounds weight relative to the mean of 1.0); got {cap}"
        )
    if families.isna().any():
        raise ValueError(
            "family_sample_weights_by_alpha received unlabeled (NaN) family values; "
            "callers must fill missing-family rows to an explicit sentinel first."
        )

    counts = families.value_counts()
    n_family = families.map(counts).to_numpy(dtype=float)
    raw = 1.0 / np.power(n_family, alpha) if alpha > 0 else np.ones(len(families), dtype=float)

    mean_raw = raw.mean()
    if mean_raw <= 0 or not math.isfinite(mean_raw):
        raise RuntimeError(f"family_sample_weights_by_alpha: degenerate mean weight {mean_raw}")
    weights = raw / mean_raw

    if cap is not None:
        for _ in range(CAP_FIXED_POINT_ITERS):
            weights = np.clip(weights, 1.0 / cap, cap)
            mean_capped = weights.mean()
            if mean_capped <= 0 or not math.isfinite(mean_capped):
                raise RuntimeError(
                    "family_sample_weights_by_alpha: degenerate mean weight after cap "
                    f"{mean_capped}"
                )
            weights = weights / mean_capped

    return weights


def per_family_mae(
    y_true: np.ndarray, y_pred: np.ndarray, families: pd.Series | np.ndarray
) -> dict[str, dict[str, float]]:
    """`{family: {"mae": ..., "n": ...}}` for every distinct family value."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} vs {y_pred.shape}"
        )
    fam = pd.Series(families).astype(str).to_numpy()
    if len(fam) != len(y_true):
        raise ValueError("families must have the same length as y_true/y_pred")

    abs_err = np.abs(y_pred - y_true)
    out: dict[str, dict[str, float]] = {}
    for f in sorted(set(fam)):
        mask = fam == f
        out[f] = {"mae": float(abs_err[mask].mean()), "n": int(mask.sum())}
    return out


def sample_weighted_mae(per_family: dict[str, dict[str, float]]) -> float:
    """Pooled (micro-average, every row equal weight) MAE recovered from a
    per-family breakdown -- equals `regression_metrics(...)["mae"]` computed
    on the same rows, provided as a companion figure to the macro-family
    metrics below so both live in one report table."""
    total_n = sum(v["n"] for v in per_family.values())
    if total_n == 0:
        raise ValueError("sample_weighted_mae: empty per_family breakdown")
    return sum(v["mae"] * v["n"] for v in per_family.values()) / total_n


def macro_family_mae(per_family: dict[str, dict[str, float]]) -> float:
    """Every family weighted equally, regardless of row count."""
    if not per_family:
        raise ValueError("macro_family_mae: empty per_family breakdown")
    return float(np.mean([v["mae"] for v in per_family.values()]))


def median_family_mae(per_family: dict[str, dict[str, float]]) -> float:
    if not per_family:
        raise ValueError("median_family_mae: empty per_family breakdown")
    return float(np.median([v["mae"] for v in per_family.values()]))


def worst_quartile_family_mae(
    per_family: dict[str, dict[str, float]], quartile_fraction: float = 0.25
) -> dict[str, Any]:
    """Mean MAE of the worst-performing quartile of *families* (each family
    counted once, not once per row) -- the families driving tail risk under
    family-OOD deployment. At least one family is always included."""
    if not (0.0 < quartile_fraction < 1.0):
        raise ValueError(f"quartile_fraction must be in (0, 1); got {quartile_fraction}")
    if not per_family:
        raise ValueError("worst_quartile_family_mae: empty per_family breakdown")

    ranked = sorted(per_family.items(), key=lambda kv: kv[1]["mae"], reverse=True)
    k = max(1, math.ceil(len(ranked) * quartile_fraction))
    worst = ranked[:k]
    return {
        "n_families_in_worst_quartile": k,
        "quartile_fraction": quartile_fraction,
        "families": [f for f, _ in worst],
        "mae": float(np.mean([v["mae"] for _, v in worst])),
    }


def group_bootstrap_ci(
    per_family: dict[str, dict[str, float]],
    *,
    seed: int,
    n_boot: int = 2000,
    ci: float = 0.95,
    statistic: str = "macro",
) -> dict[str, Any]:
    """Bootstrap CI for a family-level statistic by resampling *families*
    (with replacement), not rows -- captures uncertainty from having only a
    handful of families, which a row-level bootstrap (`bootstrap_metric_ci`)
    cannot see because it never changes which families are represented.

    `statistic`: "macro" (mean over resampled families' MAE, unweighted) or
    "median" (median over resampled families' MAE).
    """
    if statistic not in ("macro", "median"):
        raise ValueError(f"Unknown statistic {statistic!r}; expected 'macro' or 'median'")
    families = sorted(per_family)
    n = len(families)
    if n == 0:
        raise ValueError("group_bootstrap_ci: empty per_family breakdown")
    maes = np.array([per_family[f]["mae"] for f in families], dtype=float)

    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = maes[idx]
        values[i] = float(np.mean(sample)) if statistic == "macro" else float(np.median(sample))

    point = float(np.mean(maes)) if statistic == "macro" else float(np.median(maes))
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "n_boot": n_boot,
        "n_families": n,
        "statistic": statistic,
        "seed": seed,
    }


def paired_group_bootstrap_delta_ci(
    per_family_baseline: dict[str, dict[str, float]],
    per_family_candidate: dict[str, dict[str, float]],
    *,
    seed: int,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap CI for `macro_family_mae(candidate) - macro_family_mae(baseline)`,
    resampling the *same* family indices for both series on each draw
    (paired resampling) so the CI reflects consistency of the improvement
    across families rather than treating the two models' family-MAE lists as
    independent. Requires identical family sets (same outer folds/seeds).
    """
    families = sorted(per_family_baseline)
    if families != sorted(per_family_candidate):
        raise ValueError(
            "paired_group_bootstrap_delta_ci: baseline and candidate must cover "
            "the exact same set of families (identical folds/seeds required)."
        )
    n = len(families)
    base_maes = np.array([per_family_baseline[f]["mae"] for f in families], dtype=float)
    cand_maes = np.array([per_family_candidate[f]["mae"] for f in families], dtype=float)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = float(np.mean(cand_maes[idx]) - np.mean(base_maes[idx]))

    point = float(np.mean(cand_maes) - np.mean(base_maes))
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(deltas, [alpha, 1.0 - alpha])
    return {
        "point_delta_macro_family_mae": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": ci,
        "n_boot": n_boot,
        "n_families": n,
        "reproducible_improvement": bool(hi < 0.0),
        "seed": seed,
    }


def evaluate_family_weighting_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    major_families: list[str],
    *,
    macro_improvement_min_pct: float = 5.0,
    pooled_worsen_abs_tolerance: float = 0.05,
    lomfo_max_worsened_major_families: int = 1,
    bootstrap_seed: int,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Phase 2.2/2.3 family-weighting GO/NO-GO gate, as a pure function of two
    already-computed result dicts (never fits a model) so it is unit-testable
    with synthetic numbers alone.

    `baseline` and `candidate` must each provide: `macro_family_mae` (float),
    `worst_quartile_family_mae` (float), `sample_weighted_mae` (float),
    `per_family_mae` (`{family: {"mae": ..., "n": ...}}`, identical family
    sets required), and `leave_one_major_family_out`
    (`{family: {"mae_mean": ...}}` covering every entry of `major_families`).

    GO iff, vs `baseline`: (a) macro-family MAE improves by
    `>= macro_improvement_min_pct` percent, AND (b) worst-quartile-family MAE
    does not worsen, AND (c) sample-weighted (pooled) MAE does not worsen by
    more than `pooled_worsen_abs_tolerance` absolute units, AND (d) a paired
    group-bootstrap CI of the macro-family-MAE delta is entirely negative
    (reproducible across which families are resampled), AND (e) no more than
    `lomfo_max_worsened_major_families` of `major_families` worsens (LOMFO
    MAE) by more than the same pooled tolerance. This tolerance/threshold set
    is frozen at the Phase 2.2 values by default; CLAUDE.md "Do NOT weaken
    criterion 5 post hoc" -- pass different values only as a pre-registered,
    documented design decision, never after inspecting results.
    """
    macro_improvement = relative_improvement_pct(
        baseline["macro_family_mae"], candidate["macro_family_mae"]
    )
    worst_q_improvement = relative_improvement_pct(
        baseline["worst_quartile_family_mae"], candidate["worst_quartile_family_mae"]
    )
    pooled_delta = candidate["sample_weighted_mae"] - baseline["sample_weighted_mae"]
    paired_ci = paired_group_bootstrap_delta_ci(
        baseline["per_family_mae"], candidate["per_family_mae"], seed=bootstrap_seed, n_boot=n_boot
    )
    n_worsened = sum(
        1
        for fam in major_families
        if candidate["leave_one_major_family_out"][fam]["mae_mean"]
        - baseline["leave_one_major_family_out"][fam]["mae_mean"]
        > pooled_worsen_abs_tolerance
    )
    criteria = {
        "macro_family_mae_improvement_ge_min_pct": macro_improvement >= macro_improvement_min_pct,
        "worst_quartile_family_mae_not_worse": worst_q_improvement >= 0.0,
        "pooled_mae_not_worse_by_gt_tolerance": pooled_delta <= pooled_worsen_abs_tolerance,
        "paired_bootstrap_reproducible_improvement": bool(paired_ci["reproducible_improvement"]),
        "lomfo_majority_not_worsened": n_worsened <= lomfo_max_worsened_major_families,
    }
    return {
        "macro_family_mae_relative_improvement_pct": macro_improvement,
        "worst_quartile_family_mae_relative_improvement_pct": worst_q_improvement,
        "sample_weighted_mae_delta_abs": pooled_delta,
        "paired_group_bootstrap_macro_family_mae_delta_ci": paired_ci,
        "n_major_families_worsened_by_gt_tolerance": n_worsened,
        "n_major_families_total": len(major_families),
        "criteria": criteria,
        "verdict": "GO" if all(criteria.values()) else "NO-GO",
    }


def max_single_family_improvement_share(
    per_family_baseline: dict[str, dict[str, float]],
    per_family_candidate: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """How concentrated a candidate's improvement over `baseline` is in a
    single family (Phase 2.6A "not one-family-driven" criterion).

    `delta_f = baseline_mae_f - candidate_mae_f` (positive = family `f`
    improved). `max_share` is the largest single family's `delta_f` divided
    by the sum of all *positive* deltas -- i.e. what fraction of the total
    improvement (summed only over families that improved) is attributable to
    the single best-improving family. Requires identical family sets (same
    outer folds/seeds), matching `paired_group_bootstrap_delta_ci`.
    """
    families = sorted(per_family_baseline)
    if families != sorted(per_family_candidate):
        raise ValueError(
            "max_single_family_improvement_share: baseline and candidate must "
            "cover the exact same set of families (identical folds/seeds required)."
        )
    deltas = {f: per_family_baseline[f]["mae"] - per_family_candidate[f]["mae"] for f in families}
    total_positive = sum(d for d in deltas.values() if d > 0)
    if total_positive <= 0:
        return {
            "max_family": None,
            "max_share": 0.0,
            "deltas": deltas,
            "total_positive_delta": total_positive,
        }
    max_family = max(deltas, key=lambda f: deltas[f])
    return {
        "max_family": max_family,
        "max_share": deltas[max_family] / total_positive,
        "deltas": deltas,
        "total_positive_delta": total_positive,
    }


CENSORED_LEARNING_GATE_VERSION = "1.0"


def evaluate_censored_learning_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    major_families: list[str],
    fold_seed_stability: dict[str, Any],
    *,
    macro_improvement_min_pct: float = 5.0,
    pooled_worsen_abs_tolerance: float = 0.05,
    lomfo_max_worsened_major_families: int = 1,
    fold_driven_max_fraction_worse: float = 0.5,
    family_driven_max_share: float = 0.5,
    bootstrap_seed: int,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Phase 2.6A censored-learning GO/NO-GO gate (EXPERIMENT_PROTOCOL.md /
    CLAUDE.md task brief Part D), a pure function of two already-computed
    result dicts plus a fold/seed-stability summary -- never fits a model,
    so it is unit-testable with synthetic numbers alone.

    `baseline`/`candidate` have the same shape required by
    `evaluate_family_weighting_gate`. `fold_seed_stability` is a
    `{"fraction_worse_than_baseline": float, ...}` dict, e.g. from a
    per-(fold, seed) pooled-MAE comparison (Phase 2.3's `fold_seed_stability`
    helper).

    GO iff, vs `baseline`: (1) macro-family MAE improves by
    `>= macro_improvement_min_pct` percent, AND (2) worst-quartile-family MAE
    does not worsen, AND (3) sample-weighted (pooled) MAE does not worsen by
    more than `pooled_worsen_abs_tolerance` absolute units, AND (4) a paired
    group-bootstrap CI of the macro-family-MAE delta is entirely negative
    (reproducible across which families are resampled), AND (5) the
    improvement is not one-fold-or-family-driven -- no more than
    `fold_driven_max_fraction_worse` of (fold, seed) combinations are worse
    than baseline, AND no single family accounts for more than
    `family_driven_max_share` of the total positive per-family improvement,
    AND (6) no more than `lomfo_max_worsened_major_families` of
    `major_families` worsens (LOMFO MAE) by more than the same pooled
    tolerance.
    """
    macro_improvement = relative_improvement_pct(
        baseline["macro_family_mae"], candidate["macro_family_mae"]
    )
    worst_q_improvement = relative_improvement_pct(
        baseline["worst_quartile_family_mae"], candidate["worst_quartile_family_mae"]
    )
    pooled_delta = candidate["sample_weighted_mae"] - baseline["sample_weighted_mae"]
    paired_ci = paired_group_bootstrap_delta_ci(
        baseline["per_family_mae"], candidate["per_family_mae"], seed=bootstrap_seed, n_boot=n_boot
    )
    family_concentration = max_single_family_improvement_share(
        baseline["per_family_mae"], candidate["per_family_mae"]
    )
    n_worsened = sum(
        1
        for fam in major_families
        if candidate["leave_one_major_family_out"][fam]["mae_mean"]
        - baseline["leave_one_major_family_out"][fam]["mae_mean"]
        > pooled_worsen_abs_tolerance
    )
    not_one_fold_driven = (
        fold_seed_stability["fraction_worse_than_baseline"] <= fold_driven_max_fraction_worse
    )
    not_one_family_driven = family_concentration["max_share"] <= family_driven_max_share

    criteria = {
        "macro_family_mae_improvement_ge_min_pct": macro_improvement >= macro_improvement_min_pct,
        "worst_quartile_family_mae_not_worse": worst_q_improvement >= 0.0,
        "pooled_mae_not_worse_by_gt_tolerance": pooled_delta <= pooled_worsen_abs_tolerance,
        "paired_bootstrap_reproducible_improvement": bool(paired_ci["reproducible_improvement"]),
        "not_one_fold_or_family_driven": bool(not_one_fold_driven and not_one_family_driven),
        "lomfo_majority_not_worsened": n_worsened <= lomfo_max_worsened_major_families,
    }
    return {
        "macro_family_mae_relative_improvement_pct": macro_improvement,
        "worst_quartile_family_mae_relative_improvement_pct": worst_q_improvement,
        "sample_weighted_mae_delta_abs": pooled_delta,
        "paired_group_bootstrap_macro_family_mae_delta_ci": paired_ci,
        "fold_seed_stability": fold_seed_stability,
        "not_one_fold_driven": not_one_fold_driven,
        "family_concentration": family_concentration,
        "not_one_family_driven": not_one_family_driven,
        "n_major_families_worsened_by_gt_tolerance": n_worsened,
        "n_major_families_total": len(major_families),
        "criteria": criteria,
        "verdict": "GO" if all(criteria.values()) else "NO-GO",
    }


FINAL_CANDIDATE_PROMOTION_GATE_VERSION = "1.0"


def evaluate_final_candidate_promotion_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    major_families: list[str],
    fold_seed_stability: dict[str, Any],
    *,
    uncertainty_and_ad_reproducible: bool,
    macro_worsen_tolerance: float = 0.0,
    worst_quartile_worsen_tolerance: float = 0.0,
    pooled_worsen_abs_tolerance: float = 0.05,
    major_family_collapse_abs_tolerance: float = 0.15,
    fold_driven_max_fraction_worse: float = 0.5,
    family_driven_max_share: float = 0.5,
    bootstrap_seed: int,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Phase 2.6B Part G final-candidate promotion gate: a pure function of
    two already-computed result dicts, a fold/seed-stability summary, and an
    externally-verified determinism flag -- never fits a model.

    Unlike `evaluate_family_weighting_gate`/`evaluate_censored_learning_gate`
    (which require a >=5% macro-family improvement to *adopt* a change),
    this gate asks the weaker "may this candidate/ensemble replace the
    frozen single-model baseline as the final predictor" question (this
    phase's task brief, Part G): (1) macro-family MAE is not worse than
    `macro_worsen_tolerance` absolute units, (2) worst-quartile-family MAE
    is not worse than `worst_quartile_worsen_tolerance`, (3) pooled MAE is
    not worse by more than `pooled_worsen_abs_tolerance`, (4) a paired
    group-bootstrap CI of the macro-family-MAE delta does not show the
    candidate reliably worse (its lower bound is not > 0), (5) no major
    family (LOMFO) worsens by more than
    `major_family_collapse_abs_tolerance`, (6) the result is not
    one-fold-or-family-driven, and (7) uncertainty/applicability-domain
    outputs were independently confirmed reproducible (checked by the
    caller and passed in as `uncertainty_and_ad_reproducible`; this gate
    performs no determinism check itself).
    """
    macro_delta = candidate["macro_family_mae"] - baseline["macro_family_mae"]
    worst_q_delta = candidate["worst_quartile_family_mae"] - baseline["worst_quartile_family_mae"]
    pooled_delta = candidate["sample_weighted_mae"] - baseline["sample_weighted_mae"]
    paired_ci = paired_group_bootstrap_delta_ci(
        baseline["per_family_mae"], candidate["per_family_mae"], seed=bootstrap_seed, n_boot=n_boot
    )
    family_concentration = max_single_family_improvement_share(
        baseline["per_family_mae"], candidate["per_family_mae"]
    )
    lomfo_deltas = {
        fam: candidate["leave_one_major_family_out"][fam]["mae_mean"]
        - baseline["leave_one_major_family_out"][fam]["mae_mean"]
        for fam in major_families
    }
    max_lomfo_delta = max(lomfo_deltas.values()) if lomfo_deltas else 0.0
    not_one_fold_driven = (
        fold_seed_stability["fraction_worse_than_baseline"] <= fold_driven_max_fraction_worse
    )
    not_one_family_driven = family_concentration["max_share"] <= family_driven_max_share

    criteria = {
        "macro_family_mae_not_worse": macro_delta <= macro_worsen_tolerance,
        "worst_quartile_family_mae_not_worse": worst_q_delta <= worst_quartile_worsen_tolerance,
        "pooled_mae_not_worse_by_gt_tolerance": pooled_delta <= pooled_worsen_abs_tolerance,
        "paired_bootstrap_not_reliably_worse": not (paired_ci["ci_low"] > 0.0),
        "no_major_family_collapse": max_lomfo_delta <= major_family_collapse_abs_tolerance,
        "not_one_fold_or_family_driven": bool(not_one_fold_driven and not_one_family_driven),
        "uncertainty_and_ad_reproducible": bool(uncertainty_and_ad_reproducible),
    }
    return {
        "macro_family_mae_delta_abs": macro_delta,
        "worst_quartile_family_mae_delta_abs": worst_q_delta,
        "sample_weighted_mae_delta_abs": pooled_delta,
        "paired_group_bootstrap_macro_family_mae_delta_ci": paired_ci,
        "lomfo_deltas": lomfo_deltas,
        "max_lomfo_delta": max_lomfo_delta,
        "fold_seed_stability": fold_seed_stability,
        "family_concentration": family_concentration,
        "criteria": criteria,
        "verdict": "PROMOTE" if all(criteria.values()) else "RETAIN_BASELINE",
    }


def select_best_family_weighting_candidate(
    gate_results: dict[str, dict[str, Any]],
    macro_family_mae_by_candidate: dict[str, float],
) -> dict[str, Any]:
    """Deterministic candidate selection from a `{name: gate_result}` map
    (each `gate_result` from `evaluate_family_weighting_gate`).

    Among candidates with verdict "GO", picks the lowest macro-family MAE,
    breaking ties by candidate name (ascending) for full determinism. If no
    candidate reaches GO, still reports the best-by-macro-family-MAE
    candidate (for the report table) but flags `any_go=False` -- CLAUDE.md
    "Do not select seeds/splits/models by final-test performance" is
    satisfied because this ranks only on train-only family-CV metrics, never
    the final test set.
    """
    if not gate_results:
        raise ValueError("select_best_family_weighting_candidate: no candidates given")

    def _sort_key(name: str) -> tuple[float, str]:
        return (macro_family_mae_by_candidate[name], name)

    go_candidates = sorted(
        (name for name, g in gate_results.items() if g["verdict"] == "GO"), key=_sort_key
    )
    all_ranked = sorted(gate_results, key=_sort_key)
    selected = go_candidates[0] if go_candidates else all_ranked[0]
    return {
        "selected": selected,
        "any_go": bool(go_candidates),
        "go_candidates_ranked": go_candidates,
        "all_candidates_ranked_by_macro_family_mae": all_ranked,
    }
