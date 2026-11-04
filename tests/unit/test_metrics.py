import math
import warnings

import numpy as np
import pytest

from sse_transfer.evaluation.metrics import (
    bootstrap_metric_ci,
    regression_metrics,
    relative_improvement_pct,
)


def test_regression_metrics_perfect_predictions() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = regression_metrics(y, y)
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["median_absolute_error"] == 0.0
    assert result["r2"] == 1.0
    assert result["spearman_rho"] == 1.0
    assert result["n"] == 4


def test_regression_metrics_known_constant_offset() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = y_true + 2.0
    result = regression_metrics(y_true, y_pred)
    assert result["mae"] == pytest.approx(2.0)
    assert result["rmse"] == pytest.approx(2.0)
    assert result["median_absolute_error"] == pytest.approx(2.0)


def test_regression_metrics_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0, 2.0]), np.array([1.0]))


def test_regression_metrics_constant_y_true_spearman_is_nan_without_warning() -> None:
    """Rank correlation is mathematically undefined when y_true has zero
    variance; must be NaN without scipy ever being asked to compute it
    (no ConstantInputWarning)."""
    y_true = np.array([2.0, 2.0, 2.0, 2.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = regression_metrics(y_true, y_pred)
    assert math.isnan(result["spearman_rho"])
    # Other metrics remain well-defined and unaffected.
    assert result["mae"] == pytest.approx(1.0)


def test_regression_metrics_constant_y_pred_spearman_is_nan_without_warning() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([3.0, 3.0, 3.0, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = regression_metrics(y_true, y_pred)
    assert math.isnan(result["spearman_rho"])


def test_regression_metrics_n_le_1_spearman_is_nan() -> None:
    result = regression_metrics(np.array([1.0]), np.array([1.0]))
    assert math.isnan(result["spearman_rho"])
    assert math.isnan(result["r2"])


def test_regression_metrics_nonconstant_spearman_matches_known_value() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([5.0, 3.0, 4.0, 2.0, 1.0])
    result = regression_metrics(y_true, y_pred)
    assert result["spearman_rho"] == pytest.approx(-0.9)


def test_bootstrap_ci_contains_point_estimate_and_has_nonneg_width() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.1, size=50)
    result = bootstrap_metric_ci(y_true, y_pred, "mae", seed=1, n_boot=200)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]
    assert result["ci_high"] >= result["ci_low"]
    assert result["n_boot"] == 200


def test_bootstrap_ci_is_deterministic_for_fixed_seed() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.3])
    first = bootstrap_metric_ci(y_true, y_pred, "mae", seed=42, n_boot=100)
    second = bootstrap_metric_ci(y_true, y_pred, "mae", seed=42, n_boot=100)
    assert first == second


def test_bootstrap_ci_spearman_ignores_constant_resamples() -> None:
    """Regression test: a bootstrap resample can draw only duplicate-valued
    rows, making spearman_rho NaN for that resample. The reported CI bounds
    must be computed only from finite resamples, not corrupted by NaN."""
    # Half the rows share an identical y_true/y_pred value, so a sizeable
    # fraction of resamples (drawn with replacement) land on a constant
    # subsample and yield NaN spearman_rho.
    y_true = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 1.0, 1.0, 2.1, 2.9, 4.2])
    result = bootstrap_metric_ci(y_true, y_pred, "spearman_rho", seed=0, n_boot=500)
    assert not math.isnan(result["ci_low"])
    assert not math.isnan(result["ci_high"])
    assert 0 < result["n_finite"] <= 500
    assert result["ci_low"] <= result["ci_high"]


def test_bootstrap_ci_all_constant_resamples_yields_nan_bounds() -> None:
    y_true = np.array([5.0, 5.0])
    y_pred = np.array([5.0, 5.0])
    result = bootstrap_metric_ci(y_true, y_pred, "spearman_rho", seed=0, n_boot=50)
    assert result["n_finite"] == 0
    assert math.isnan(result["ci_low"])
    assert math.isnan(result["ci_high"])


def test_bootstrap_ci_unknown_metric_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_metric_ci(np.array([1.0]), np.array([1.0]), "not_a_metric", seed=0)


def test_relative_improvement_pct_matches_full_precision_reference() -> None:
    # Values with more significant digits than any report ever displays (2dp),
    # so a regression that rounds an input before dividing changes the result.
    baseline_mae = 1.760612345678
    candidate_mae = 1.809712345678
    expected = 100.0 * (baseline_mae - candidate_mae) / baseline_mae
    assert relative_improvement_pct(baseline_mae, candidate_mae) == pytest.approx(
        expected, abs=1e-12
    )
    # Same pair rounded to the 2dp a verdict .md would print gives a materially
    # different percentage -- this pins full-precision inputs as the contract.
    rounded_result = relative_improvement_pct(round(baseline_mae, 2), round(candidate_mae, 2))
    assert abs(rounded_result - expected) > 1e-6


def test_relative_improvement_pct_sign_convention() -> None:
    # Candidate strictly better (lower MAE) -> positive percent.
    assert relative_improvement_pct(2.0, 1.0) == pytest.approx(50.0)
    # Candidate strictly worse (higher MAE) -> negative percent.
    assert relative_improvement_pct(1.0, 2.0) == pytest.approx(-100.0)
    # Identical MAE -> zero.
    assert relative_improvement_pct(1.0, 1.0) == pytest.approx(0.0)


def test_relative_improvement_pct_zero_baseline_is_nan() -> None:
    import math

    assert math.isnan(relative_improvement_pct(0.0, 1.0))
