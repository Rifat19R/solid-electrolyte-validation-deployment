import numpy as np
import pandas as pd
import pytest

from sse_transfer.uncertainty.applicability_domain import (
    ad_score_error_correlation,
    ad_score_quantile_error_table,
    fit_standardization,
    nearest_neighbor_ad_score,
)


def test_nearest_neighbor_ad_score_zero_for_exact_match() -> None:
    x_ref = pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, 10.0]})
    x_query = pd.DataFrame({"a": [0.0], "b": [0.0]})
    scores = nearest_neighbor_ad_score(x_ref, x_query)
    assert scores[0] == pytest.approx(0.0, abs=1e-9)


def test_nearest_neighbor_ad_score_larger_for_far_point() -> None:
    x_ref = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    x_query = pd.DataFrame({"a": [1.5, 100.0]})
    scores = nearest_neighbor_ad_score(x_ref, x_query)
    assert scores[1] > scores[0]


def test_nearest_neighbor_ad_score_deterministic() -> None:
    x_ref = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [1.0, 1.0, 2.0]})
    x_query = pd.DataFrame({"a": [0.5, 5.0], "b": [1.0, 5.0]})
    first = nearest_neighbor_ad_score(x_ref, x_query)
    second = nearest_neighbor_ad_score(x_ref, x_query)
    assert np.array_equal(first, second)


def test_fit_standardization_handles_zero_variance_column() -> None:
    x_ref = pd.DataFrame({"a": [1.0, 1.0, 1.0]})
    stats = fit_standardization(x_ref)
    assert stats["std"][0] == pytest.approx(1.0)


def test_nearest_neighbor_ad_score_rejects_mismatched_columns() -> None:
    x_ref = pd.DataFrame({"a": [0.0, 1.0]})
    x_query = pd.DataFrame({"b": [0.0, 1.0]})
    with pytest.raises(ValueError):
        nearest_neighbor_ad_score(x_ref, x_query)


def test_ad_score_error_correlation_positive_when_monotonic() -> None:
    ad = np.array([0.1, 0.2, 0.3, 0.4])
    err = np.array([1.0, 2.0, 3.0, 4.0])
    result = ad_score_error_correlation(ad, err)
    assert result["spearman_r"] == pytest.approx(1.0)


def test_ad_score_error_correlation_nan_when_ad_constant() -> None:
    result = ad_score_error_correlation(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(result["spearman_r"])


def test_ad_score_quantile_error_table_bins_are_monotonic_on_sorted_input() -> None:
    ad = np.arange(8, dtype=float)
    err = np.arange(8, dtype=float)
    table = ad_score_quantile_error_table(ad, err, n_bins=4)
    means = [row["mean_abs_residual"] for row in table]
    assert means == sorted(means)
    assert sum(row["n"] for row in table) == 8


def test_ad_score_quantile_error_table_rejects_bad_n_bins() -> None:
    with pytest.raises(ValueError):
        ad_score_quantile_error_table(np.array([1.0]), np.array([1.0]), n_bins=0)
