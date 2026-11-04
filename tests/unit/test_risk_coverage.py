import numpy as np
import pytest

from sse_transfer.uncertainty.risk_coverage import risk_coverage_mae_at_retained_fraction


def test_risk_coverage_mae_full_coverage_equals_pooled_mae() -> None:
    ad = np.array([0.4, 0.1, 0.3, 0.2])
    err = np.array([4.0, 1.0, 3.0, 2.0])
    out = risk_coverage_mae_at_retained_fraction(ad, err, retained_fractions=(1.0,))
    assert out[0]["retained_fraction"] == 1.0
    assert out[0]["n_retained"] == 4
    assert out[0]["mae"] == pytest.approx(np.mean(err))


def test_risk_coverage_mae_monotonic_when_ad_and_error_are_correlated() -> None:
    # Lowest-AD rows also have the lowest error, so retaining a smaller
    # in-domain fraction should never increase MAE.
    ad = np.arange(20, dtype=float)
    err = np.arange(20, dtype=float)
    out = risk_coverage_mae_at_retained_fraction(ad, err, retained_fractions=(1.0, 0.75, 0.5, 0.25))
    maes = [row["mae"] for row in out]
    assert maes == sorted(maes, reverse=True)


def test_risk_coverage_mae_n_retained_rounds_and_floors_at_one() -> None:
    ad = np.array([0.1, 0.2, 0.3])
    err = np.array([1.0, 2.0, 3.0])
    out = risk_coverage_mae_at_retained_fraction(ad, err, retained_fractions=(0.1,))
    assert out[0]["n_retained"] == 1
    assert out[0]["mae"] == pytest.approx(1.0)


def test_risk_coverage_mae_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        risk_coverage_mae_at_retained_fraction(np.array([1.0, 2.0]), np.array([1.0]))


def test_risk_coverage_mae_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        risk_coverage_mae_at_retained_fraction(np.array([]), np.array([]))


def test_risk_coverage_mae_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        risk_coverage_mae_at_retained_fraction(
            np.array([1.0]), np.array([1.0]), retained_fractions=(1.5,)
        )
    with pytest.raises(ValueError):
        risk_coverage_mae_at_retained_fraction(
            np.array([1.0]), np.array([1.0]), retained_fractions=(0.0,)
        )
