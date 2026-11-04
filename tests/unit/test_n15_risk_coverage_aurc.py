import pytest

from sse_transfer.evaluation.n15_risk_coverage_aurc import aurc_from_risk_coverage


def test_aurc_constant_mae_equals_that_mae() -> None:
    rows = [
        {"retained_fraction": 1.0, "mae": 2.0},
        {"retained_fraction": 0.75, "mae": 2.0},
        {"retained_fraction": 0.5, "mae": 2.0},
        {"retained_fraction": 0.25, "mae": 2.0},
    ]
    out = aurc_from_risk_coverage(rows)
    assert out["aurc"] == pytest.approx(2.0)
    assert out["coverage_min"] == pytest.approx(0.25)
    assert out["coverage_max"] == pytest.approx(1.0)
    assert out["n_points"] == 4


def test_aurc_linear_curve_matches_trapezoid_by_hand() -> None:
    # mae = coverage (linear), over [0.25, 1.0]: area = 0.5*(0.25+1.0)*(0.75) = 0.46875
    rows = [
        {"retained_fraction": 1.0, "mae": 1.0},
        {"retained_fraction": 0.25, "mae": 0.25},
    ]
    out = aurc_from_risk_coverage(rows)
    expected_area = 0.5 * (0.25 + 1.0) * 0.75
    assert out["aurc"] == pytest.approx(expected_area / 0.75)


def test_aurc_order_independent() -> None:
    rows_asc = [
        {"retained_fraction": 0.25, "mae": 1.0},
        {"retained_fraction": 0.5, "mae": 2.0},
        {"retained_fraction": 1.0, "mae": 3.0},
    ]
    rows_desc = list(reversed(rows_asc))
    assert aurc_from_risk_coverage(rows_asc)["aurc"] == pytest.approx(
        aurc_from_risk_coverage(rows_desc)["aurc"]
    )


def test_aurc_rejects_single_point() -> None:
    with pytest.raises(ValueError):
        aurc_from_risk_coverage([{"retained_fraction": 1.0, "mae": 1.0}])
