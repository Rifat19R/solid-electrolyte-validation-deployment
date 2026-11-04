import math

import pandas as pd

from sse_transfer.data.obelix_loader import censoring_bound, is_censored, label_status


def test_is_censored_detects_inequality_strings() -> None:
    values = pd.Series(["1.58489e-06", "<1E-10", "0.0032", ">5e-3", "  <2e-4", "3.2e-07"])
    result = is_censored(values)
    assert result.tolist() == [False, True, False, True, True, False]


def test_is_censored_all_false_for_purely_numeric_series() -> None:
    values = pd.Series([1e-6, 2e-7, 3e-8])
    assert not is_censored(values).any()


def test_label_status_classifies_exact_left_right_and_missing() -> None:
    values = pd.Series(["3.2e-07", "<1E-10", ">5e-3", None, float("nan"), ""])
    result = label_status(values)
    assert result.tolist() == [
        "EXACT",
        "LEFT_CENSORED",
        "RIGHT_CENSORED",
        "MISSING",
        "MISSING",
        "MISSING",
    ]


def test_label_status_never_converts_a_censored_value_to_exact() -> None:
    values = pd.Series(["<1E-10", ">5e-3"])
    result = label_status(values)
    assert "EXACT" not in result.tolist()


def test_censoring_bound_extracts_numeric_magnitude() -> None:
    values = pd.Series(["3.2e-07", "<1E-10", ">5e-3"])
    bounds = censoring_bound(values)
    assert math.isclose(bounds.iloc[0], 3.2e-07)
    assert math.isclose(bounds.iloc[1], 1e-10)
    assert math.isclose(bounds.iloc[2], 5e-3)


def test_censoring_bound_is_nan_for_missing() -> None:
    bounds = censoring_bound(pd.Series([None, float("nan")]))
    assert bounds.isna().all()
