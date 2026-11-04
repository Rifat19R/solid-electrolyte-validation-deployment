import pandas as pd

from sse_transfer.features.composition_features import (
    COMPOSITION_FEATURE_NAMES,
    composition_features,
    is_disordered_composition,
)


def test_composition_features_are_deterministic() -> None:
    series = pd.Series(["Li7La3Zr2O12", "Li3PS4"])
    first = composition_features(series)
    second = composition_features(series)
    pd.testing.assert_frame_equal(first, second)


def test_composition_features_have_expected_shape_and_columns() -> None:
    series = pd.Series(["Li7La3Zr2O12"])
    result = composition_features(series)
    assert result.shape == (1, len(COMPOSITION_FEATURE_NAMES))
    assert list(result.columns) == list(COMPOSITION_FEATURE_NAMES)
    assert not result.isna().any().any()


def test_composition_features_differ_for_different_compositions() -> None:
    series = pd.Series(["Li7La3Zr2O12", "Li3PS4"])
    result = composition_features(series)
    assert not result.iloc[0].equals(result.iloc[1])


def test_composition_features_unparseable_row_yields_nan_not_exception() -> None:
    series = pd.Series(["Li7La3Zr2O12", "not-a-formula!!!"])
    result = composition_features(series)
    assert not result.iloc[0].isna().any()
    assert result.iloc[1].isna().all()


def test_is_disordered_composition_true_for_fractional_amounts() -> None:
    assert is_disordered_composition("Li5.46Hf12.13P18.0O72.0") is True


def test_is_disordered_composition_false_for_integer_amounts() -> None:
    assert is_disordered_composition("Li7La3Zr2O12") is False


def test_is_disordered_composition_false_for_unparseable_formula() -> None:
    assert is_disordered_composition("not-a-formula!!!") is False
