import numpy as np
import pandas as pd
import pytest

from sse_transfer.evaluation.n15_baseline_variability import (
    constant_predictor_metrics,
    within_composition_variability,
)


def test_constant_predictor_metrics_mae_matches_hand_calc() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    out = constant_predictor_metrics(y_true, const_value=2.5)
    assert out["mae"] == pytest.approx(np.mean(np.abs(y_true - 2.5)))


def test_within_composition_variability_basic_groups() -> None:
    ledger = pd.DataFrame(
        {
            "reduced_formula": ["LiF", "LiF", "LiF", "Li2O", "LiCl"],
            "log_target": [-3.0, -3.2, -2.8, -4.0, -5.0],
        }
    )
    out = within_composition_variability(ledger, min_repeats=2)
    # LiF has 3 repeats (range 0.4); Li2O and LiCl have 1 each -> excluded.
    assert out["n_groups_with_repeats"] == 1
    assert out["per_group"]["LiF"]["n_repeats"] == 3
    assert out["per_group"]["LiF"]["range"] == pytest.approx(0.4)
    assert out["range_median"] == pytest.approx(0.4)


def test_within_composition_variability_no_repeats_returns_empty() -> None:
    ledger = pd.DataFrame({"reduced_formula": ["LiF", "Li2O"], "log_target": [-3.0, -4.0]})
    out = within_composition_variability(ledger, min_repeats=2)
    assert out["n_groups_with_repeats"] == 0
    assert out["per_group"] == {}


def test_within_composition_variability_label_disclaims_noise_floor() -> None:
    ledger = pd.DataFrame({"reduced_formula": ["LiF", "LiF"], "log_target": [-3.0, -3.5]})
    out = within_composition_variability(ledger, min_repeats=2)
    label = out["interpretation_label"].lower()
    assert "heterogeneity" in label
    assert "not an irreducible noise floor" in label
