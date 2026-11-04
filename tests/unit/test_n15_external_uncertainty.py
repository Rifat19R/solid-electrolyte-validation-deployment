import numpy as np
import pytest

from sse_transfer.evaluation.n15_external_uncertainty import (
    row_paired_bootstrap_mae_ci_and_ranking,
)


def test_row_paired_bootstrap_reproducible_with_fixed_seed() -> None:
    rng = np.random.default_rng(0)
    residuals = {
        "a": rng.normal(0, 1, size=50),
        "b": rng.normal(0.5, 1, size=50),
    }
    out1 = row_paired_bootstrap_mae_ci_and_ranking(residuals, seed=20260921, n_boot=200)
    out2 = row_paired_bootstrap_mae_ci_and_ranking(residuals, seed=20260921, n_boot=200)
    assert out1["per_model"]["a"]["ci_low"] == pytest.approx(out2["per_model"]["a"]["ci_low"])
    assert out1["per_model"]["a"]["ci_high"] == pytest.approx(out2["per_model"]["a"]["ci_high"])
    assert out1["per_model"]["a"]["probability_rank_1"] == pytest.approx(
        out2["per_model"]["a"]["probability_rank_1"]
    )


def test_row_paired_bootstrap_best_model_always_rank_1_when_dominant() -> None:
    # model "a" has zero residual everywhere; model "b" always has larger error.
    residuals = {
        "a": np.zeros(30),
        "b": np.full(30, 5.0),
    }
    out = row_paired_bootstrap_mae_ci_and_ranking(residuals, seed=1, n_boot=100)
    assert out["best_model_point_estimate"] == "a"
    assert out["per_model"]["a"]["probability_rank_1"] == pytest.approx(1.0)
    assert out["per_model"]["b"]["probability_rank_1"] == pytest.approx(0.0)
    assert out["pairwise_win_rate_descriptive_only"]["a"]["b"] == pytest.approx(1.0)


def test_row_paired_bootstrap_rank_frequencies_sum_to_one() -> None:
    rng = np.random.default_rng(2)
    residuals = {m: rng.normal(0, 1, size=40) for m in ["a", "b", "c"]}
    out = row_paired_bootstrap_mae_ci_and_ranking(residuals, seed=3, n_boot=150)
    for m in ["a", "b", "c"]:
        total = sum(out["per_model"][m]["rank_frequency"].values())
        assert total == pytest.approx(1.0)


def test_row_paired_bootstrap_rejects_mismatched_lengths() -> None:
    residuals = {"a": np.zeros(10), "b": np.zeros(5)}
    with pytest.raises(ValueError):
        row_paired_bootstrap_mae_ci_and_ranking(residuals, seed=0, n_boot=10)


def test_row_paired_bootstrap_rejects_empty() -> None:
    with pytest.raises(ValueError):
        row_paired_bootstrap_mae_ci_and_ranking({}, seed=0, n_boot=10)
