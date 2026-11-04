from __future__ import annotations

import numpy as np
import pytest

from sse_transfer.evaluation.regime_utility import (
    bootstrap_regime_pair_delta_ci,
    cluster_bootstrap_regime_pair_delta_ci,
    formula_balanced_mae,
    formula_cluster_ids,
    macro_family_mae_excluding,
    median_abs_mae_gap,
    rank_agreement,
    regime_summary,
    top_model_selection_regret,
)


def test_median_abs_mae_gap_basic() -> None:
    internal = {"a": 1.0, "b": 2.0, "c": 3.0}
    external = {"a": 1.5, "b": 2.5, "c": 6.0}
    # gaps: 0.5, 0.5, 3.0 -> median 0.5
    assert median_abs_mae_gap(internal, external) == pytest.approx(0.5)


def test_median_abs_mae_gap_raises_on_model_mismatch() -> None:
    with pytest.raises(ValueError):
        median_abs_mae_gap({"a": 1.0}, {"b": 1.0})


def test_rank_agreement_perfect_agreement() -> None:
    internal = {"a": 1.0, "b": 2.0, "c": 3.0}
    external = {"a": 0.1, "b": 0.2, "c": 0.3}
    out = rank_agreement(internal, external)
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["kendall_tau"] == pytest.approx(1.0)


def test_rank_agreement_perfect_disagreement() -> None:
    internal = {"a": 1.0, "b": 2.0, "c": 3.0}
    external = {"a": 0.3, "b": 0.2, "c": 0.1}
    out = rank_agreement(internal, external)
    assert out["spearman_rho"] == pytest.approx(-1.0)
    assert out["kendall_tau"] == pytest.approx(-1.0)


def test_rank_agreement_constant_input_returns_nan_not_crash() -> None:
    internal = {"a": 1.0, "b": 1.0, "c": 1.0}
    external = {"a": 0.3, "b": 0.2, "c": 0.1}
    out = rank_agreement(internal, external)
    assert np.isnan(out["spearman_rho"])
    assert np.isnan(out["kendall_tau"])


def test_top_model_selection_regret_zero_when_picks_agree() -> None:
    internal = {"a": 1.0, "b": 2.0, "c": 3.0}
    external = {"a": 0.5, "b": 5.0, "c": 6.0}
    # argmin internal = "a", argmin external = "a" -> regret 0
    assert top_model_selection_regret(internal, external) == pytest.approx(0.0)


def test_top_model_selection_regret_positive_when_picks_disagree() -> None:
    internal = {"a": 1.0, "b": 2.0, "c": 3.0}
    external = {"a": 5.0, "b": 0.5, "c": 6.0}
    # argmin internal = "a" (external 5.0), true best external = "b" (0.5)
    assert top_model_selection_regret(internal, external) == pytest.approx(4.5)


def test_top_model_selection_regret_never_negative() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        models = [f"m{i}" for i in range(5)]
        internal = dict(zip(models, rng.uniform(0, 3, size=5).tolist(), strict=False))
        external = dict(zip(models, rng.uniform(0, 3, size=5).tolist(), strict=False))
        assert top_model_selection_regret(internal, external) >= 0.0


def test_regime_summary_has_all_four_statistics() -> None:
    internal = {"a": 1.0, "b": 2.0, "c": 3.0}
    external = {"a": 0.5, "b": 5.0, "c": 6.0}
    out = regime_summary(internal, external)
    assert set(out) == {
        "median_abs_mae_gap",
        "spearman_rho",
        "kendall_tau",
        "top_model_selection_regret",
        "n_models",
    }


def test_bootstrap_regime_pair_delta_ci_zero_when_identical_regimes() -> None:
    internal = {"a": 1.0, "b": 2.0}
    rng = np.random.default_rng(1)
    residuals = {
        "a": rng.normal(0, 0.5, size=50),
        "b": rng.normal(0, 0.5, size=50),
    }
    out = bootstrap_regime_pair_delta_ci(
        internal, internal, residuals, statistic="median_abs_mae_gap", seed=42, n_boot=200
    )
    assert out["point_delta"] == pytest.approx(0.0)
    assert out["ci_low"] <= 0.0 <= out["ci_high"]


def test_bootstrap_regime_pair_delta_ci_raises_on_model_mismatch() -> None:
    with pytest.raises(ValueError):
        bootstrap_regime_pair_delta_ci(
            {"a": 1.0},
            {"a": 1.0},
            {"b": np.array([0.1, 0.2])},
            statistic="median_abs_mae_gap",
            seed=1,
            n_boot=10,
        )


def test_bootstrap_regime_pair_delta_ci_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        bootstrap_regime_pair_delta_ci(
            {"a": 1.0, "b": 1.0},
            {"a": 1.0, "b": 1.0},
            {"a": np.array([0.1, 0.2, 0.3]), "b": np.array([0.1, 0.2])},
            statistic="median_abs_mae_gap",
            seed=1,
            n_boot=10,
        )


def test_bootstrap_regime_pair_delta_ci_deterministic_for_fixed_seed() -> None:
    internal_a = {"a": 1.0, "b": 2.0}
    internal_b = {"a": 1.5, "b": 1.5}
    rng = np.random.default_rng(2)
    residuals = {"a": rng.normal(0, 1.0, size=30), "b": rng.normal(0, 1.0, size=30)}
    out1 = bootstrap_regime_pair_delta_ci(
        internal_a, internal_b, residuals, statistic="median_abs_mae_gap", seed=7, n_boot=100
    )
    out2 = bootstrap_regime_pair_delta_ci(
        internal_a, internal_b, residuals, statistic="median_abs_mae_gap", seed=7, n_boot=100
    )
    assert out1 == out2


def test_macro_family_mae_excluding_removes_target_family() -> None:
    per_family = {
        "a": {"mae": 1.0, "n": 5},
        "b": {"mae": 2.0, "n": 5},
        "__MISSING__": {"mae": 100.0, "n": 5},
    }
    assert macro_family_mae_excluding(per_family, "__MISSING__") == pytest.approx(1.5)


def test_macro_family_mae_excluding_matches_manual_mean_when_family_absent() -> None:
    per_family = {"a": {"mae": 1.0, "n": 5}, "b": {"mae": 3.0, "n": 5}}
    # excluding a family not present is a no-op over the remaining families
    assert macro_family_mae_excluding(per_family, "nonexistent") == pytest.approx(2.0)


def test_macro_family_mae_excluding_raises_when_nothing_remains() -> None:
    per_family = {"__MISSING__": {"mae": 1.0, "n": 5}}
    with pytest.raises(ValueError):
        macro_family_mae_excluding(per_family, "__MISSING__")


def test_bootstrap_regime_pair_delta_ci_invalid_statistic_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_regime_pair_delta_ci(
            {"a": 1.0}, {"a": 1.0}, {"a": np.array([0.1])}, statistic="nonsense", seed=1, n_boot=10
        )


def test_cluster_bootstrap_deterministic_for_fixed_seed() -> None:
    internal_a = {"a": 1.0, "b": 2.0}
    internal_b = {"a": 1.5, "b": 1.5}
    rng = np.random.default_rng(3)
    residuals = {"a": rng.normal(0, 1.0, size=20), "b": rng.normal(0, 1.0, size=20)}
    cluster_id = np.repeat(np.arange(10), 2)  # 10 clusters of 2 rows each
    out1 = cluster_bootstrap_regime_pair_delta_ci(
        internal_a,
        internal_b,
        residuals,
        cluster_id,
        statistic="median_abs_mae_gap",
        seed=7,
        n_boot=100,
    )
    out2 = cluster_bootstrap_regime_pair_delta_ci(
        internal_a,
        internal_b,
        residuals,
        cluster_id,
        statistic="median_abs_mae_gap",
        seed=7,
        n_boot=100,
    )
    assert out1 == out2
    assert out1["n_clusters"] == 10


def test_cluster_bootstrap_singleton_clusters_matches_row_level_ci_distribution() -> None:
    # When every cluster has exactly one row, cluster bootstrap degenerates
    # to row-level bootstrap (same resampling unit) -- point estimate and CI
    # bounds should match the existing row-level function under the same seed.
    internal_a = {"a": 1.0, "b": 2.0}
    internal_b = {"a": 1.5, "b": 1.5}
    rng = np.random.default_rng(5)
    residuals = {"a": rng.normal(0, 1.0, size=50), "b": rng.normal(0, 1.0, size=50)}
    cluster_id = np.arange(50)
    row_level = bootstrap_regime_pair_delta_ci(
        internal_a, internal_b, residuals, statistic="median_abs_mae_gap", seed=11, n_boot=500
    )
    cluster_level = cluster_bootstrap_regime_pair_delta_ci(
        internal_a,
        internal_b,
        residuals,
        cluster_id,
        statistic="median_abs_mae_gap",
        seed=11,
        n_boot=500,
    )
    assert cluster_level["point_delta"] == pytest.approx(row_level["point_delta"])
    assert cluster_level["n_clusters"] == 50


def test_cluster_bootstrap_row_count_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap_regime_pair_delta_ci(
            {"a": 1.0},
            {"a": 1.0},
            {"a": np.array([0.1, 0.2])},
            np.array([0]),
            statistic="median_abs_mae_gap",
            seed=1,
            n_boot=10,
        )


def test_median_abs_mae_gap_rejects_nan_input() -> None:
    with pytest.raises(ValueError):
        median_abs_mae_gap({"a": float("nan"), "b": 1.0}, {"a": 1.0, "b": 1.0})


def test_rank_agreement_rejects_inf_input() -> None:
    with pytest.raises(ValueError):
        rank_agreement({"a": float("inf"), "b": 1.0, "c": 2.0}, {"a": 1.0, "b": 1.0, "c": 2.0})


def test_top_model_selection_regret_rejects_nan_input() -> None:
    with pytest.raises(ValueError):
        top_model_selection_regret({"a": float("nan"), "b": 1.0}, {"a": 1.0, "b": 1.0})


# --- reviewer remediation Issue 10: formula-balanced sensitivity ---


def test_formula_cluster_ids_groups_exact_duplicates() -> None:
    formulas = ["LiFePO4", "Li7La3Zr2O12", "LiFePO4", "LiFePO4", "Li2S"]
    ids = formula_cluster_ids(formulas)
    assert ids[0] == ids[2] == ids[3]
    assert ids[1] != ids[0]
    assert ids[4] != ids[0] and ids[4] != ids[1]
    assert len(set(ids.tolist())) == 3


def test_formula_cluster_ids_all_unique_gives_one_row_per_cluster() -> None:
    formulas = [f"F{i}" for i in range(10)]
    ids = formula_cluster_ids(formulas)
    assert len(set(ids.tolist())) == 10


def test_formula_balanced_mae_matches_explicit_group_averaging() -> None:
    # Two formulas: "A" has 3 rows (residuals 1,2,3 -> group mean 2.0),
    # "B" has 1 row (residual 10.0 -> group mean 10.0).
    # Formula-balanced MAE = mean(2.0, 10.0) = 6.0, NOT the row mean (16/4=4.0).
    formulas = ["A", "A", "A", "B"]
    residuals = np.array([1.0, 2.0, 3.0, 10.0])
    result = formula_balanced_mae(formulas, residuals)
    assert result == pytest.approx(6.0)
    # Row-weighted mean would differ, confirming this is a genuine reweighting.
    assert result != pytest.approx(float(np.mean(residuals)))


def test_formula_balanced_mae_equals_row_mean_when_all_formulas_unique() -> None:
    rng = np.random.default_rng(0)
    residuals = rng.uniform(0, 2, size=50)
    formulas = [f"F{i}" for i in range(50)]
    assert formula_balanced_mae(formulas, residuals) == pytest.approx(float(np.mean(residuals)))


def test_formula_balanced_mae_invariant_to_row_order_within_group() -> None:
    formulas_a = ["A", "A", "B", "B"]
    residuals_a = np.array([1.0, 3.0, 5.0, 7.0])
    formulas_b = ["B", "A", "B", "A"]
    residuals_b = np.array([5.0, 1.0, 7.0, 3.0])
    assert formula_balanced_mae(formulas_a, residuals_a) == pytest.approx(
        formula_balanced_mae(formulas_b, residuals_b)
    )


def test_formula_balanced_mae_raises_on_length_mismatch() -> None:
    with pytest.raises(ValueError):
        formula_balanced_mae(["A", "B"], np.array([1.0]))


def test_formula_balanced_mae_raises_on_empty_input() -> None:
    with pytest.raises(ValueError):
        formula_balanced_mae([], np.array([]))


def test_formula_cluster_bootstrap_deterministic_for_fixed_seed() -> None:
    internal_a = {"a": 1.0, "b": 2.0}
    internal_b = {"a": 1.5, "b": 1.5}
    rng = np.random.default_rng(9)
    residuals = {"a": rng.normal(0, 1.0, size=40), "b": rng.normal(0, 1.0, size=40)}
    formulas = [f"F{i // 2}" for i in range(40)]  # 20 formula groups of 2 rows each
    cluster_id = formula_cluster_ids(formulas)
    out1 = cluster_bootstrap_regime_pair_delta_ci(
        internal_a,
        internal_b,
        residuals,
        cluster_id,
        statistic="median_abs_mae_gap",
        seed=13,
        n_boot=100,
    )
    out2 = cluster_bootstrap_regime_pair_delta_ci(
        internal_a,
        internal_b,
        residuals,
        cluster_id,
        statistic="median_abs_mae_gap",
        seed=13,
        n_boot=100,
    )
    assert out1 == out2
    assert out1["n_clusters"] == 20


def test_formula_cluster_bootstrap_singleton_formulas_matches_row_level_point() -> None:
    # Every formula unique -> formula-cluster bootstrap's point estimate must
    # match the plain row-level bootstrap's point estimate exactly (same
    # statistic, same frozen residuals; only the resampling draw sequence
    # differs between rng.integers and rng.choice).
    internal_a = {"a": 1.0, "b": 2.0}
    internal_b = {"a": 1.5, "b": 1.5}
    rng = np.random.default_rng(21)
    residuals = {"a": rng.normal(0, 1.0, size=30), "b": rng.normal(0, 1.0, size=30)}
    formulas = [f"F{i}" for i in range(30)]
    cluster_id = formula_cluster_ids(formulas)
    row_level = bootstrap_regime_pair_delta_ci(
        internal_a, internal_b, residuals, statistic="median_abs_mae_gap", seed=5, n_boot=300
    )
    formula_level = cluster_bootstrap_regime_pair_delta_ci(
        internal_a,
        internal_b,
        residuals,
        cluster_id,
        statistic="median_abs_mae_gap",
        seed=5,
        n_boot=300,
    )
    assert formula_level["point_delta"] == pytest.approx(row_level["point_delta"])
    assert formula_level["n_clusters"] == 30
