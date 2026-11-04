import numpy as np
import pandas as pd
import pytest

from sse_transfer.evaluation.family_robustness import (
    evaluate_censored_learning_gate,
    evaluate_family_weighting_gate,
    family_sample_weights,
    family_sample_weights_by_alpha,
    group_bootstrap_ci,
    macro_family_mae,
    max_single_family_improvement_share,
    median_family_mae,
    paired_group_bootstrap_delta_ci,
    per_family_mae,
    sample_weighted_mae,
    select_best_family_weighting_candidate,
    worst_quartile_family_mae,
)


def test_family_sample_weights_uniform_is_all_ones() -> None:
    fam = pd.Series(["a", "a", "b", "b", "b"])
    w = family_sample_weights(fam, "uniform")
    assert np.allclose(w, 1.0)


def test_family_sample_weights_inverse_family_size_gives_equal_total_weight_per_family() -> None:
    fam = pd.Series(["a", "a", "b", "b", "b", "b"])
    w = family_sample_weights(fam, "inverse_family_size")
    total_a = w[fam.to_numpy() == "a"].sum()
    total_b = w[fam.to_numpy() == "b"].sum()
    assert total_a == pytest.approx(total_b)
    assert w.mean() == pytest.approx(1.0)


def test_family_sample_weights_sqrt_inverse_between_uniform_and_full_inverse() -> None:
    fam = pd.Series(["a"] * 1 + ["b"] * 9)
    w_uniform = family_sample_weights(fam, "uniform")
    w_sqrt = family_sample_weights(fam, "sqrt_inverse_family_size")
    w_inv = family_sample_weights(fam, "inverse_family_size")
    # Weight on the singleton family "a": uniform < sqrt-inverse < full inverse.
    assert w_uniform[0] < w_sqrt[0] < w_inv[0]


def test_family_sample_weights_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError):
        family_sample_weights(pd.Series(["a", "b"]), "not_a_scheme")


def test_family_sample_weights_rejects_missing_family() -> None:
    with pytest.raises(ValueError):
        family_sample_weights(pd.Series(["a", None]), "inverse_family_size")


def test_family_sample_weights_deterministic() -> None:
    fam = pd.Series(["x", "y", "y", "z", "z", "z"])
    first = family_sample_weights(fam, "sqrt_inverse_family_size")
    second = family_sample_weights(fam, "sqrt_inverse_family_size")
    assert np.array_equal(first, second)


def _example_per_family() -> dict[str, dict[str, float]]:
    # Family A: 8 rows, MAE 1.0 (dominates a pooled/sample-weighted average).
    # Family B: 1 row, MAE 5.0. Family C: 1 row, MAE 3.0.
    y_true = np.array([0.0] * 8 + [0.0] + [0.0])
    y_pred = np.array([1.0] * 8 + [5.0] + [3.0])
    fam = np.array(["A"] * 8 + ["B"] + ["C"])
    return per_family_mae(y_true, y_pred, fam)


def test_per_family_mae_basic() -> None:
    pf = _example_per_family()
    assert pf == {
        "A": {"mae": pytest.approx(1.0), "n": 8},
        "B": {"mae": pytest.approx(5.0), "n": 1},
        "C": {"mae": pytest.approx(3.0), "n": 1},
    }


def test_per_family_mae_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        per_family_mae(np.array([1.0, 2.0]), np.array([1.0]), np.array(["a", "b"]))


def test_sample_weighted_mae_matches_pooled_row_average() -> None:
    pf = _example_per_family()
    # (8*1.0 + 1*5.0 + 1*3.0) / 10 = 1.6
    assert sample_weighted_mae(pf) == pytest.approx(1.6)


def test_macro_family_mae_gives_each_family_equal_weight() -> None:
    pf = _example_per_family()
    # (1.0 + 5.0 + 3.0) / 3 = 3.0 -- very different from the pooled 1.6.
    assert macro_family_mae(pf) == pytest.approx(3.0)


def test_macro_family_mae_differs_from_sample_weighted_when_sizes_are_skewed() -> None:
    pf = _example_per_family()
    assert macro_family_mae(pf) != pytest.approx(sample_weighted_mae(pf))


def test_median_family_mae() -> None:
    pf = _example_per_family()
    assert median_family_mae(pf) == pytest.approx(3.0)


def test_worst_quartile_family_mae_always_includes_at_least_one_family() -> None:
    pf = {"a": {"mae": 1.0, "n": 5}, "b": {"mae": 2.0, "n": 5}}
    result = worst_quartile_family_mae(pf, quartile_fraction=0.25)
    assert result["n_families_in_worst_quartile"] == 1
    assert result["families"] == ["b"]
    assert result["mae"] == pytest.approx(2.0)


def test_worst_quartile_family_mae_selects_highest_mae_families() -> None:
    pf = {
        "a": {"mae": 1.0, "n": 5},
        "b": {"mae": 2.0, "n": 5},
        "c": {"mae": 3.0, "n": 5},
        "d": {"mae": 4.0, "n": 5},
    }
    result = worst_quartile_family_mae(pf, quartile_fraction=0.25)
    assert result["n_families_in_worst_quartile"] == 1
    assert result["families"] == ["d"]


def test_worst_quartile_family_mae_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        worst_quartile_family_mae({"a": {"mae": 1.0, "n": 1}}, quartile_fraction=1.5)


def test_group_bootstrap_ci_contains_point_and_is_deterministic() -> None:
    pf = _example_per_family()
    first = group_bootstrap_ci(pf, seed=0, n_boot=500)
    second = group_bootstrap_ci(pf, seed=0, n_boot=500)
    assert first == second
    assert first["ci_low"] <= first["point"] <= first["ci_high"]
    assert first["n_families"] == 3


def test_group_bootstrap_ci_median_statistic() -> None:
    pf = _example_per_family()
    result = group_bootstrap_ci(pf, seed=1, n_boot=500, statistic="median")
    assert result["point"] == pytest.approx(3.0)


def test_group_bootstrap_ci_rejects_unknown_statistic() -> None:
    with pytest.raises(ValueError):
        group_bootstrap_ci(_example_per_family(), seed=0, statistic="mean")


def test_paired_group_bootstrap_delta_ci_zero_when_identical() -> None:
    pf = _example_per_family()
    result = paired_group_bootstrap_delta_ci(pf, pf, seed=0, n_boot=200)
    assert result["point_delta_macro_family_mae"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)
    assert result["reproducible_improvement"] is False


def test_paired_group_bootstrap_delta_ci_detects_reproducible_improvement() -> None:
    baseline = {"a": {"mae": 2.0, "n": 5}, "b": {"mae": 3.0, "n": 5}, "c": {"mae": 4.0, "n": 5}}
    candidate = {"a": {"mae": 1.0, "n": 5}, "b": {"mae": 1.5, "n": 5}, "c": {"mae": 2.0, "n": 5}}
    result = paired_group_bootstrap_delta_ci(baseline, candidate, seed=0, n_boot=2000)
    assert result["point_delta_macro_family_mae"] < 0
    assert result["reproducible_improvement"] is True


def test_paired_group_bootstrap_delta_ci_requires_identical_family_sets() -> None:
    with pytest.raises(ValueError):
        paired_group_bootstrap_delta_ci(
            {"a": {"mae": 1.0, "n": 1}},
            {"b": {"mae": 1.0, "n": 1}},
            seed=0,
        )


# ---------------------------------------------------------------------------
# Phase 2.3: family_sample_weights_by_alpha (weight formula correctness)
# ---------------------------------------------------------------------------


def _mixed_family_series() -> pd.Series:
    return pd.Series(["a"] * 1 + ["b"] * 3 + ["c"] * 9)


def test_family_sample_weights_by_alpha_zero_is_uniform() -> None:
    fam = _mixed_family_series()
    w = family_sample_weights_by_alpha(fam, 0.0)
    assert np.allclose(w, 1.0)


def test_family_sample_weights_by_alpha_one_matches_inverse_family_size() -> None:
    fam = _mixed_family_series()
    w_alpha = family_sample_weights_by_alpha(fam, 1.0)
    w_scheme = family_sample_weights(fam, "inverse_family_size")
    assert np.array_equal(w_alpha, w_scheme)


def test_family_sample_weights_by_alpha_half_matches_sqrt_inverse_family_size() -> None:
    fam = _mixed_family_series()
    w_alpha = family_sample_weights_by_alpha(fam, 0.5)
    w_scheme = family_sample_weights(fam, "sqrt_inverse_family_size")
    assert np.allclose(w_alpha, w_scheme)


def test_family_sample_weights_by_alpha_is_monotonic_in_alpha_for_smallest_family() -> None:
    fam = _mixed_family_series()  # family "a" has n=1, the smallest.
    w0 = family_sample_weights_by_alpha(fam, 0.0)
    w25 = family_sample_weights_by_alpha(fam, 0.25)
    w50 = family_sample_weights_by_alpha(fam, 0.50)
    w75 = family_sample_weights_by_alpha(fam, 0.75)
    w100 = family_sample_weights_by_alpha(fam, 1.00)
    assert w0[0] < w25[0] < w50[0] < w75[0] < w100[0]


def test_family_sample_weights_by_alpha_normalizes_mean_to_one() -> None:
    fam = _mixed_family_series()
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = family_sample_weights_by_alpha(fam, alpha)
        assert w.mean() == pytest.approx(1.0)


def test_family_sample_weights_by_alpha_rejects_negative_alpha() -> None:
    with pytest.raises(ValueError):
        family_sample_weights_by_alpha(pd.Series(["a", "b"]), -0.1)


def test_family_sample_weights_by_alpha_rejects_missing_family() -> None:
    with pytest.raises(ValueError):
        family_sample_weights_by_alpha(pd.Series(["a", None]), 1.0)


def test_family_sample_weights_by_alpha_fold_only_uses_input_rows_only() -> None:
    """Weights must depend only on the family counts within the given input
    (e.g. one fold's training partition), never on some larger/global
    population -- calling on two different subsets with different family
    balance must give different weights for the same family label."""
    fold_a = pd.Series(["small"] * 2 + ["big"] * 8)
    fold_b = pd.Series(["small"] * 5 + ["big"] * 5)
    w_a = family_sample_weights_by_alpha(fold_a, 1.0)
    w_b = family_sample_weights_by_alpha(fold_b, 1.0)
    # "small" is rarer in fold_a (2 vs 8) than in fold_b (5 vs 5), so its
    # inverse-size weight must be strictly larger in fold_a.
    assert w_a[0] > w_b[0]


# ---------------------------------------------------------------------------
# Phase 2.3: weight-cap sensitivity
# ---------------------------------------------------------------------------


def test_family_sample_weights_by_alpha_cap_bounds_weights_and_renormalizes() -> None:
    fam = pd.Series(["tiny"] * 1 + ["huge"] * 99)
    uncapped = family_sample_weights_by_alpha(fam, 1.0)
    capped = family_sample_weights_by_alpha(fam, 1.0, cap=3.0)
    assert uncapped.max() > 3.0  # the uncapped case actually exceeds the cap (sanity)
    assert capped.max() <= 3.0 + 1e-9
    assert capped.min() >= 1.0 / 3.0 - 1e-9
    assert capped.mean() == pytest.approx(1.0)


def test_family_sample_weights_by_alpha_cap_is_deterministic() -> None:
    fam = pd.Series(["tiny"] * 1 + ["mid"] * 5 + ["huge"] * 50)
    first = family_sample_weights_by_alpha(fam, 0.75, cap=3.0)
    second = family_sample_weights_by_alpha(fam, 0.75, cap=3.0)
    assert np.array_equal(first, second)


def test_family_sample_weights_by_alpha_cap_noop_when_already_within_bound() -> None:
    fam = _mixed_family_series()
    uncapped = family_sample_weights_by_alpha(fam, 0.25)
    capped = family_sample_weights_by_alpha(fam, 0.25, cap=1000.0)
    assert np.allclose(uncapped, capped)


def test_family_sample_weights_by_alpha_rejects_cap_le_one() -> None:
    with pytest.raises(ValueError):
        family_sample_weights_by_alpha(pd.Series(["a", "b"]), 1.0, cap=1.0)


# ---------------------------------------------------------------------------
# Phase 2.3: GO/NO-GO gate arithmetic (pure function, synthetic inputs)
# ---------------------------------------------------------------------------


def _gate_result(
    macro: float,
    worst_q: float,
    pooled: float,
    lomfo_mae: float,
    families: tuple[str, ...] = ("f1", "f2", "f3"),
) -> dict:
    return {
        "macro_family_mae": macro,
        "worst_quartile_family_mae": worst_q,
        "sample_weighted_mae": pooled,
        "per_family_mae": {f: {"mae": macro, "n": 10} for f in families},
        "leave_one_major_family_out": {f: {"mae_mean": lomfo_mae} for f in families},
    }


def test_evaluate_family_weighting_gate_all_pass_is_go() -> None:
    baseline = _gate_result(macro=2.0, worst_q=3.0, pooled=1.5, lomfo_mae=1.0)
    candidate = _gate_result(macro=1.5, worst_q=2.5, pooled=1.5, lomfo_mae=1.0)
    gate = evaluate_family_weighting_gate(baseline, candidate, ["f1", "f2", "f3"], bootstrap_seed=0)
    assert gate["verdict"] == "GO"
    assert all(gate["criteria"].values())


def test_evaluate_family_weighting_gate_insufficient_macro_improvement_is_no_go() -> None:
    baseline = _gate_result(macro=2.0, worst_q=3.0, pooled=1.5, lomfo_mae=1.0)
    candidate = _gate_result(macro=1.99, worst_q=2.5, pooled=1.5, lomfo_mae=1.0)  # <5% improvement
    gate = evaluate_family_weighting_gate(baseline, candidate, ["f1", "f2", "f3"], bootstrap_seed=0)
    assert gate["criteria"]["macro_family_mae_improvement_ge_min_pct"] is False
    assert gate["verdict"] == "NO-GO"


def test_evaluate_family_weighting_gate_pooled_worse_beyond_tolerance_is_no_go() -> None:
    baseline = _gate_result(macro=2.0, worst_q=3.0, pooled=1.5, lomfo_mae=1.0)
    candidate = _gate_result(macro=1.5, worst_q=2.5, pooled=1.6, lomfo_mae=1.0)  # +0.10 > 0.05
    gate = evaluate_family_weighting_gate(baseline, candidate, ["f1", "f2", "f3"], bootstrap_seed=0)
    assert gate["criteria"]["pooled_mae_not_worse_by_gt_tolerance"] is False
    assert gate["verdict"] == "NO-GO"


def test_evaluate_family_weighting_gate_lomfo_majority_worsened_is_no_go() -> None:
    baseline = _gate_result(macro=2.0, worst_q=3.0, pooled=1.5, lomfo_mae=1.0)
    # +0.2 > 0.05 tolerance for all three families
    candidate = _gate_result(macro=1.5, worst_q=2.5, pooled=1.5, lomfo_mae=1.2)
    gate = evaluate_family_weighting_gate(baseline, candidate, ["f1", "f2", "f3"], bootstrap_seed=0)
    # all 3 major families worsened by > tolerance -> exceeds default max of 1
    assert gate["n_major_families_worsened_by_gt_tolerance"] == 3
    assert gate["criteria"]["lomfo_majority_not_worsened"] is False
    assert gate["verdict"] == "NO-GO"


def test_evaluate_family_weighting_gate_criterion_e_threshold_is_configurable() -> None:
    baseline = _gate_result(macro=2.0, worst_q=3.0, pooled=1.5, lomfo_mae=1.0)
    candidate = _gate_result(macro=1.5, worst_q=2.5, pooled=1.5, lomfo_mae=1.2)
    gate = evaluate_family_weighting_gate(
        baseline,
        candidate,
        ["f1", "f2", "f3"],
        lomfo_max_worsened_major_families=3,
        bootstrap_seed=0,
    )
    assert gate["criteria"]["lomfo_majority_not_worsened"] is True


# ---------------------------------------------------------------------------
# Phase 2.3: candidate selection determinism
# ---------------------------------------------------------------------------


def test_select_best_family_weighting_candidate_picks_lowest_macro_among_go() -> None:
    gate_results = {
        "alpha_025": {"verdict": "GO"},
        "alpha_050": {"verdict": "GO"},
        "alpha_075": {"verdict": "NO-GO"},
    }
    macro = {"alpha_025": 2.0, "alpha_050": 1.5, "alpha_075": 1.0}
    result = select_best_family_weighting_candidate(gate_results, macro)
    assert result["selected"] == "alpha_050"
    assert result["any_go"] is True
    assert result["go_candidates_ranked"] == ["alpha_050", "alpha_025"]


def test_select_best_family_weighting_candidate_falls_back_when_no_go() -> None:
    gate_results = {
        "alpha_025": {"verdict": "NO-GO"},
        "alpha_050": {"verdict": "NO-GO"},
    }
    macro = {"alpha_025": 2.0, "alpha_050": 1.5}
    result = select_best_family_weighting_candidate(gate_results, macro)
    assert result["any_go"] is False
    assert result["selected"] == "alpha_050"  # best-by-macro even though NO-GO


def test_select_best_family_weighting_candidate_breaks_ties_by_name() -> None:
    gate_results = {"b_candidate": {"verdict": "GO"}, "a_candidate": {"verdict": "GO"}}
    macro = {"b_candidate": 1.0, "a_candidate": 1.0}  # exact tie
    result = select_best_family_weighting_candidate(gate_results, macro)
    assert result["selected"] == "a_candidate"


def test_select_best_family_weighting_candidate_is_deterministic_across_calls() -> None:
    gate_results = {
        "x": {"verdict": "GO"},
        "y": {"verdict": "GO"},
        "z": {"verdict": "NO-GO"},
    }
    macro = {"x": 1.2, "y": 1.1, "z": 0.9}
    first = select_best_family_weighting_candidate(gate_results, macro)
    second = select_best_family_weighting_candidate(gate_results, macro)
    assert first == second


def test_select_best_family_weighting_candidate_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        select_best_family_weighting_candidate({}, {})


# ---------------------------------------------------------------------------
# Phase 2.6A: max_single_family_improvement_share
# ---------------------------------------------------------------------------


def test_max_single_family_improvement_share_all_improvement_from_one_family() -> None:
    baseline = {"a": {"mae": 2.0, "n": 5}, "b": {"mae": 1.0, "n": 5}, "c": {"mae": 1.0, "n": 5}}
    candidate = {"a": {"mae": 1.0, "n": 5}, "b": {"mae": 1.0, "n": 5}, "c": {"mae": 1.0, "n": 5}}
    result = max_single_family_improvement_share(baseline, candidate)
    assert result["max_family"] == "a"
    assert result["max_share"] == pytest.approx(1.0)


def test_max_single_family_improvement_share_spread_evenly() -> None:
    baseline = {"a": {"mae": 2.0, "n": 5}, "b": {"mae": 2.0, "n": 5}}
    candidate = {"a": {"mae": 1.0, "n": 5}, "b": {"mae": 1.0, "n": 5}}
    result = max_single_family_improvement_share(baseline, candidate)
    assert result["max_share"] == pytest.approx(0.5)


def test_max_single_family_improvement_share_zero_when_no_family_improves() -> None:
    baseline = {"a": {"mae": 1.0, "n": 5}, "b": {"mae": 1.0, "n": 5}}
    candidate = {"a": {"mae": 1.5, "n": 5}, "b": {"mae": 1.5, "n": 5}}
    result = max_single_family_improvement_share(baseline, candidate)
    assert result["max_family"] is None
    assert result["max_share"] == pytest.approx(0.0)


def test_max_single_family_improvement_share_requires_identical_family_sets() -> None:
    with pytest.raises(ValueError):
        max_single_family_improvement_share(
            {"a": {"mae": 1.0, "n": 1}}, {"b": {"mae": 1.0, "n": 1}}
        )


# ---------------------------------------------------------------------------
# Phase 2.6A: evaluate_censored_learning_gate
# ---------------------------------------------------------------------------


def _censored_gate_result(
    per_family: dict[str, float],
    pooled: float,
    lomfo: dict[str, float],
    n_per_family: int = 10,
) -> dict:
    macro = sum(per_family.values()) / len(per_family)
    worst_q = max(per_family.values())
    return {
        "macro_family_mae": macro,
        "worst_quartile_family_mae": worst_q,
        "sample_weighted_mae": pooled,
        "per_family_mae": {f: {"mae": v, "n": n_per_family} for f, v in per_family.items()},
        "leave_one_major_family_out": {f: {"mae_mean": v} for f, v in lomfo.items()},
    }


def test_evaluate_censored_learning_gate_all_pass_is_go() -> None:
    baseline = _censored_gate_result(
        {"f1": 2.0, "f2": 2.0, "f3": 2.0},
        pooled=1.5,
        lomfo={"f1": 1.0, "f2": 1.0, "f3": 1.0},
    )
    candidate = _censored_gate_result(
        {"f1": 1.5, "f2": 1.5, "f3": 1.5},
        pooled=1.45,
        lomfo={"f1": 1.0, "f2": 1.0, "f3": 1.0},
    )
    fold_stability = {"fraction_worse_than_baseline": 0.2}
    gate = evaluate_censored_learning_gate(
        baseline, candidate, ["f1", "f2", "f3"], fold_stability, bootstrap_seed=0
    )
    assert gate["verdict"] == "GO"
    assert all(gate["criteria"].values())


def test_evaluate_censored_learning_gate_fails_on_fold_driven_improvement() -> None:
    baseline = _censored_gate_result(
        {"f1": 2.0, "f2": 2.0, "f3": 2.0},
        pooled=1.5,
        lomfo={"f1": 1.0, "f2": 1.0, "f3": 1.0},
    )
    candidate = _censored_gate_result(
        {"f1": 1.5, "f2": 1.5, "f3": 1.5},
        pooled=1.45,
        lomfo={"f1": 1.0, "f2": 1.0, "f3": 1.0},
    )
    fold_stability = {"fraction_worse_than_baseline": 0.9}  # worse than baseline most of the time
    gate = evaluate_censored_learning_gate(
        baseline, candidate, ["f1", "f2", "f3"], fold_stability, bootstrap_seed=0
    )
    assert gate["not_one_fold_driven"] is False
    assert gate["criteria"]["not_one_fold_or_family_driven"] is False
    assert gate["verdict"] == "NO-GO"


def test_evaluate_censored_learning_gate_fails_on_family_driven_improvement() -> None:
    baseline = _censored_gate_result(
        {"f1": 4.0, "f2": 1.0, "f3": 1.0},
        pooled=1.5,
        lomfo={"f1": 1.0, "f2": 1.0, "f3": 1.0},
    )
    # All of the macro-family improvement comes from f1 alone.
    candidate = _censored_gate_result(
        {"f1": 1.0, "f2": 1.0, "f3": 1.0},
        pooled=1.45,
        lomfo={"f1": 1.0, "f2": 1.0, "f3": 1.0},
    )
    fold_stability = {"fraction_worse_than_baseline": 0.2}
    gate = evaluate_censored_learning_gate(
        baseline, candidate, ["f1", "f2", "f3"], fold_stability, bootstrap_seed=0
    )
    assert gate["not_one_family_driven"] is False
    assert gate["criteria"]["not_one_fold_or_family_driven"] is False
    assert gate["verdict"] == "NO-GO"


def test_evaluate_censored_learning_gate_lomfo_majority_worsened_is_no_go() -> None:
    baseline = _censored_gate_result(
        {"f1": 2.0, "f2": 2.0, "f3": 2.0},
        pooled=1.5,
        lomfo={"f1": 1.0, "f2": 1.0, "f3": 1.0},
    )
    candidate = _censored_gate_result(
        {"f1": 1.5, "f2": 1.5, "f3": 1.5},
        pooled=1.5,
        lomfo={"f1": 1.2, "f2": 1.2, "f3": 1.2},  # all 3 worsened by > 0.05
    )
    fold_stability = {"fraction_worse_than_baseline": 0.2}
    gate = evaluate_censored_learning_gate(
        baseline, candidate, ["f1", "f2", "f3"], fold_stability, bootstrap_seed=0
    )
    assert gate["criteria"]["lomfo_majority_not_worsened"] is False
    assert gate["verdict"] == "NO-GO"


def test_evaluate_censored_learning_gate_thresholds_are_configurable() -> None:
    baseline = _censored_gate_result(
        {"f1": 2.0, "f2": 2.0, "f3": 2.0},
        pooled=1.5,
        lomfo={"f1": 1.0, "f2": 1.0, "f3": 1.0},
    )
    candidate = _censored_gate_result(
        {"f1": 1.5, "f2": 1.5, "f3": 1.5},
        pooled=1.45,
        lomfo={"f1": 1.0, "f2": 1.0, "f3": 1.0},
    )
    fold_stability = {"fraction_worse_than_baseline": 0.9}
    gate = evaluate_censored_learning_gate(
        baseline,
        candidate,
        ["f1", "f2", "f3"],
        fold_stability,
        fold_driven_max_fraction_worse=0.95,
        bootstrap_seed=0,
    )
    assert gate["criteria"]["not_one_fold_or_family_driven"] is True
    assert gate["verdict"] == "GO"
