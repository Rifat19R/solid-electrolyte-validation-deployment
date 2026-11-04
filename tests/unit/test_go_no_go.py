from __future__ import annotations

from sse_transfer.evaluation.go_no_go import (
    STRICT_COHORT_MIN_N,
    evaluate_g1_external_cohort,
    evaluate_g2_cross_dataset_validity,
    evaluate_g3_regime_comparison_clarity,
    evaluate_g4_ad_risk_utility,
    go_no_go_verdict,
)


def test_g1_passes_at_threshold() -> None:
    g1 = evaluate_g1_external_cohort(STRICT_COHORT_MIN_N)
    assert g1["passed"] is True


def test_g1_fails_below_threshold() -> None:
    g1 = evaluate_g1_external_cohort(STRICT_COHORT_MIN_N - 1)
    assert g1["passed"] is False


def test_g2_passes_when_authorized_single_run_matching_lock() -> None:
    g2 = evaluate_g2_cross_dataset_validity(
        n6_run_count=1, n6_authorized=True, n6_lock_hash_matches=True
    )
    assert g2["passed"] is True


def test_g2_fails_on_unauthorized_rerun() -> None:
    g2 = evaluate_g2_cross_dataset_validity(
        n6_run_count=2, n6_authorized=False, n6_lock_hash_matches=True
    )
    assert g2["passed"] is False


def test_g2_fails_on_lock_mismatch() -> None:
    g2 = evaluate_g2_cross_dataset_validity(
        n6_run_count=1, n6_authorized=True, n6_lock_hash_matches=False
    )
    assert g2["passed"] is False


def test_g3_passes_when_one_regime_wins_majority_of_statistics() -> None:
    summary = {
        "A": {
            "median_abs_mae_gap": 0.1,
            "spearman_rho": 0.9,
            "kendall_tau": 0.8,
            "top_model_selection_regret": 0.0,
        },
        "B": {
            "median_abs_mae_gap": 0.5,
            "spearman_rho": 0.1,
            "kendall_tau": 0.1,
            "top_model_selection_regret": 0.3,
        },
    }
    g3 = evaluate_g3_regime_comparison_clarity(summary)
    assert g3["passed"] is True
    assert g3["majority_regime"] == "A"
    assert g3["majority_win_count"] == 4


def test_g3_bare_minimum_two_of_four_win_still_passes() -> None:
    """With exactly 3 regimes and 4 statistics, the pigeonhole principle
    guarantees some regime wins >=2 (if every regime won <=1, total wins
    would be <=3 < 4, a contradiction) -- so `passed=False` is
    unreachable for this project's actual 3-regime comparison, and no
    test here can construct a genuinely failing 3-regime case (see the
    CAVEAT in `evaluate_g3_regime_comparison_clarity`'s docstring). This
    test instead documents the bare-minimum passing case (a 2-of-4 win,
    as opposed to test_g3_matches_real_n7_numbers_random_cv_wins_all_four's
    4-of-4 unanimous win) so the two passing shapes stay distinguishable."""
    summary = {
        "A": {
            "median_abs_mae_gap": 0.1,  # A wins gap
            "spearman_rho": 0.1,
            "kendall_tau": 0.1,
            "top_model_selection_regret": 0.9,
        },
        "B": {
            "median_abs_mae_gap": 0.5,
            "spearman_rho": 0.9,  # B wins spearman
            "kendall_tau": 0.1,
            "top_model_selection_regret": 0.9,
        },
        "C": {
            "median_abs_mae_gap": 0.5,
            "spearman_rho": 0.1,
            "kendall_tau": 0.9,  # C wins kendall
            "top_model_selection_regret": 0.1,  # C wins regret too -> C has 2
        },
    }
    g3 = evaluate_g3_regime_comparison_clarity(summary)
    assert g3["passed"] is True
    assert g3["majority_regime"] == "C"
    assert g3["majority_win_count"] == 2


def test_g3_passed_is_structurally_guaranteed_with_three_regimes() -> None:
    """Meta-test documenting the pigeonhole fact directly: for ANY set of
    per-regime statistic values across exactly 3 regimes and 4 statistics,
    `passed` is always True. Randomized to make the guarantee concrete
    rather than relying on one hand-picked example."""
    import random

    rng = random.Random(20260921)
    for _ in range(200):
        summary = {
            r: {
                "median_abs_mae_gap": rng.uniform(0, 1),
                "spearman_rho": rng.uniform(-1, 1),
                "kendall_tau": rng.uniform(-1, 1),
                "top_model_selection_regret": rng.uniform(0, 1),
            }
            for r in ("A", "B", "C")
        }
        g3 = evaluate_g3_regime_comparison_clarity(summary)
        assert g3["passed"] is True


def test_g3_matches_real_n7_numbers_random_cv_wins_all_four() -> None:
    # Actual frozen N7 per-regime summary (artifacts/historical/n7_regime_utility.json)
    summary = {
        "A_random_5fold_cv": {
            "median_abs_mae_gap": 0.40391290796208346,
            "spearman_rho": 0.5952380952380953,
            "kendall_tau": 0.4999999999999999,
            "top_model_selection_regret": 0.0,
        },
        "B_family_group_kfold": {
            "median_abs_mae_gap": 0.5512956388465935,
            "spearman_rho": 0.28571428571428575,
            "kendall_tau": 0.21428571428571427,
            "top_model_selection_regret": 0.14706094536605985,
        },
        "C_lomfo": {
            "median_abs_mae_gap": 0.4097676806206274,
            "spearman_rho": -0.023809523809523815,
            "kendall_tau": -0.07142857142857142,
            "top_model_selection_regret": 0.14706094536605985,
        },
    }
    g3 = evaluate_g3_regime_comparison_clarity(summary)
    assert g3["passed"] is True
    assert g3["majority_regime"] == "A_random_5fold_cv"
    assert g3["majority_win_count"] == 4


def test_g4_passes_with_majority_positive_and_monotonic() -> None:
    g4 = evaluate_g4_ad_risk_utility(
        n_models_positive_spearman=8, n_models_total=8, n_non_monotonic_steps_best_model=0
    )
    assert g4["passed"] is True


def test_g4_fails_with_minority_positive() -> None:
    g4 = evaluate_g4_ad_risk_utility(
        n_models_positive_spearman=2, n_models_total=8, n_non_monotonic_steps_best_model=0
    )
    assert g4["passed"] is False


def test_g4_fails_with_too_many_non_monotonic_steps() -> None:
    g4 = evaluate_g4_ad_risk_utility(
        n_models_positive_spearman=8, n_models_total=8, n_non_monotonic_steps_best_model=2
    )
    assert g4["passed"] is False


def test_go_when_g1_g2_g3_pass_regardless_of_g4() -> None:
    g1 = {"passed": True}
    g2 = {"passed": True}
    g3 = {"passed": True}
    g4_fail = {"passed": False}
    verdict = go_no_go_verdict(g1, g2, g3, g4_fail)
    assert verdict["verdict"] == "GO"
    assert verdict["g4_also_passed"] is False


def test_no_go_when_any_required_gate_fails() -> None:
    g1 = {"passed": True}
    g2 = {"passed": True}
    g3 = {"passed": False}
    g4 = {"passed": True}
    verdict = go_no_go_verdict(g1, g2, g3, g4)
    assert verdict["verdict"] == "NO-GO"
