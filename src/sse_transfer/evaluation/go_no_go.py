"""N10 — GO/NO-GO gate, as a pure function of already-computed N2/N6/
N7/N8 results (CLAUDE.md "Keep evaluation independent from training"; this
project's established gate-as-pure-function convention, mirroring
`evaluation.family_robustness.evaluate_family_weighting_gate`). Never fits
a model, never reads a raw file -- callers assemble the inputs from N2's
cohort-size record, N6's run-authorization flag, N7's per-regime summary,
and N8's per-model AD results.

Criteria text is copied verbatim from artifacts/frozen/external_transfer_lock.json's
`predeclared_n10_go_no_go_criteria`, frozen before N6 opened any external
target.
"""

from __future__ import annotations

from typing import Any

GO_NO_GO_VERSION = "1.0"

STRICT_COHORT_MIN_N = 100


def evaluate_g1_external_cohort(strict_cohort_n: int) -> dict[str, Any]:
    passed = strict_cohort_n >= STRICT_COHORT_MIN_N
    return {
        "gate": "G1_external_cohort_valid_and_adequately_sized",
        "criterion": f"strict cohort n >= {STRICT_COHORT_MIN_N}",
        "strict_cohort_n": strict_cohort_n,
        "passed": passed,
    }


def evaluate_g2_cross_dataset_validity(
    *, n6_run_count: int, n6_authorized: bool, n6_lock_hash_matches: bool
) -> dict[str, Any]:
    passed = n6_run_count == 1 and n6_authorized and n6_lock_hash_matches
    return {
        "gate": "G2_cross_dataset_evaluation_technically_and_scientifically_valid",
        "criterion": "N6 ran exactly once (run_count=1), authorized=true, against the exact "
        "locked hash",
        "n6_run_count": n6_run_count,
        "n6_authorized": n6_authorized,
        "n6_lock_hash_matches": n6_lock_hash_matches,
        "passed": passed,
    }


def evaluate_g3_regime_comparison_clarity(
    per_regime_summary: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """G3 passes if the SAME regime is identified as most externally
    predictive on >=2 of the 4 statistics (median_abs_mae_gap
    ascending=better, spearman_rho/kendall_tau descending=better,
    top_model_selection_regret ascending=better).

    CAVEAT (found during independent post-hoc audit, not before): with
    exactly 3 regimes and 4 statistics -- this project's fixed, predeclared
    comparison -- the pigeonhole principle guarantees some regime wins at
    least 2 of 4 (if every regime won <=1, total wins would be <=3 < 4,
    a contradiction), so `passed` can never be False in this project's
    actual usage and this function alone has no real discriminating power
    here. It is retained rather than tightened post-hoc, per this project's
    no-criteria-change-after-seeing-results rule -- the pretest lock froze
    ">=2 of 4" before N6 opened external targets, and loosening OR
    tightening it now would itself be exactly the kind of after-the-fact
    adjustment the lock forbids. `majority_win_count` (2, 3, or 4) is the
    field that actually carries information: this project's real result is
    4/4 (unanimous), not a bare 2/4 minimum, and every report/manuscript
    passage should cite that count, not just the boolean `passed`."""
    regimes = sorted(per_regime_summary)
    best_by_stat: dict[str, str] = {}
    best_by_stat["median_abs_mae_gap"] = min(
        regimes, key=lambda r: per_regime_summary[r]["median_abs_mae_gap"]
    )
    best_by_stat["spearman_rho"] = max(regimes, key=lambda r: per_regime_summary[r]["spearman_rho"])
    best_by_stat["kendall_tau"] = max(regimes, key=lambda r: per_regime_summary[r]["kendall_tau"])
    best_by_stat["top_model_selection_regret"] = min(
        regimes, key=lambda r: per_regime_summary[r]["top_model_selection_regret"]
    )

    counts: dict[str, int] = {r: 0 for r in regimes}
    for winner in best_by_stat.values():
        counts[winner] += 1
    majority_regime, majority_count = max(counts.items(), key=lambda kv: kv[1])
    passed = majority_count >= 2

    return {
        "gate": "G3_validation_regime_comparison_gives_clear_reproducible_result",
        "criterion": "the same regime wins >=2 of the 4 statistics (median_abs_mae_gap, "
        "spearman_rho, kendall_tau, top_model_selection_regret)",
        "best_regime_by_statistic": best_by_stat,
        "win_counts_by_regime": counts,
        "majority_regime": majority_regime,
        "majority_win_count": majority_count,
        "passed": passed,
    }


def evaluate_g4_ad_risk_utility(
    *, n_models_positive_spearman: int, n_models_total: int, n_non_monotonic_steps_best_model: int
) -> dict[str, Any]:
    """Preferred, not required (never blocks GO on its own)."""
    majority_positive = n_models_positive_spearman >= (n_models_total / 2)
    monotonic_enough = n_non_monotonic_steps_best_model <= 1
    passed = majority_positive and monotonic_enough
    return {
        "gate": "G4_external_ad_risk_analysis_interpretable_utility",
        "criterion": "Spearman(AD,|error|) positive for >=half of models AND best model's "
        "risk-coverage has <=1 non-monotonic step",
        "n_models_positive_spearman": n_models_positive_spearman,
        "n_models_total": n_models_total,
        "n_non_monotonic_steps_best_model": n_non_monotonic_steps_best_model,
        "passed": passed,
        "requirement_level": "STRONGLY_PREFERRED_NOT_REQUIRED",
    }


def go_no_go_verdict(g1: dict, g2: dict, g3: dict, g4: dict) -> dict[str, Any]:
    """GO iff G1 AND G2 AND G3 all pass. G4 is reported but never
    blocks GO on its own, per the lock's `npj_go_rule`."""
    required_pass = g1["passed"] and g2["passed"] and g3["passed"]
    return {
        "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4},
        "required_gates_all_pass": required_pass,
        "g4_also_passed": g4["passed"],
        "verdict": "GO" if required_pass else "NO-GO",
    }
