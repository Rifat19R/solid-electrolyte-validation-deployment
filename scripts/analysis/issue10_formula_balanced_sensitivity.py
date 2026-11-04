#!/usr/bin/env python3
"""Reviewer remediation Issue 10: formula-balanced external-estimand
sensitivity and formula-cluster bootstrap.

Spec: reports/remediation/specs/issue10_formula_balanced_sensitivity.md (hashed
before this script was written; see reports/remediation/specs/HASHES.txt).

Pure post-hoc analysis over already-frozen artifacts:
  - artifacts/frozen/n6_external_transfer_primary.json (one-shot external predictions)
  - data/processed/external_cohort/hargreaves_strict.csv (cohort rows)
  - artifacts/historical/n4b_nested_internal_validation.json (regimes A/B/C)
  - artifacts/historical/n20b_formula_group_kfold.json (regime D)
  - artifacts/historical/n15_stable_model_sensitivity.json (six-model subset)
  - artifacts/historical/n7b_regime_utility_nested.json (row-weighted reference)

No model is fit, no new inference is run, no frozen artifact is modified,
and the frozen 113-row OBELiX official test split is never opened.

Run with: python scripts/analysis/issue10_formula_balanced_sensitivity.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.evaluation.regime_utility import (  # noqa: E402
    cluster_bootstrap_regime_pair_delta_ci,
    formula_balanced_mae,
    formula_cluster_ids,
    regime_summary,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

STRICT_COHORT_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_strict.csv"
N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
N4B_PATH = ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
N20B_PATH = ROOT / "artifacts" / "authoritative" / "n20b_formula_group_kfold_corrected.json"
N15_STABLE_PATH = (
    ROOT / "artifacts" / "authoritative" / "n15_artifacts" / "n15_stable_model_sensitivity.json"
)
N7B_PATH = ROOT / "artifacts" / "authoritative" / "n7b_regime_utility_nested_corrected.json"
PHASE1_FACTUAL_FIXES_PATH = (
    ROOT / "artifacts" / "authoritative" / "phase1_factual_fixes_corrected.json"
)
OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "issue10_formula_balanced_sensitivity_corrected.json"
REPORT_PATH = ROOT / "reports" / "rerun" / "issue10_formula_balanced_sensitivity.md"

BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
STATISTICS = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]
EXPECTED_N = 204


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def load_strict_cohort_formulas() -> list[str]:
    """Return the strict cohort's reduced_formula column, row order preserved.

    Only verifies the total row count against EXPECTED_N. Does NOT require
    (or assume) that formulas are unique -- duplication, if any, is measured
    separately via Counter in main() rather than treated as a precondition
    failure here, so this function (and the sensitivity it feeds) is correct
    for any future cohort, not only one that already happens to have zero
    repeated-formula groups.
    """
    with STRICT_COHORT_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != EXPECTED_N:
        raise RuntimeError(f"acs10: expected {EXPECTED_N} strict-cohort rows, got {len(rows)}")
    return [r["reduced_formula"] for r in rows]


def load_n6_external(
    expected_formula_counts: Counter[str],
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, list[str]]]:
    """Load frozen per-model external predictions, verified against the
    strict cohort by formula MULTISET (Counter), not by requiring every
    formula to be unique. `formula_list`/`residual_list` are read as
    parallel arrays straight from the frozen JSON and never reordered
    relative to each other, so a model's residuals stay row-aligned with
    its own formulas regardless of whether any formula repeats; the Counter
    check only guards against a model's prediction set silently covering a
    different population of formulas than the strict cohort file.
    """
    data = json.loads(N6_PATH.read_text(encoding="utf-8"))
    if data.get("run_count") != 1 or data.get("authorized") is not True:
        raise RuntimeError(
            "issue10: the n6 external-transfer record is not the authorized one-shot lock"
        )
    mae_by_model: dict[str, float] = {}
    residuals_by_model: dict[str, np.ndarray] = {}
    formula_order_by_model: dict[str, list[str]] = {}
    for model, payload in data["results"].items():
        strict = payload["strict"]
        if int(strict["n"]) != EXPECTED_N:
            raise RuntimeError(f"acs10: model {model!r} strict n != {EXPECTED_N}")
        formula_list = list(strict["predictions"]["reduced_formula"])
        residual_list = list(strict["predictions"]["residual"])
        if len(formula_list) != EXPECTED_N or len(residual_list) != EXPECTED_N:
            raise RuntimeError(
                f"acs10: model {model!r} has {len(formula_list)} formulas / "
                f"{len(residual_list)} residuals, expected {EXPECTED_N} each -- residual "
                "and formula arrays must stay row-aligned"
            )
        model_formula_counts = Counter(formula_list)
        if model_formula_counts != expected_formula_counts:
            raise RuntimeError(
                f"acs10: model {model!r} formula multiset does not match the strict "
                "cohort file's formula multiset -- predictions and formulas would misalign"
            )
        mae_by_model[model] = float(strict["mae"])
        residuals_by_model[model] = np.array(residual_list, dtype=float)
        formula_order_by_model[model] = formula_list
    return mae_by_model, residuals_by_model, formula_order_by_model


def formula_duplication_stats(formulas: list[str]) -> dict[str, int | bool]:
    """Counter-based duplication statistics over a cohort's reduced_formula
    column. Never uses `len(rows) - len(unique)` as the number of repeated
    GROUPS (that quantity is `n_duplicate_excess_rows`, a different count):
    a formula occurring 3 times contributes 1 to `n_repeated_formula_groups`
    but 2 to `n_duplicate_excess_rows`."""
    counts = Counter(formulas)
    n_unique = len(counts)
    n_repeated_groups = sum(1 for c in counts.values() if c > 1)
    n_duplicate_excess_rows = len(formulas) - n_unique
    return {
        "n_rows": len(formulas),
        "n_unique_reduced_formulas": n_unique,
        "n_repeated_formula_groups": n_repeated_groups,
        "n_duplicate_excess_rows": n_duplicate_excess_rows,
        "premise_holds_zero_repeated_formulas": n_repeated_groups == 0,
    }


def compare_pairwise_bootstrap_ci(mine: dict[str, float], ref: dict[str, float]) -> dict:
    """Compare one formula-cluster pairwise-bootstrap result against its
    pre-existing row-level reference. `point_delta` agreement (<=1e-9) is a
    required scientific invariant and raises if violated -- point estimates
    must agree regardless of resampling unit when the premise holds. CI
    endpoint agreement is NOT required: a cluster bootstrap over singleton
    formula groups (`rng.choice`) and a row-level bootstrap (`rng.integers`)
    are different pseudorandom draw sequences over the same distributional
    target, so a CI difference is sampling noise, not evidence of a
    population-weighting effect (see the frozen spec's decision rule).
    Returns the observed comparison rather than asserting it."""
    point_delta_diff = abs(mine["point_delta"] - ref["point_delta"])
    if point_delta_diff > 1e-9:
        raise RuntimeError(
            f"acs10: formula-cluster point_delta ({mine['point_delta']}) does not match "
            f"the pre-existing row-level reference point_delta ({ref['point_delta']}) to "
            "1e-9 -- this IS a required scientific invariant, unlike the CI endpoints"
        )
    ci_low_diff = abs(mine["ci_low"] - ref["ci_low"])
    ci_high_diff = abs(mine["ci_high"] - ref["ci_high"])
    return {
        "point_delta_matches_reference": True,
        "ci_matches_reference_exactly": bool(ci_low_diff <= 1e-9 and ci_high_diff <= 1e-9),
        "max_abs_ci_endpoint_difference": float(max(ci_low_diff, ci_high_diff)),
    }


def compute_formula_balanced(
    mae_by_model: dict[str, float],
    residuals_by_model: dict[str, np.ndarray],
    formula_order_by_model: dict[str, list[str]],
    *,
    premise_holds: bool,
) -> dict[str, dict[str, float]]:
    """Formula-balanced MAE is always computed explicitly (never skipped).
    The exact-equality assertion against the frozen row-weighted MAE is only
    a required invariant when the premise check found zero repeated-formula
    groups (per the spec's decision rule); if a future cohort has genuine
    repeats, a non-trivial difference here is an expected, reportable
    result, not a bug, so it is not treated as a RuntimeError in that case.
    """
    out: dict[str, dict[str, float]] = {}
    for model, residuals in residuals_by_model.items():
        formulas = formula_order_by_model[model]
        abs_residuals = np.abs(residuals)
        fb_mae = formula_balanced_mae(formulas, abs_residuals)
        row_mae = mae_by_model[model]
        if premise_holds and abs(fb_mae - row_mae) > 1e-9:
            raise RuntimeError(
                f"acs10: formula-balanced MAE ({fb_mae}) != frozen row-weighted MAE "
                f"({row_mae}) for model {model!r} even though the premise check found zero "
                "repeated-formula groups; this should be a mathematical identity"
            )
        out[model] = {"row_weighted_mae": row_mae, "formula_balanced_mae": fb_mae}
    return out


def formula_cluster_bootstrap_mae_ci(
    residuals_by_model: dict[str, np.ndarray],
    formula_order_by_model: dict[str, list[str]],
    *,
    seed: int,
    n_boot: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for model, residuals in residuals_by_model.items():
        abs_residuals = np.abs(residuals)
        formulas = formula_order_by_model[model]
        cluster_ids = formula_cluster_ids(formulas)
        unique_clusters = np.unique(cluster_ids)
        cluster_row_idx = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
        point = formula_balanced_mae(formulas, abs_residuals)
        boot = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            group_means = np.array([abs_residuals[cluster_row_idx[c]].mean() for c in sampled])
            boot[i] = float(np.mean(group_means))
        out[model] = {
            "point": point,
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
        }
    return out


def main() -> None:
    cohort_formulas = load_strict_cohort_formulas()
    dup_stats = formula_duplication_stats(cohort_formulas)
    n_unique_reduced_formulas = int(dup_stats["n_unique_reduced_formulas"])
    n_repeated_formula_groups = int(dup_stats["n_repeated_formula_groups"])
    n_duplicate_excess_rows = int(dup_stats["n_duplicate_excess_rows"])
    premise_holds = bool(dup_stats["premise_holds_zero_repeated_formulas"])

    mae_by_model, residuals_by_model, formula_order_by_model = load_n6_external(
        Counter(cohort_formulas)
    )

    fb = compute_formula_balanced(
        mae_by_model, residuals_by_model, formula_order_by_model, premise_holds=premise_holds
    )

    row_ranking = sorted(mae_by_model, key=lambda m: mae_by_model[m])
    fb_ranking = sorted(fb, key=lambda m: fb[m]["formula_balanced_mae"])
    ranking_identical = row_ranking == fb_ranking

    n4b = json.loads(N4B_PATH.read_text(encoding="utf-8"))
    n20b = json.loads(N20B_PATH.read_text(encoding="utf-8"))
    n15_stable = json.loads(N15_STABLE_PATH.read_text(encoding="utf-8"))
    n7b = json.loads(N7B_PATH.read_text(encoding="utf-8"))
    phase1_fixes = json.loads(PHASE1_FACTUAL_FIXES_PATH.read_text(encoding="utf-8"))
    # Regime D (formula-GroupKFold) was added post hoc, after N7b's bootstrap
    # infrastructure; its own row-level A-vs-D pairwise bootstrap lives here,
    # not in N7b (which only ever computed B/C vs A). Reused, not recomputed.
    d_minus_a_row_level_reference = phase1_fixes["item4_pairwise_diffs"]["formulaGKF_minus_A"]

    models_8 = sorted(mae_by_model)
    stable_models = n15_stable["retained_models"]

    internal_by_regime_8 = {
        "A_random_5fold_cv": {
            m: float(n4b["results"][m]["A_random_5fold_cv"]["pooled_mae"]) for m in models_8
        },
        "B_family_group_kfold": {
            m: float(n4b["results"][m]["B_family_group_kfold"]["pooled_mae"]) for m in models_8
        },
        "C_lomfo": {
            m: float(n4b["results"][m]["C_lomfo"]["pooled_mae_over_major_family_rows"])
            for m in models_8
        },
        "D_formula_group_kfold": {m: float(n20b["results"][m]["pooled_mae"]) for m in models_8},
    }

    fb_external_mae = {m: fb[m]["formula_balanced_mae"] for m in models_8}

    regime_summaries_8 = {
        regime: regime_summary(internal_by_regime_8[regime], fb_external_mae)
        for regime in internal_by_regime_8
    }
    regime_summaries_6 = {
        regime: regime_summary(
            {m: internal_by_regime_8[regime][m] for m in stable_models},
            {m: fb_external_mae[m] for m in stable_models},
        )
        for regime in internal_by_regime_8
    }

    # Row-weighted reference summaries, recomputed identically for a direct
    # side-by-side comparison in the report (not re-derived from N7b's own
    # bootstrap, which used a different, already-frozen seed draw sequence).
    row_external_mae = {m: mae_by_model[m] for m in models_8}
    regime_summaries_8_row = {
        regime: regime_summary(internal_by_regime_8[regime], row_external_mae)
        for regime in internal_by_regime_8
    }

    summaries_identical = all(
        regime_summaries_8[regime] == regime_summaries_8_row[regime]
        for regime in internal_by_regime_8
    )

    # Formula-cluster bootstrap: external MAE CIs.
    formula_cluster_mae_ci = formula_cluster_bootstrap_mae_ci(
        residuals_by_model, formula_order_by_model, seed=BOOTSTRAP_SEED, n_boot=N_BOOT
    )

    # Formula-cluster bootstrap: regime-utility pairwise deltas vs A, for B/C/D.
    pairwise_formula_cluster: dict[str, dict[str, dict]] = {}
    for regime, label in [
        ("B_family_group_kfold", "B"),
        ("C_lomfo", "C"),
        ("D_formula_group_kfold", "D"),
    ]:
        cluster_ids_ref = formula_cluster_ids(formula_order_by_model[models_8[0]])
        pairwise_formula_cluster[f"{label}_minus_A"] = {
            stat: cluster_bootstrap_regime_pair_delta_ci(
                internal_by_regime_8["A_random_5fold_cv"],
                internal_by_regime_8[regime],
                {m: residuals_by_model[m] for m in models_8},
                cluster_ids_ref,
                statistic=stat,
                seed=BOOTSTRAP_SEED,
                n_boot=N_BOOT,
            )
            for stat in STATISTICS
        }

    # Verify, not assume: the formula-cluster pairwise point_delta is a
    # required scientific invariant (per the spec's decision rule, point
    # estimates must agree regardless of resampling unit when the premise
    # holds) and is hard-checked to 1e-9. CI endpoints are NOT required to
    # match exactly -- the spec explicitly notes that a cluster bootstrap
    # over singleton groups (rng.choice) and a row-level bootstrap
    # (rng.integers) are different pseudorandom draw sequences over the same
    # distributional target, so any CI difference is sampling noise, not a
    # population-weighting effect. CI agreement is therefore computed and
    # reported, not asserted.
    row_level_reference_by_pair = {
        "B_minus_A": n7b["pairwise_bootstrap_ci"]["B_minus_A"],
        "C_minus_A": n7b["pairwise_bootstrap_ci"]["C_minus_A"],
        "D_minus_A": d_minus_a_row_level_reference,
    }
    pairwise_bootstrap_ci_comparison: dict[str, dict[str, dict]] = {}
    for pair, ref_by_stat in row_level_reference_by_pair.items():
        pairwise_bootstrap_ci_comparison[pair] = {}
        for stat in STATISTICS:
            mine = pairwise_formula_cluster[pair][stat]
            ref = ref_by_stat[stat]
            try:
                pairwise_bootstrap_ci_comparison[pair][stat] = compare_pairwise_bootstrap_ci(
                    mine, ref
                )
            except RuntimeError as exc:
                raise RuntimeError(f"acs10: formula-cluster {pair}/{stat}: {exc}") from exc
    pairwise_row_level_reference = row_level_reference_by_pair

    output = {
        "phase": "acs_issue10_formula_balanced_sensitivity",
        "spec_path": "reports/remediation/specs/issue10_formula_balanced_sensitivity.md",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "n6_source": "artifacts/frozen/n6_external_transfer_primary.json",
        "strict_cohort_source": str(STRICT_COHORT_PATH.relative_to(ROOT)),
        "n_strict_cohort_rows": EXPECTED_N,
        "n_unique_reduced_formulas": n_unique_reduced_formulas,
        "n_repeated_formula_groups": n_repeated_formula_groups,
        "n_duplicate_excess_rows": n_duplicate_excess_rows,
        "premise_holds_zero_repeated_formulas": premise_holds,
        "per_model_row_vs_formula_balanced_mae": fb,
        "row_weighted_ranking_by_mae": row_ranking,
        "formula_balanced_ranking_by_mae": fb_ranking,
        "ranking_identical": ranking_identical,
        "regime_summaries_formula_balanced_8_model": regime_summaries_8,
        "regime_summaries_row_weighted_8_model_reference": regime_summaries_8_row,
        "regime_utility_summaries_identical_to_row_weighted": summaries_identical,
        "regime_summaries_formula_balanced_6_model_sensitivity": regime_summaries_6,
        "stable_models_6": stable_models,
        "formula_cluster_bootstrap_external_mae_ci": formula_cluster_mae_ci,
        "pairwise_bootstrap_ci_formula_cluster_level": pairwise_formula_cluster,
        "pairwise_bootstrap_ci_row_level_reference": pairwise_row_level_reference,
        "pairwise_bootstrap_ci_row_level_reference_sources": {
            "B_minus_A": (
                "artifacts/historical/n7b_regime_utility_nested.json:pairwise_bootstrap_ci"
            ),
            "C_minus_A": (
                "artifacts/historical/n7b_regime_utility_nested.json:pairwise_bootstrap_ci"
            ),
            "D_minus_A": "artifacts/historical/phase1_factual_fixes.json:item4_pairwise_diffs."
            "formulaGKF_minus_A",
        },
        "pairwise_bootstrap_ci_comparison_vs_row_level_reference": pairwise_bootstrap_ci_comparison,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_boot": N_BOOT,
        "decision": {
            "external_model_ordering_changed": not ranking_identical,
            "regime_utility_point_estimates_changed": not summaries_identical,
            "central_claim_affected": False
            if (premise_holds and ranking_identical and summaries_identical)
            else "SEE_REPORT_MANUAL_REVIEW_REQUIRED",
        },
    }
    guard_output(OUTPUT_PATH).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    print(f"premise_holds_zero_repeated_formulas={premise_holds}")
    print(
        f"ranking_identical={ranking_identical}, regime_summaries_identical={summaries_identical}"
    )

    lines = [
        "# Issue 10 — Formula-balanced external-estimand sensitivity",
        "",
        "**Spec:** `reports/remediation/specs/issue10_formula_balanced_sensitivity.md`.",
        "",
        f"**Premise check:** the strict external cohort has {EXPECTED_N} rows, "
        f"{n_unique_reduced_formulas} unique `reduced_formula` values, "
        f"{n_repeated_formula_groups} repeated-formula groups (formula identities with "
        f"count > 1) and {n_duplicate_excess_rows} duplicate-excess rows "
        f"(total rows minus unique formulas). "
        + (
            "Zero repeated-formula groups: the cohort-construction pipeline "
            "(`aggregate_by_formula`, `src/sse_transfer/data/hargreaves_cohort.py`) already "
            "collapses the 213 pre-aggregation retained rows to one aggregated formula-level "
            "record per unique reduced formula (an unweighted per-group median target) before "
            "N6 ever reads a target value, so the primary (row-weighted) and formula-balanced "
            "estimands coincide exactly for this frozen cohort."
            if premise_holds
            else "Repeated-formula groups exist: the sensitivity below is a genuine, "
            "non-trivial reweighting for this run; see per-model table."
        ),
        "",
        "## Row-weighted vs. formula-balanced external MAE",
        "",
        "| Model | Row-weighted MAE | Formula-balanced MAE | Identical |",
        "| --- | --- | --- | --- |",
    ]
    for model in models_8:
        r = fb[model]
        row_mae = r["row_weighted_mae"]
        fb_mae = r["formula_balanced_mae"]
        identical = "Yes" if abs(row_mae - fb_mae) < 1e-9 else "No"
        lines.append(f"| {model} | {row_mae:.4f} | {fb_mae:.4f} | {identical} |")
    lines.append("")
    lines.append(f"**Ranking identical to row-weighted ranking:** {ranking_identical}")
    lines.append(f"**Row-weighted ranking:** {', '.join(row_ranking)}")
    lines.append(f"**Formula-balanced ranking:** {', '.join(fb_ranking)}")
    lines.append("")
    lines.append("## Regime-utility statistics against formula-balanced external MAE (8-model)")
    lines.append("")
    lines.append("| Regime | Median |gap| | Spearman rho | Kendall tau | Regret |")
    lines.append("| --- | --- | --- | --- | --- |")
    regime_display = {
        "A_random_5fold_cv": "A",
        "B_family_group_kfold": "B",
        "C_lomfo": "C",
        "D_formula_group_kfold": "D",
    }
    for regime, label in regime_display.items():
        s = regime_summaries_8[regime]
        lines.append(
            f"| {label} | {s['median_abs_mae_gap']:.4f} | {s['spearman_rho']:.4f} | "
            f"{s['kendall_tau']:.4f} | {s['top_model_selection_regret']:.4f} |"
        )
    lines.append("")
    lines.append(f"**Identical to row-weighted regime-utility statistics:** {summaries_identical}")
    lines.append("")
    lines.append("## Formula-cluster bootstrap: external MAE 95% CI (vs. row-level reference)")
    lines.append("")
    lines.append("| Model | Point | Formula-cluster-bootstrap CI |")
    lines.append("| --- | --- | --- |")
    for model in models_8:
        c = formula_cluster_mae_ci[model]
        lines.append(f"| {model} | {c['point']:.4f} | [{c['ci_low']:.4f}, {c['ci_high']:.4f}] |")
    lines.append("")
    lines.append(
        "## Formula-cluster vs. row-level pairwise regime-utility bootstrap (B/C/D minus A)"
    )
    lines.append("")
    lines.append(
        "Point-delta equality (<=1e-9) is a required scientific invariant and is hard-checked "
        "above (the script raises if violated). CI-endpoint equality is NOT required -- the "
        "spec notes that singleton-cluster resampling (`rng.choice`) and row-level resampling "
        "(`rng.integers`) are different pseudorandom draw sequences over the same "
        "distributional target, so any CI difference is sampling noise, not evidence of a "
        "population-weighting effect. It is therefore only computed and reported here, not "
        "asserted."
    )
    lines.append("")
    lines.append(
        "| Pair | Statistic | Point-delta matches | CI matches exactly | Max abs CI endpoint diff |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for pair, by_stat in pairwise_bootstrap_ci_comparison.items():
        for stat, cmp in by_stat.items():
            lines.append(
                f"| {pair} | {stat} | {cmp['point_delta_matches_reference']} | "
                f"{cmp['ci_matches_reference_exactly']} | "
                f"{cmp['max_abs_ci_endpoint_difference']:.2e} |"
            )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    claim_survives = premise_holds and ranking_identical and summaries_identical
    claim_text = "No" if claim_survives else "REQUIRES MANUAL REVIEW"
    lines.append(
        f"- External model ordering changed: {not ranking_identical}\n"
        f"- Regime-utility point estimates changed: {not summaries_identical}\n"
        f"- Central claim affected: {claim_text}"
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_PATH).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
