#!/usr/bin/env python3
"""Corrective 01, Issue 10: formula-balanced external-estimand sensitivity,
recomputed against the canonical corrected 8-model view.

This is a SEPARATE script from scripts/analysis/issue10_formula_balanced_sensitivity.py,
not a monkeypatch of it, because that script's `load_n6_external` hard-asserts
`data["authorized"] is True` against whatever N6 artifact it is pointed at --
a correctness guard against treating a non-one-shot artifact as the primary
one-shot result. The corrective N6 view
(artifacts/authoritative/n6_external_transfer_corrected.json)
is intentionally marked "authorized": "N/A_CORRECTIVE_NON_PRIMARY", so that
guard correctly refuses it (see reports/remediation/corrective01_phase5_old_vs_corrected.md
section 9). This script performs the IDENTICAL computation (same functions,
same formulas, same bootstrap seed/count) against the corrected inputs,
with the authorization check relaxed to accept the corrective marker
explicitly rather than bypassed silently. It does not call, modify, or
rerun scripts/analysis/issue10_formula_balanced_sensitivity.py or touch the
historical artifacts/historical/issue10_formula_balanced_sensitivity.json. The
underlying historical one-shot N6 lock (artifacts/frozen/n6_external_transfer_primary.json,
run_count=1, authorized=true) is never read, touched, or re-authorized by
this script.

Run with: python scripts/analysis/corrective01_issue10_formula_balanced_sensitivity.py
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

CORR = ROOT / "artifacts" / "authoritative"
RERUN = ROOT / "artifacts" / "rerun"  # outputs; the authoritative record is never rewritten
STRICT_COHORT_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_strict.csv"
N6_PATH = CORR / "n6_external_transfer_corrected.json"
N4B_PATH = CORR / "n4b_nested_internal_validation_corrected.json"
N20B_PATH = CORR / "n20b_formula_group_kfold_corrected.json"
# Only `retained_models` (the six-model set) is read; identical in the
# pre-correction file the authoritative run used and in the corrected file.
N15_STABLE_PATH = CORR / "n15_artifacts" / "n15_stable_model_sensitivity.json"
N7B_PATH = CORR / "n7b_regime_utility_nested_corrected.json"
PHASE1_FACTUAL_FIXES_PATH = CORR / "phase1_factual_fixes_corrected.json"
OUTPUT_PATH = RERUN / "issue10_formula_balanced_sensitivity_corrected.json"
REPORT_PATH = (
    ROOT
    / "reports"
    / "rerun"
    / "corrective01_reports"
    / "issue10_formula_balanced_sensitivity_corrected.md"
)

BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
STATISTICS = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]
EXPECTED_N = 204


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def load_strict_cohort_formulas() -> list[str]:
    with STRICT_COHORT_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != EXPECTED_N:
        raise RuntimeError(
            f"corrective01_issue10: expected {EXPECTED_N} strict-cohort rows, got {len(rows)}"
        )
    return [r["reduced_formula"] for r in rows]


def load_n6_external_corrected(
    expected_formula_counts: Counter[str],
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, list[str]]]:
    data = json.loads(N6_PATH.read_text(encoding="utf-8"))
    if data.get("authorized") != "N/A_CORRECTIVE_NON_PRIMARY":
        raise RuntimeError(
            "corrective01_issue10: n6_external_transfer_corrected.json does not carry the "
            "expected 'N/A_CORRECTIVE_NON_PRIMARY' marker -- refusing to treat an unexpected "
            "artifact as the corrective external-transfer view"
        )
    mae_by_model: dict[str, float] = {}
    residuals_by_model: dict[str, np.ndarray] = {}
    formula_order_by_model: dict[str, list[str]] = {}
    for model, payload in data["results"].items():
        strict = payload["strict"]
        if int(strict["n"]) != EXPECTED_N:
            raise RuntimeError(f"corrective01_issue10: model {model!r} strict n != {EXPECTED_N}")
        formula_list = list(strict["predictions"]["reduced_formula"])
        residual_list = list(strict["predictions"]["residual"])
        if len(formula_list) != EXPECTED_N or len(residual_list) != EXPECTED_N:
            raise RuntimeError(
                f"corrective01_issue10: model {model!r} formula/residual length mismatch"
            )
        model_formula_counts = Counter(formula_list)
        if model_formula_counts != expected_formula_counts:
            raise RuntimeError(
                f"corrective01_issue10: model {model!r} formula multiset does not match the "
                "strict cohort file's formula multiset"
            )
        mae_by_model[model] = float(strict["mae"])
        residuals_by_model[model] = np.array(residual_list, dtype=float)
        formula_order_by_model[model] = formula_list
    return mae_by_model, residuals_by_model, formula_order_by_model


def formula_duplication_stats(formulas: list[str]) -> dict[str, int | bool]:
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


def compute_formula_balanced(
    mae_by_model: dict[str, float],
    residuals_by_model: dict[str, np.ndarray],
    formula_order_by_model: dict[str, list[str]],
    *,
    premise_holds: bool,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model, residuals in residuals_by_model.items():
        formulas = formula_order_by_model[model]
        abs_residuals = np.abs(residuals)
        fb_mae = formula_balanced_mae(formulas, abs_residuals)
        row_mae = mae_by_model[model]
        if premise_holds and abs(fb_mae - row_mae) > 1e-9:
            raise RuntimeError(
                f"corrective01_issue10: formula-balanced MAE ({fb_mae}) != frozen "
                f"row-weighted MAE ({row_mae}) for model {model!r}"
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

    mae_by_model, residuals_by_model, formula_order_by_model = load_n6_external_corrected(
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

    row_external_mae = {m: mae_by_model[m] for m in models_8}
    regime_summaries_8_row = {
        regime: regime_summary(internal_by_regime_8[regime], row_external_mae)
        for regime in internal_by_regime_8
    }

    summaries_identical = all(
        regime_summaries_8[regime] == regime_summaries_8_row[regime]
        for regime in internal_by_regime_8
    )

    formula_cluster_mae_ci = formula_cluster_bootstrap_mae_ci(
        residuals_by_model, formula_order_by_model, seed=BOOTSTRAP_SEED, n_boot=N_BOOT
    )

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

    row_level_reference_by_pair = {
        "B_minus_A": n7b["pairwise_bootstrap_ci"]["B_minus_A"],
        "C_minus_A": n7b["pairwise_bootstrap_ci"]["C_minus_A"],
        "D_minus_A": d_minus_a_row_level_reference,
    }

    def compare_pairwise_bootstrap_ci(mine: dict[str, float], ref: dict[str, float]) -> dict:
        point_delta_diff = abs(mine["point_delta"] - ref["point_delta"])
        if point_delta_diff > 1e-9:
            raise RuntimeError(
                f"corrective01_issue10: formula-cluster point_delta ({mine['point_delta']}) "
                f"does not match reference point_delta ({ref['point_delta']}) to 1e-9"
            )
        ci_low_diff = abs(mine["ci_low"] - ref["ci_low"])
        ci_high_diff = abs(mine["ci_high"] - ref["ci_high"])
        return {
            "point_delta_matches_reference": True,
            "ci_matches_reference_exactly": bool(ci_low_diff <= 1e-9 and ci_high_diff <= 1e-9),
            "max_abs_ci_endpoint_difference": float(max(ci_low_diff, ci_high_diff)),
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
                raise RuntimeError(f"corrective01_issue10: {pair}/{stat}: {exc}") from exc

    output = {
        "phase": "corrective01_issue10_formula_balanced_sensitivity",
        "spec_path": (
            "reports/remediation/specs/corrective01_pretrained_element_embedding_bugfix.md"
        ),
        "historical_spec_path": "reports/remediation/specs/issue10_formula_balanced_sensitivity.md",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "n6_source": str(N6_PATH.relative_to(ROOT)),
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
        "pairwise_bootstrap_ci_row_level_reference": row_level_reference_by_pair,
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
        "# Corrective 01 — Issue 10 (formula-balanced external-estimand sensitivity), "
        "corrected 8-model view",
        "",
        "**Spec:** `reports/remediation/specs/corrective01_pretrained_element_embedding_bugfix.md` "
        "(historical analysis spec: "
        "`reports/remediation/specs/issue10_formula_balanced_sensitivity.md`).",
        "",
        "This recomputes Issue 10 against the corrected `pretrained_element_embedding` "
        "model via a new script "
        "(`scripts/analysis/corrective01_issue10_formula_balanced_sensitivity.py`) "
        "rather than rerunning `scripts/analysis/issue10_formula_balanced_sensitivity.py`, "
        "because that "
        "script's authorization guard correctly refuses to treat the corrective, "
        "non-primary N6 view as the authorized one-shot lock. The historical "
        "`artifacts/historical/issue10_formula_balanced_sensitivity.json` is untouched. The "
        "original "
        "one-shot external-transfer lock (`artifacts/frozen/n6_external_transfer_primary.json`, "
        "`run_count=1`, `authorized=true`) is not read or modified by this script.",
        "",
        f"**Premise check:** {EXPECTED_N} strict-cohort rows, {n_unique_reduced_formulas} unique "
        f"reduced formulas, {n_repeated_formula_groups} repeated-formula groups, "
        f"{n_duplicate_excess_rows} duplicate-excess rows. Premise holds: {premise_holds} "
        "(unchanged from the historical analysis -- cohort construction is untouched by this "
        "corrective work).",
        "",
        "## Row-weighted vs. formula-balanced external MAE (corrected 8-model view)",
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
    lines.append("## Formula-cluster bootstrap: external MAE 95% CI")
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
