#!/usr/bin/env python3
"""Reviewer remediation Issue 09, Variant A: exclude __MISSING__ rows from
evaluation (cheap, no rerun).

Spec: reports/remediation/specs/issue09_missing_family_sensitivity.md (hashed
before this script was written).

Pure post-hoc recombination of already-frozen per-family MAE, exactly like
Issue 03's common-support control, dropping the "__MISSING__" entry instead
of restricting to major families. No model is fit.

Run with: python scripts/analysis/issue09a_missing_family_excluded.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.evaluation.regime_utility import (  # noqa: E402
    bootstrap_regime_pair_delta_ci,
    regime_summary,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

N4B_PATH = ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "issue09a_missing_family_excluded_corrected.json"
REPORT_PATH = ROOT / "reports" / "rerun" / "issue09a_missing_family_excluded.md"

REGIMES = ["A_random_5fold_cv", "B_family_group_kfold", "C_lomfo"]
REGIME_LABELS = {"A_random_5fold_cv": "A", "B_family_group_kfold": "B", "C_lomfo": "C"}
MISSING_FAMILY = "__MISSING__"
BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
STATISTICS = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def pooled_mae_excluding_missing(per_family_mae: dict[str, dict[str, float]]) -> tuple[float, int]:
    total_n = 0
    total_weighted = 0.0
    for fam, entry in per_family_mae.items():
        if fam == MISSING_FAMILY:
            continue
        total_n += entry["n"]
        total_weighted += entry["mae"] * entry["n"]
    return total_weighted / total_n, total_n


def load_n6_external() -> tuple[dict[str, float], dict[str, np.ndarray]]:
    data = json.loads(N6_PATH.read_text(encoding="utf-8"))
    mae_by_model: dict[str, float] = {}
    residuals_by_model: dict[str, np.ndarray] = {}
    for model, payload in data["results"].items():
        strict = payload["strict"]
        mae_by_model[model] = float(strict["mae"])
        residuals_by_model[model] = np.array(strict["predictions"]["residual"], dtype=float)
    return mae_by_model, residuals_by_model


def main() -> None:
    n4b = json.loads(N4B_PATH.read_text(encoding="utf-8"))
    models = sorted(n4b["results"].keys())
    external_mae, external_residuals = load_n6_external()

    full_internal: dict[str, dict[str, float]] = {r: {} for r in REGIMES}
    excl_internal: dict[str, dict[str, float]] = {r: {} for r in REGIMES}
    excl_n_check_by_regime: dict[str, set[int]] = {r: set() for r in REGIMES}

    for regime in REGIMES:
        for model in models:
            payload = n4b["results"][model][regime]
            mae_key = (
                "pooled_mae" if "pooled_mae" in payload else "pooled_mae_over_major_family_rows"
            )
            full_internal[regime][model] = float(payload[mae_key])
            excl_mae, excl_n = pooled_mae_excluding_missing(payload["per_family_mae"])
            excl_internal[regime][model] = excl_mae
            excl_n_check_by_regime[regime].add(excl_n)

    excl_n_by_regime: dict[str, int] = {}
    for regime, ns in excl_n_check_by_regime.items():
        if len(ns) != 1:
            raise RuntimeError(
                f"acs09a: n after exclusion not consistent across models for {regime}: {ns}"
            )
        excl_n_by_regime[regime] = ns.pop()
    # A and B evaluate the full 449-row population; excluding __MISSING__ gives 431 for both.
    if excl_n_by_regime["A_random_5fold_cv"] != excl_n_by_regime["B_family_group_kfold"]:
        raise RuntimeError(f"acs09a: A/B population mismatch after exclusion: {excl_n_by_regime}")
    excl_n = excl_n_by_regime["A_random_5fold_cv"]
    # C (LOMFO) only ever covers major families (396, dropping to 378 once
    # __MISSING__ -- itself a major family by count -- is excluded); this is
    # expected and reported separately, not an inconsistency.
    excl_n_lomfo = excl_n_by_regime["C_lomfo"]

    full_summary = {r: regime_summary(full_internal[r], external_mae) for r in REGIMES}
    excl_summary = {r: regime_summary(excl_internal[r], external_mae) for r in REGIMES}

    pairwise_full: dict[str, dict] = {}
    pairwise_excl: dict[str, dict] = {}
    for regime in ("B_family_group_kfold", "C_lomfo"):
        label = REGIME_LABELS[regime]
        pairwise_full[f"{label}_minus_A"] = {
            stat: bootstrap_regime_pair_delta_ci(
                full_internal["A_random_5fold_cv"],
                full_internal[regime],
                external_residuals,
                statistic=stat,
                seed=BOOTSTRAP_SEED,
                n_boot=N_BOOT,
            )
            for stat in STATISTICS
        }
        pairwise_excl[f"{label}_minus_A"] = {
            stat: bootstrap_regime_pair_delta_ci(
                excl_internal["A_random_5fold_cv"],
                excl_internal[regime],
                external_residuals,
                statistic=stat,
                seed=BOOTSTRAP_SEED,
                n_boot=N_BOOT,
            )
            for stat in STATISTICS
        }

    output = {
        "phase": "acs_issue09a_missing_family_excluded",
        "spec_path": "reports/remediation/specs/issue09_missing_family_sensitivity.md",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "n4b_source": str(N4B_PATH.relative_to(ROOT)),
        "n_full": 449,
        "n_excluding_missing_A_B": excl_n,
        "n_excluding_missing_C_lomfo": excl_n_lomfo,
        "n_missing_rows": 449 - excl_n,
        "full_internal_mae": full_internal,
        "excl_internal_mae": excl_internal,
        "full_regime_summary": full_summary,
        "excl_regime_summary": excl_summary,
        "pairwise_full": pairwise_full,
        "pairwise_excluding_missing": pairwise_excl,
    }
    guard_output(OUTPUT_PATH).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    print(f"n_excluding_missing={excl_n} (expect 431 = 449-18)")

    lines = [
        "# Issue 09 — `__MISSING__` family sensitivity",
        "",
        "**Spec:** `reports/remediation/specs/issue09_missing_family_sensitivity.md`.",
        "",
        "## Variant A — exclude the 18 `__MISSING__` rows from evaluation (cheap, no rerun)",
        "",
        f"Recomputed regimes A/B/C pooled internal MAE over the "
        f"{excl_n}-row population excluding `__MISSING__`, by row-count-weighted "
        f"recombination of already-frozen per-family MAE (no retraining).",
        "",
        "| Regime | Population | Median \\|gap\\| | Spearman rho | Kendall tau | Regret |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for regime in REGIMES:
        label = REGIME_LABELS[regime]
        f = full_summary[regime]
        e = excl_summary[regime]
        lines.append(
            f"| {label} | full (449) | {f['median_abs_mae_gap']:.4f} | {f['spearman_rho']:.4f} | "
            f"{f['kendall_tau']:.4f} | {f['top_model_selection_regret']:.4f} |"
        )
        lines.append(
            f"| {label} | excl. __MISSING__ ({excl_n}) | {e['median_abs_mae_gap']:.4f} | "
            f"{e['spearman_rho']:.4f} | {e['kendall_tau']:.4f} | "
            f"{e['top_model_selection_regret']:.4f} |"
        )
    lines.append("")
    lines.append("### Pairwise deltas vs. A, full vs. excluding __MISSING__ (95% bootstrap CI)")
    lines.append("")
    lines.append("| Comparison | Statistic | Population | Delta | 95% CI | Excludes 0 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key in pairwise_full:
        for stat in STATISTICS:
            fr = pairwise_full[key][stat]
            er = pairwise_excl[key][stat]
            f_excl = "Yes" if fr["ci_low"] * fr["ci_high"] > 0 else "No"
            e_excl = "Yes" if er["ci_low"] * er["ci_high"] > 0 else "No"
            lines.append(
                f"| {key.replace('_minus_A', ' minus A')} | {stat} | full | "
                f"{fr['point_delta']:.4f} | "
                f"[{fr['ci_low']:.4f}, {fr['ci_high']:.4f}] | {f_excl} |"
            )
            lines.append(
                f"| {key.replace('_minus_A', ' minus A')} | {stat} | excl. __MISSING__ | "
                f"{er['point_delta']:.4f} | [{er['ci_low']:.4f}, {er['ci_high']:.4f}] | {e_excl} |"
            )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_PATH).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
