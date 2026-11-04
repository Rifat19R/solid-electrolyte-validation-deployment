#!/usr/bin/env python3
"""Reviewer remediation Issue 03: common-support LOMFO sensitivity.

Spec: reports/remediation/specs/issue03_common_support_lomfo.md (hashed in
reports/remediation/specs/HASHES.txt before this script was written).

Pure post-hoc recombination of already-frozen per-family MAE aggregates
(`artifacts/historical/n4b_nested_internal_validation.json`) -- no model is fit,
no data is read beyond that one frozen artifact plus N6's frozen external
residuals. Restricts regimes A and B's internal pooled MAE to the identical
396-row common support LOMFO (regime C) actually scores, then recomputes
the regime-utility statistics via the existing, unmodified
`regime_summary`/`bootstrap_regime_pair_delta_ci` functions.

Run with: python scripts/analysis/issue03_common_support_lomfo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

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
OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "issue03_common_support_lomfo_corrected.json"
REPORT_PATH = ROOT / "reports" / "rerun" / "issue03_common_support_lomfo.md"

REGIMES = ["A_random_5fold_cv", "B_family_group_kfold"]
REGIME_LABELS = {"A_random_5fold_cv": "A", "B_family_group_kfold": "B"}
BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
STATISTICS = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def common_support_pooled_mae(
    per_family_mae: dict[str, dict[str, float]], major_families: dict[str, int]
) -> tuple[float, int]:
    total_n = 0
    total_weighted = 0.0
    for fam, expected_n in major_families.items():
        entry = per_family_mae[fam]
        if entry["n"] != expected_n:
            raise ValueError(
                f"common_support_pooled_mae: family {fam!r} n mismatch "
                f"(per_family_mae has {entry['n']}, major_families expects {expected_n})"
            )
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
    major_families = n4b["major_families"]
    models = sorted(n4b["results"].keys())

    external_mae, external_residuals = load_n6_external()

    full_support_internal: dict[str, dict[str, float]] = {r: {} for r in REGIMES}
    common_support_internal: dict[str, dict[str, float]] = {r: {} for r in REGIMES}
    common_support_n_check: set[int] = set()

    for regime in REGIMES:
        for model in models:
            payload = n4b["results"][model][regime]
            full_support_internal[regime][model] = float(payload["pooled_mae"])
            cs_mae, cs_n = common_support_pooled_mae(payload["per_family_mae"], major_families)
            common_support_internal[regime][model] = cs_mae
            common_support_n_check.add(cs_n)

    if common_support_n_check != {396}:
        raise RuntimeError(
            f"acs03: expected common-support n=396 for every model, got {common_support_n_check}"
        )

    # Regime C is already computed over the common support by construction.
    c_lomfo_by_model = {
        model: float(n4b["results"][model]["C_lomfo"]["pooled_mae_over_major_family_rows"])
        for model in models
    }

    full_summary = {
        regime: regime_summary(full_support_internal[regime], external_mae) for regime in REGIMES
    }
    common_summary = {
        regime: regime_summary(common_support_internal[regime], external_mae) for regime in REGIMES
    }
    common_summary["C_lomfo"] = regime_summary(c_lomfo_by_model, external_mae)

    pairwise_full: dict[str, Any] = {}
    pairwise_common: dict[str, Any] = {}
    for regime in REGIMES:
        label = REGIME_LABELS[regime]
        pairwise_full[f"{label}_minus_A_full_support"] = (
            {
                stat: bootstrap_regime_pair_delta_ci(
                    full_support_internal["A_random_5fold_cv"],
                    full_support_internal[regime],
                    external_residuals,
                    statistic=stat,
                    seed=BOOTSTRAP_SEED,
                    n_boot=N_BOOT,
                )
                for stat in STATISTICS
            }
            if regime != "A_random_5fold_cv"
            else None
        )
        pairwise_common[f"{label}_minus_A_common_support"] = (
            {
                stat: bootstrap_regime_pair_delta_ci(
                    common_support_internal["A_random_5fold_cv"],
                    common_support_internal[regime],
                    external_residuals,
                    statistic=stat,
                    seed=BOOTSTRAP_SEED,
                    n_boot=N_BOOT,
                )
                for stat in STATISTICS
            }
            if regime != "A_random_5fold_cv"
            else None
        )
    pairwise_full = {k: v for k, v in pairwise_full.items() if v is not None}
    pairwise_common = {k: v for k, v in pairwise_common.items() if v is not None}

    output = {
        "phase": "acs_issue03_common_support_lomfo",
        "spec_path": "reports/remediation/specs/issue03_common_support_lomfo.md",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "n4b_source": str(N4B_PATH.relative_to(ROOT)),
        "n6_source": str(N6_PATH.relative_to(ROOT)),
        "common_support_n": 396,
        "full_support_n": 449,
        "major_families": major_families,
        "full_support_internal_mae": full_support_internal,
        "common_support_internal_mae": common_support_internal,
        "c_lomfo_internal_mae": c_lomfo_by_model,
        "full_support_regime_summary": full_summary,
        "common_support_regime_summary": common_summary,
        "pairwise_bootstrap_ci_full_support": pairwise_full,
        "pairwise_bootstrap_ci_common_support": pairwise_common,
    }
    guard_output(OUTPUT_PATH).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")

    lines = [
        "# Issue 03 — common-support LOMFO sensitivity",
        "",
        "**Spec:** `reports/remediation/specs/issue03_common_support_lomfo.md` "
        "(hashed before this script was written; see `reports/remediation/specs/HASHES.txt`).",
        "",
        "Common support = the 396 rows belonging to a major family (n>=15), the "
        "identical population LOMFO (regime C) scores. Regimes A and B restricted "
        "to this support by row-count-weighted recombination of already-frozen "
        "per-family MAE (no retraining).",
        "",
        "## Regime-utility statistics: full (449) vs common (396) support",
        "",
        "| Regime | Support | Median |gap| | Spearman rho | Kendall tau | Regret |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for regime in REGIMES:
        f = full_summary[regime]
        c = common_summary[regime]
        label = REGIME_LABELS[regime]
        lines.append(
            f"| {label} | full (449) | {f['median_abs_mae_gap']:.4f} | "
            f"{f['spearman_rho']:.4f} | {f['kendall_tau']:.4f} | "
            f"{f['top_model_selection_regret']:.4f} |"
        )
        lines.append(
            f"| {label} | common (396) | {c['median_abs_mae_gap']:.4f} | "
            f"{c['spearman_rho']:.4f} | {c['kendall_tau']:.4f} | "
            f"{c['top_model_selection_regret']:.4f} |"
        )
    c_sum = common_summary["C_lomfo"]
    lines.append(
        f"| C (LOMFO) | common (396, native) | {c_sum['median_abs_mae_gap']:.4f} | "
        f"{c_sum['spearman_rho']:.4f} | {c_sum['kendall_tau']:.4f} | "
        f"{c_sum['top_model_selection_regret']:.4f} |"
    )
    lines.append("")
    lines.append("## Pairwise deltas vs. regime A, full vs. common support (95% bootstrap CI)")
    lines.append("")
    lines.append("| Comparison | Statistic | Support | Delta | 95% CI | Excludes 0 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key, stats in pairwise_full.items():
        for stat, res in stats.items():
            excl = "Yes" if res["ci_low"] * res["ci_high"] > 0 else "No"
            lines.append(
                f"| {key.replace('_minus_A_full_support', ' minus A')} | {stat} | full | "
                f"{res['point_delta']:.4f} | [{res['ci_low']:.4f}, {res['ci_high']:.4f}] | {excl} |"
            )
    for key, stats in pairwise_common.items():
        for stat, res in stats.items():
            excl = "Yes" if (res["ci_low"] * res["ci_high"] > 0) else "No"
            lines.append(
                f"| {key.replace('_minus_A_common_support', ' minus A')} | {stat} | common | "
                f"{res['point_delta']:.4f} | [{res['ci_low']:.4f}, {res['ci_high']:.4f}] | {excl} |"
            )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_PATH).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
