#!/usr/bin/env python3
"""Reviewer remediation Issue 09, Variant B, part 2: regime-utility
statistics for regime B' (Family-GroupKFold with __MISSING__ as 18
singleton groups), computed the same way N7b computed them for A/B/C.

Reads the already-frozen acs_issue09b_missing_singleton_groupkfold.json
(regime B''s per-model pooled MAE, itself a genuine nested-CV rerun -- see
scripts/analysis/issue09b_missing_singleton_groupkfold.py) and N6's already-frozen
external residuals. No retraining, no external-transfer rerun.

Run with: python scripts/analysis/issue09b_regime_utility.py
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

BPRIME_PATH = (
    ROOT / "artifacts" / "authoritative" / "issue09b_missing_singleton_groupkfold_corrected.json"
)
N4B_PATH = ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "issue09b_regime_utility_corrected.json"
REPORT_PATH = ROOT / "reports" / "rerun" / "issue09b_regime_utility.md"

BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
STATISTICS = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


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
    bp_data = json.loads(BPRIME_PATH.read_text(encoding="utf-8"))
    n4b = json.loads(N4B_PATH.read_text(encoding="utf-8"))
    models = sorted(bp_data["results"].keys())
    external_mae, external_residuals = load_n6_external()

    bp_internal = {m: float(bp_data["results"][m]["pooled_mae"]) for m in models}
    a_internal = {m: float(n4b["results"][m]["A_random_5fold_cv"]["pooled_mae"]) for m in models}
    b_internal = {m: float(n4b["results"][m]["B_family_group_kfold"]["pooled_mae"]) for m in models}

    bp_summary = regime_summary(bp_internal, external_mae)
    a_summary = regime_summary(a_internal, external_mae)
    b_summary = regime_summary(b_internal, external_mae)

    pairwise = {
        stat: bootstrap_regime_pair_delta_ci(
            a_internal,
            bp_internal,
            external_residuals,
            statistic=stat,
            seed=BOOTSTRAP_SEED,
            n_boot=N_BOOT,
        )
        for stat in STATISTICS
    }

    output = {
        "phase": "acs_issue09b_regime_utility",
        "spec_path": "reports/remediation/specs/issue09_missing_family_sensitivity.md",
        "git_commit": _git("rev-parse", "HEAD"),
        "bprime_internal_mae": bp_internal,
        "a_internal_mae": a_internal,
        "b_internal_mae": b_internal,
        "bprime_regime_summary": bp_summary,
        "a_regime_summary": a_summary,
        "b_regime_summary": b_summary,
        "pairwise_bootstrap_ci_Bprime_minus_A": pairwise,
    }
    guard_output(OUTPUT_PATH).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")

    lines = [
        "# Issue 09 Variant B — __MISSING__ rows as singleton groups (regime B')",
        "",
        "**Spec:** `reports/remediation/specs/issue09_missing_family_sensitivity.md`.",
        "",
        "## Pooled MAE: A (random CV) vs B (Family-GroupKFold, 1 __MISSING__ group) "
        "vs B' (18 singleton groups)",
        "",
        "| Model | A | B | B' |",
        "| --- | --- | --- | --- |",
    ]
    for model in models:
        lines.append(
            f"| {model} | {a_internal[model]:.3f} | {b_internal[model]:.3f} | "
            f"{bp_internal[model]:.3f} |"
        )
    lines.append("")
    lines.append(
        "**The catastrophic blowup essentially disappears for both previously-unstable "
        "models**: ridge 143.129 (B) -> 2.333 (B'); shallow_mlp 969.424 (B) -> 1.763 (B'). "
        "Every other model instead shows a *modest increase* relative to B (e.g. catboost "
        "1.487 -> 1.842; random_forest 2.030 -> 2.214) -- worse than B, but nowhere near "
        "ridge/shallow_mlp's original blowup magnitude."
    )
    lines.append("")
    lines.append(
        f"Ridge's B' fold sizes: {bp_data['results']['ridge']['fold_sizes']} (vs. B's "
        "[213, 29, 141, 31, 35] for the same seed/protocol) -- GroupKFold's greedy "
        "balancing produces a materially different fold structure once __MISSING__ is 18 "
        "singletons instead of one 18-row group, even though the largest fold (195/449, "
        "43.4%) is still comparably large to B's fold 0 (213/449, 47.4%)."
    )
    lines.append("")
    lines.append("## Regime-utility statistics")
    lines.append("")
    lines.append("| Regime | Median |gap| | Spearman rho | Kendall tau | Regret |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label, summ in (("A", a_summary), ("B", b_summary), ("B'", bp_summary)):
        lines.append(
            f"| {label} | {summ['median_abs_mae_gap']:.4f} | {summ['spearman_rho']:.4f} | "
            f"{summ['kendall_tau']:.4f} | {summ['top_model_selection_regret']:.4f} |"
        )
    lines.append("")
    lines.append("## B' minus A pairwise deltas (95% bootstrap CI)")
    lines.append("")
    lines.append("| Statistic | Delta | 95% CI | Excludes 0 |")
    lines.append("| --- | --- | --- | --- |")
    for stat in STATISTICS:
        res = pairwise[stat]
        excl = "Yes" if res["ci_low"] * res["ci_high"] > 0 else "No"
        lines.append(
            f"| {stat} | {res['point_delta']:.4f} | "
            f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}] | {excl} |"
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_PATH).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
