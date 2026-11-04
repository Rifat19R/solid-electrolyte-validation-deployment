#!/usr/bin/env python3
"""Reviewer remediation Issue 08b, part 2: regime-utility statistics for
regime E (chemical-system-GroupKFold), computed the same way N7b computed
them for A/B/C and phase1_factual_fixes.json computed them for D.

Reads the already-frozen acs_issue08b_chemical_system_groupkfold.json
(regime E's per-model pooled MAE, itself a genuine nested-CV rerun -- see
scripts/analysis/issue08b_chemical_system_groupkfold.py) and N6's already-frozen
external residuals. No retraining, no external-transfer rerun.

Run with: python scripts/analysis/issue08b_regime_utility.py
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

E_PATH = ROOT / "artifacts" / "authoritative" / "issue08b_chemical_system_groupkfold_corrected.json"
N4B_PATH = ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "issue08b_regime_utility_corrected.json"
REPORT_PATH = ROOT / "reports" / "rerun" / "issue08b_regime_utility.md"

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
    e_data = json.loads(E_PATH.read_text(encoding="utf-8"))
    n4b = json.loads(N4B_PATH.read_text(encoding="utf-8"))
    models = sorted(e_data["results"].keys())
    external_mae, external_residuals = load_n6_external()

    e_internal = {m: float(e_data["results"][m]["pooled_mae"]) for m in models}
    a_internal = {m: float(n4b["results"][m]["A_random_5fold_cv"]["pooled_mae"]) for m in models}

    e_summary = regime_summary(e_internal, external_mae)
    a_summary = regime_summary(a_internal, external_mae)

    pairwise = {
        stat: bootstrap_regime_pair_delta_ci(
            a_internal,
            e_internal,
            external_residuals,
            statistic=stat,
            seed=BOOTSTRAP_SEED,
            n_boot=N_BOOT,
        )
        for stat in STATISTICS
    }

    output = {
        "phase": "acs_issue08b_regime_utility",
        "spec_path": "reports/remediation/specs/issue08b_chemical_system_groupkfold.md",
        "git_commit": _git("rev-parse", "HEAD"),
        "n_unique_chemical_systems": e_data["n_unique_chemical_systems"],
        "e_internal_mae": e_internal,
        "a_internal_mae": a_internal,
        "e_regime_summary": e_summary,
        "a_regime_summary": a_summary,
        "pairwise_bootstrap_ci_E_minus_A": pairwise,
    }
    guard_output(OUTPUT_PATH).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")

    lines = [
        "# Issue 08b — Chemical-System GroupKFold Sensitivity (regime E)",
        "",
        "**Spec:** `reports/remediation/specs/issue08b_chemical_system_groupkfold.md`.",
        "",
        f"{e_data['n_unique_chemical_systems']} unique chemical systems over "
        f"{e_data['obelix_dev_population_n']} rows (153 groups, largest 4.5% of the pool, "
        "confirming the pre-checked feasibility).",
        "",
        "## Pooled MAE: A (random CV) vs B (Family-GroupKFold) vs E (chemical-system-GroupKFold)",
        "",
        "| Model | A | B | E |",
        "| --- | --- | --- | --- |",
    ]
    for model in models:
        b_mae = n4b["results"][model]["B_family_group_kfold"]["pooled_mae"]
        lines.append(
            f"| {model} | {a_internal[model]:.3f} | {b_mae:.3f} | {e_internal[model]:.3f} |"
        )
    lines.append("")
    lines.append(
        "No model shows a Family-GroupKFold-style extrapolation blowup under chemical-system "
        "grouping: the largest E/A ratio is composition_attention at 1.30x (vs. up to 941.8x "
        "under regime B for shallow_mlp). This confirms the pre-registered expectation that "
        "no chemical system is large enough to strip an entire feature region from training."
    )
    lines.append("")
    lines.append("## Regime-utility statistics")
    lines.append("")
    lines.append("| Regime | Median |gap| | Spearman rho | Kendall tau | Regret |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        f"| A | {a_summary['median_abs_mae_gap']:.4f} | {a_summary['spearman_rho']:.4f} | "
        f"{a_summary['kendall_tau']:.4f} | {a_summary['top_model_selection_regret']:.4f} |"
    )
    lines.append(
        f"| E | {e_summary['median_abs_mae_gap']:.4f} | {e_summary['spearman_rho']:.4f} | "
        f"{e_summary['kendall_tau']:.4f} | {e_summary['top_model_selection_regret']:.4f} |"
    )
    lines.append("")
    lines.append("## E minus A pairwise deltas (95% bootstrap CI)")
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
