#!/usr/bin/env python3
"""N7 — validation-regime utility comparison (cross-dataset study).

Pure post-hoc analysis of already-frozen N4 (internal, artifacts/frozen/
n4_internal_validation.json) and N6 (external, artifacts/frozen/
n6_external_transfer_primary.json) results, per artifacts/frozen/external_transfer_lock.json's
`predeclared_n7_protocol`. No model is fit here; no new data is read. Answers
H1 (does family-aware validation estimate external error better than random
CV?) and H2 (does it rank models better?) using the four predeclared
statistics, each with a bootstrap CI over the strict external cohort's rows.

Run with:
  python scripts/transport_analysis/n7_regime_utility.py
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
    REGIME_UTILITY_VERSION,
    bootstrap_regime_pair_delta_ci,
    regime_summary,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

N4_PATH = ROOT / "artifacts" / "frozen" / "n4_internal_validation.json"
N6_PATH = ROOT / "artifacts" / "frozen" / "n6_external_transfer_primary.json"
LOCK_PATH = ROOT / "artifacts" / "frozen" / "external_transfer_lock.json"
OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "n7_regime_utility.json"
REPORT_PATH = ROOT / "reports" / "rerun" / "n7_regime_utility.md"

REGIMES = ["A_random_5fold_cv", "B_family_group_kfold", "C_lomfo"]
REGIME_LABELS = {"A_random_5fold_cv": "A", "B_family_group_kfold": "B", "C_lomfo": "C"}
BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
STATISTICS = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def load_n4_internal() -> dict[str, dict[str, float]]:
    """{regime: {model: pooled_mae}}.

    Regime C (LOMFO)'s payload uses `pooled_mae_over_major_family_rows`
    (it only ever covers major-family rows, a partial population -- see
    `regime_caveat` below), while regimes A/B use `pooled_mae` (full
    449-row coverage). Both are the same underlying quantity (mean
    absolute error over whichever rows that regime held out), just named
    differently in N4's frozen output; this loader normalizes the key
    without altering either regime's already-frozen number.
    """
    data = json.loads(N4_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {r: {} for r in REGIMES}
    for model, regimes in data["results"].items():
        for regime, payload in regimes.items():
            key = "pooled_mae" if "pooled_mae" in payload else "pooled_mae_over_major_family_rows"
            out[regime][model] = float(payload[key])
    return out


def load_n6_external() -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """({model: strict external MAE}, {model: strict per-row residual array})."""
    data = json.loads(N6_PATH.read_text(encoding="utf-8"))
    mae_by_model: dict[str, float] = {}
    residuals_by_model: dict[str, np.ndarray] = {}
    for model, payload in data["results"].items():
        strict = payload["strict"]
        mae_by_model[model] = float(strict["mae"])
        residuals_by_model[model] = np.array(strict["predictions"]["residual"], dtype=float)
    return mae_by_model, residuals_by_model


def main() -> None:
    internal_by_regime = load_n4_internal()
    external_mae, external_residuals = load_n6_external()
    lock_hash = json.loads(N6_PATH.read_text(encoding="utf-8"))["lock_file_sha256"]

    per_regime_summary = {
        regime: regime_summary(internal_by_regime[regime], external_mae) for regime in REGIMES
    }

    pairwise_ci: dict[str, dict[str, Any]] = {}
    for i, regime_a in enumerate(REGIMES):
        for regime_b in REGIMES[i + 1 :]:
            key = f"{REGIME_LABELS[regime_b]}_minus_{REGIME_LABELS[regime_a]}"
            pairwise_ci[key] = {
                stat: bootstrap_regime_pair_delta_ci(
                    internal_by_regime[regime_a],
                    internal_by_regime[regime_b],
                    external_residuals,
                    statistic=stat,
                    seed=BOOTSTRAP_SEED,
                    n_boot=N_BOOT,
                )
                for stat in STATISTICS
            }

    # H1: does a family-aware regime (B or C) have a smaller median abs MAE
    # gap than random CV (A), with the bootstrap CI on that difference
    # excluding zero (negative = family-aware is better)?
    h1_support = {
        regime_pair: pairwise_ci[regime_pair]["median_abs_mae_gap"]["ci_high"] < 0.0
        for regime_pair in ("B_minus_A", "C_minus_A")
    }
    # H2: does a family-aware regime rank models better (higher Spearman/
    # Kendall) than random CV, CI excluding zero on the positive side?
    h2_support = {
        f"{regime_pair}_{stat}": pairwise_ci[regime_pair][stat]["ci_low"] > 0.0
        for regime_pair in ("B_minus_A", "C_minus_A")
        for stat in ("spearman_rho", "kendall_tau")
    }

    manifest = {
        "phase": "N7_regime_utility",
        "regime_utility_version": REGIME_UTILITY_VERSION,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "lock_file_sha256": lock_hash,
        "n4_source": str(N4_PATH.relative_to(ROOT)),
        "n6_source": str(N6_PATH.relative_to(ROOT)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_boot": N_BOOT,
        "regime_caveat": (
            "Regime C (LOMFO)'s pooled MAE is computed over only the major-family "
            "rows of the 449-row OBELiX pool (partial coverage), while regimes A and "
            "B cover all 449 rows. This is a known asymmetry inherited unchanged from "
            "N4's frozen numbers, not something N7 corrects -- comparisons involving C "
            "should be read with that caveat."
        ),
        "per_regime_summary": per_regime_summary,
        "pairwise_bootstrap_ci": pairwise_ci,
        "h1_family_aware_smaller_mae_gap_than_random_cv": h1_support,
        "h2_family_aware_better_rank_agreement_than_random_cv": h2_support,
    }

    guard_output(OUTPUT_PATH).write_text(
        json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
    )
    _write_report(manifest, internal_by_regime, external_mae)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {REPORT_PATH}")


def _write_report(
    manifest: dict[str, Any],
    internal_by_regime: dict[str, dict[str, float]],
    external_mae: dict[str, float],
) -> None:
    lines: list[str] = []
    lines.append("# N7 — Validation-Regime Utility Comparison\n\n")
    lines.append(
        "Status: **DONE (descriptive analysis of frozen N4/N6 numbers; no new fitting)**\n"
    )
    lines.append(f"Lock file hash: `{manifest['lock_file_sha256']}`\n\n")
    lines.append(f"> {manifest['regime_caveat']}\n\n")

    lines.append("## 1. Per-regime summary (vs. strict external MAE, n=204)\n\n")
    lines.append(
        "| Regime | Median \\|MAE gap\\| | Spearman rho | Kendall tau | Top-model regret |\n"
    )
    lines.append("|---|---|---|---|---|\n")
    for regime in REGIMES:
        s = manifest["per_regime_summary"][regime]
        regret = s["top_model_selection_regret"]
        lines.append(
            f"| {REGIME_LABELS[regime]} ({regime}) | {s['median_abs_mae_gap']:.4f} | "
            f"{s['spearman_rho']:.4f} | {s['kendall_tau']:.4f} | {regret:.4f} |\n"
        )

    lines.append(
        "\n## 2. Pairwise bootstrap CIs (regime_b - regime_a; negative gap-delta or "
        "positive rank-delta favors regime_b), n_boot=2000, seed=20260921\n\n"
    )
    lines.append("| Pair | Statistic | Point delta | 95% CI |\n")
    lines.append("|---|---|---|---|\n")
    for pair, stats in manifest["pairwise_bootstrap_ci"].items():
        for stat_name, s in stats.items():
            lines.append(
                f"| {pair} | {stat_name} | {s['point_delta']:.4f} | "
                f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}] |\n"
            )

    lines.append(
        "\n## 3. H1 / H2 (predeclared thresholds, per "
        "artifacts/frozen/external_transfer_lock.json)\n\n"
    )
    lines.append(
        "H1 (family-aware regime's median \\|MAE gap\\| is reliably smaller than random CV's, "
        "CI excludes zero on the negative side):\n\n"
    )
    for pair, supported in manifest["h1_family_aware_smaller_mae_gap_than_random_cv"].items():
        lines.append(f"- {pair}: **{'SUPPORTED' if supported else 'NOT SUPPORTED'}**\n")
    lines.append(
        "\nH2 (family-aware regime's rank agreement with external ranking is reliably higher than "
        "random CV's, CI excludes zero on the positive side):\n\n"
    )
    for key, supported in manifest["h2_family_aware_better_rank_agreement_than_random_cv"].items():
        lines.append(f"- {key}: **{'SUPPORTED' if supported else 'NOT SUPPORTED'}**\n")

    lines.append(
        "\nNeither H1 nor H2 being SUPPORTED is a scientifically valid outcome "
        '(RESEARCH_CONTRACT.md "Negative scientific results are valid") -- this section reports '
        "whichever result the bootstrap CIs actually show, without adjusting the threshold or the "
        "regime definitions after seeing them.\n"
    )

    lines.append("\n## 4. Raw internal (N4) vs external (N6) numbers used above\n\n")
    lines.append("| Model | A internal | B internal | C internal | External (strict) |\n")
    lines.append("|---|---|---|---|---|\n")
    for model in sorted(external_mae):
        a = internal_by_regime["A_random_5fold_cv"][model]
        b = internal_by_regime["B_family_group_kfold"][model]
        c = internal_by_regime["C_lomfo"][model]
        e = external_mae[model]
        lines.append(f"| {model} | {a:.4f} | {b:.4f} | {c:.4f} | {e:.4f} |\n")

    lines.append(
        "\n## 5. Not done in N7 (by design)\n\n"
        "- No applicability-domain/risk analysis — that is N8.\n"
        "- No sensitivity analyses (strict vs relaxed, `__MISSING__` handling) — that is N9.\n"
        "- No GO/NO-GO verdict — that is N10, using G3's predeclared rule against this "
        "section's results.\n"
    )

    guard_output(REPORT_PATH).write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
