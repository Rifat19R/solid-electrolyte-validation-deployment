#!/usr/bin/env python3
"""N9b — relaxed-cohort H1/H2 sensitivity comparison, recomputed on NESTED
internal numbers (post-lock remediation of N9 §A.2).

N9's sensitivity A.2 (relaxed-cohort re-run of N7's H1/H2 statistics) read
the leaky N4 internal numbers (`artifacts/frozen/n4_internal_validation.json`),
same bug N7b already fixed for the strict cohort. This script applies the
identical fix to the relaxed-cohort comparison only: it reads N4b's nested
numbers (`artifacts/historical/n4b_nested_internal_validation.json`) and N6's
already-frozen relaxed-cohort external predictions
(`artifacts/frozen/n6_external_transfer_primary.json`, `results.*.relaxed`) --
unchanged, not rerun. Same four statistics, same bootstrap settings
(seed=20260921, n_boot=2000), same H1/H2 decision rule as N9 and N7b.

Everything else in N9 (external-MAE table, N8 AD/risk-coverage re-run,
sensitivity B `__MISSING__`-family exclusion) does not depend on the
internal-CV numbers and is intentionally NOT recomputed here -- N9 remains
the source for those. This script and its outputs are additive; N9's own
artifact/report are untouched and remain the historical (leaky-internal)
record.

Run with:
  python scripts/transport_analysis/n9b_relaxed_sensitivity_nested.py
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

N4B_PATH = ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
N9_OLD_PATH = ROOT / "artifacts" / "historical" / "n9_sensitivity.json"
N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "n9b_relaxed_sensitivity_nested_corrected.json"
REPORT_PATH = ROOT / "reports" / "rerun" / "n9b_relaxed_sensitivity_nested.md"

REGIMES = ["A_random_5fold_cv", "B_family_group_kfold", "C_lomfo"]
REGIME_LABELS = {"A_random_5fold_cv": "A", "B_family_group_kfold": "B", "C_lomfo": "C"}
BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
STATISTICS = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def load_n4b_internal() -> dict[str, dict[str, float]]:
    data = json.loads(N4B_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {r: {} for r in REGIMES}
    for model, regimes in data["results"].items():
        for regime, payload in regimes.items():
            key = "pooled_mae" if "pooled_mae" in payload else "pooled_mae_over_major_family_rows"
            out[regime][model] = float(payload[key])
    return out


def load_n6_relaxed() -> tuple[dict[str, float], dict[str, np.ndarray]]:
    data = json.loads(N6_PATH.read_text(encoding="utf-8"))
    mae_by_model: dict[str, float] = {}
    residuals_by_model: dict[str, np.ndarray] = {}
    for model, payload in data["results"].items():
        relaxed = payload["relaxed"]
        mae_by_model[model] = float(relaxed["mae"])
        residuals_by_model[model] = np.array(relaxed["predictions"]["residual"], dtype=float)
    return mae_by_model, residuals_by_model


def main() -> None:
    internal_by_regime = load_n4b_internal()
    external_mae, external_residuals = load_n6_relaxed()
    lock_hash = json.loads(N6_PATH.read_text(encoding="utf-8"))["lock_file_sha256"]
    old_n9 = json.loads(N9_OLD_PATH.read_text(encoding="utf-8")) if N9_OLD_PATH.exists() else None

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

    h1_support = {
        regime_pair: pairwise_ci[regime_pair]["median_abs_mae_gap"]["ci_high"] < 0.0
        for regime_pair in ("B_minus_A", "C_minus_A")
    }
    h2_support = {
        f"{regime_pair}_{stat}": pairwise_ci[regime_pair][stat]["ci_low"] > 0.0
        for regime_pair in ("B_minus_A", "C_minus_A")
        for stat in ("spearman_rho", "kendall_tau")
    }

    manifest = {
        "phase": "N9b_relaxed_sensitivity_nested",
        "supersedes": (
            "N9 sensitivity-A.2 only (reports/cross_dataset_study/n9_sensitivity.md) -- "
            "recomputed on N4b "
            "nested internal numbers; N9's external-MAE table, N8 AD re-run, and sensitivity B "
            "(__MISSING__-family exclusion) are unaffected and remain sourced from N9"
        ),
        "regime_utility_version": REGIME_UTILITY_VERSION,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "lock_file_sha256": lock_hash,
        "n4b_source": str(N4B_PATH.relative_to(ROOT)),
        "n6_source": str(N6_PATH.relative_to(ROOT)),
        "n6_cohort": "relaxed",
        "n6_rerun": False,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_boot": N_BOOT,
        "per_regime_summary": per_regime_summary,
        "pairwise_bootstrap_ci": pairwise_ci,
        "h1_family_aware_smaller_mae_gap_than_random_cv": h1_support,
        "h2_family_aware_better_rank_agreement_than_random_cv": h2_support,
    }

    guard_output(OUTPUT_PATH).write_text(
        json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8"
    )
    _write_report(manifest, internal_by_regime, external_mae, old_n9)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {REPORT_PATH}")


def _write_report(
    manifest: dict[str, Any],
    internal_by_regime: dict[str, dict[str, float]],
    external_mae: dict[str, float],
    old_n9: dict[str, Any] | None,
) -> None:
    lines: list[str] = []
    lines.append(
        "# N9b — Relaxed-Cohort H1/H2 Sensitivity Comparison (nested, post-lock remediation)\n\n"
    )
    lines.append(
        "Status: **DONE (recomputes ONLY N9 sensitivity-A.2 -- the relaxed-cohort H1/H2 "
        "re-run -- against N4b's nested internal numbers; N6's relaxed-cohort external "
        "predictions are the same already-frozen numbers N9 used, not rerun; N9's external-MAE "
        "table, N8 AD re-run, and sensitivity B are unaffected and remain sourced from N9)**\n"
    )
    lines.append(f"Lock file hash: `{manifest['lock_file_sha256']}`\n\n")

    lines.append("## 1. Per-regime summary (vs. relaxed external MAE, n=239)\n\n")
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

    if old_n9 is not None:
        old_summary = old_n9["sensitivity_a_strict_vs_relaxed"]["relaxed_n7_per_regime_summary"]
        lines.append(
            "\n## 2. Old (N9 sensitivity-A.2, leaky) vs. new (N9b, nested) per-regime summary\n\n"
        )
        lines.append("| Regime | Statistic | N9 (leaky) | N9b (nested) | Delta |\n")
        lines.append("|---|---|---|---|---|\n")
        for regime in REGIMES:
            old_s = old_summary[regime]
            new_s = manifest["per_regime_summary"][regime]
            for stat in [
                "median_abs_mae_gap",
                "spearman_rho",
                "kendall_tau",
                "top_model_selection_regret",
            ]:
                lines.append(
                    f"| {REGIME_LABELS[regime]} | {stat} | {old_s[stat]:.4f} | {new_s[stat]:.4f} | "
                    f"{new_s[stat] - old_s[stat]:+.4f} |\n"
                )

    lines.append(
        "\n## 3. Pairwise bootstrap CIs (regime_b - regime_a), n_boot=2000, seed=20260921\n\n"
    )
    lines.append("| Pair | Statistic | Point delta | 95% CI |\n")
    lines.append("|---|---|---|---|\n")
    for pair, stats in manifest["pairwise_bootstrap_ci"].items():
        for stat_name, s in stats.items():
            lines.append(
                f"| {pair} | {stat_name} | {s['point_delta']:.4f} | "
                f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}] |\n"
            )

    def _verdict_line(label: str, supported: bool, old_supported: bool | None) -> str:
        old_label = (
            "N/A" if old_supported is None else ("SUPPORTED" if old_supported else "NOT SUPPORTED")
        )
        changed = (old_supported is not None) and (old_supported != supported)
        new_label = "SUPPORTED" if supported else "NOT SUPPORTED"
        tail = " — CHANGED" if changed else " — unchanged"
        return f"- {label}: **{new_label}** (N9 was {old_label}{tail})\n"

    lines.append(
        "\n## 4. H1 / H2 verdicts — nested, relaxed cohort, and whether they changed from N9\n\n"
    )
    lines.append(
        "H1 (family-aware regime's median \\|MAE gap\\| is reliably smaller than random CV's):\n\n"
    )
    old_h1 = old_n9["sensitivity_a_strict_vs_relaxed"]["relaxed_n7_h1_support"] if old_n9 else None
    for pair, supported in manifest["h1_family_aware_smaller_mae_gap_than_random_cv"].items():
        old_supported = old_h1[pair] if old_h1 else None
        lines.append(_verdict_line(pair, supported, old_supported))
    lines.append(
        "\nH2 (family-aware regime's rank agreement with external ranking is reliably higher):\n\n"
    )
    old_h2 = old_n9["sensitivity_a_strict_vs_relaxed"]["relaxed_n7_h2_support"] if old_n9 else None
    for key, supported in manifest["h2_family_aware_better_rank_agreement_than_random_cv"].items():
        old_supported = old_h2[key] if old_h2 else None
        lines.append(_verdict_line(key, supported, old_supported))

    lines.append(
        "\nNeither H1 nor H2 being SUPPORTED is a scientifically valid outcome "
        '(RESEARCH_CONTRACT.md "Negative scientific results are valid") -- this section reports '
        "whichever result the bootstrap CIs actually show.\n"
    )

    lines.append(
        "\n## 5. Raw internal (N4b nested) vs external (N6 relaxed, unchanged) numbers used "
        "above\n\n"
    )
    lines.append("| Model | A internal | B internal | C internal | External (relaxed) |\n")
    lines.append("|---|---|---|---|---|\n")
    for model in sorted(external_mae):
        a = internal_by_regime["A_random_5fold_cv"][model]
        b = internal_by_regime["B_family_group_kfold"][model]
        c = internal_by_regime["C_lomfo"][model]
        e = external_mae[model]
        lines.append(f"| {model} | {a:.4f} | {b:.4f} | {c:.4f} | {e:.4f} |\n")

    lines.append(
        "\n## Not done in N9b (by design)\n\n"
        "- N9's external-MAE table (§A.1), N8 AD/risk-coverage re-run (§A.3), and sensitivity B "
        "(`__MISSING__`-family exclusion, §B) do not depend on the internal-CV numbers and are "
        "NOT recomputed here -- N9 (`reports/cross_dataset_study/n9_sensitivity.md`) remains "
        "their source.\n"
        "- N6 was not rerun.\n"
        "- No GO/NO-GO verdict change is asserted here beyond what §4 above shows directly.\n"
    )
    guard_output(REPORT_PATH).write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
