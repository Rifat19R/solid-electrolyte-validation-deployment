#!/usr/bin/env python3
"""N9 — predeclared sensitivity analyses (cross-dataset study).

Per artifacts/frozen/external_transfer_lock.json's `predeclared_n9_sensitivity_analyses`,
exactly two authorized analyses, no others:
  (A) strict vs relaxed external cohort -- re-run N7's regime-comparison
      statistics and N8's AD/risk-coverage analysis using the relaxed
      (239-row) cohort's already-frozen N6 predictions instead of strict,
      and compare conclusions.
  (B) `__MISSING__` family excluded from N4's chemical-family-level summary
      tables (macro_family_mae) -- this does not change cohort membership,
      only whether the `__MISSING__` group is included in the family-level
      average.

No model is fit here; every number is derived from N4/N6's already-frozen
outputs plus the fitted 449-row AD reference (re-fit identically to N8,
just queried against the relaxed cohort's extra rows).

Run with:
  python scripts/transport_analysis/n9_sensitivity.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.data.obelix_loader import load_obelix  # noqa: E402
from sse_transfer.data.regression_dataset import load_regression_dataset  # noqa: E402
from sse_transfer.evaluation.regime_utility import (  # noqa: E402
    bootstrap_regime_pair_delta_ci,
    macro_family_mae_excluding,
    regime_summary,
)
from sse_transfer.features.composition_pipeline import TrainFitCompositionScaler  # noqa: E402
from sse_transfer.uncertainty.applicability_domain import (  # noqa: E402
    ad_score_error_correlation,
    fit_standardization,
    nearest_neighbor_ad_score,
)
from sse_transfer.uncertainty.risk_coverage import (  # noqa: E402
    risk_coverage_mae_at_retained_fraction,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

N4_PATH = ROOT / "artifacts" / "frozen" / "n4_internal_validation.json"
N6_PATH = ROOT / "artifacts" / "frozen" / "n6_external_transfer_primary.json"
RELAXED_COHORT_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_relaxed.csv"
OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "n9_sensitivity.json"
REPORT_PATH = ROOT / "reports" / "rerun" / "n9_sensitivity.md"

REGIMES = ["A_random_5fold_cv", "B_family_group_kfold", "C_lomfo"]
REGIME_LABELS = {"A_random_5fold_cv": "A", "B_family_group_kfold": "B", "C_lomfo": "C"}
BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
STATISTICS = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]
RETAINED_FRACTIONS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
MISSING_FAMILY_LABEL = "__MISSING__"


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def load_n4_internal() -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    data = json.loads(N4_PATH.read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {r: {} for r in REGIMES}
    for model, regimes in data["results"].items():
        for regime, payload in regimes.items():
            key = "pooled_mae" if "pooled_mae" in payload else "pooled_mae_over_major_family_rows"
            out[regime][model] = float(payload[key])
    return out, data


def load_n6_cohort(
    n6: dict[str, Any], cohort: str
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    mae_by_model: dict[str, float] = {}
    residuals_by_model: dict[str, np.ndarray] = {}
    for model, payload in n6["results"].items():
        c = payload[cohort]
        mae_by_model[model] = float(c["mae"])
        residuals_by_model[model] = np.array(c["predictions"]["residual"], dtype=float)
    return mae_by_model, residuals_by_model


def load_obelix_dev_compositions() -> pd.Series:
    dataset = load_regression_dataset()
    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)
    raw_by_id = raw.set_index("ID")
    dev_ids = pd.concat([dataset.train["ID"], dataset.validation["ID"]], ignore_index=True)
    return raw_by_id.loc[dev_ids, "True Composition"].reset_index(drop=True)


def sensitivity_a_relaxed_vs_strict(
    n6: dict[str, Any], internal_by_regime: dict[str, dict[str, float]]
) -> dict[str, Any]:
    strict_mae, strict_resid = load_n6_cohort(n6, "strict")
    relaxed_mae, relaxed_resid = load_n6_cohort(n6, "relaxed")

    # N7 re-run on relaxed
    relaxed_summary = {
        regime: regime_summary(internal_by_regime[regime], relaxed_mae) for regime in REGIMES
    }
    relaxed_pairwise_ci: dict[str, dict[str, Any]] = {}
    for i, regime_a in enumerate(REGIMES):
        for regime_b in REGIMES[i + 1 :]:
            key = f"{REGIME_LABELS[regime_b]}_minus_{REGIME_LABELS[regime_a]}"
            relaxed_pairwise_ci[key] = {
                stat: bootstrap_regime_pair_delta_ci(
                    internal_by_regime[regime_a],
                    internal_by_regime[regime_b],
                    relaxed_resid,
                    statistic=stat,
                    seed=BOOTSTRAP_SEED,
                    n_boot=N_BOOT,
                )
                for stat in STATISTICS
            }
    relaxed_h1 = {
        pair: relaxed_pairwise_ci[pair]["median_abs_mae_gap"]["ci_high"] < 0.0
        for pair in ("B_minus_A", "C_minus_A")
    }
    relaxed_h2 = {
        f"{pair}_{stat}": relaxed_pairwise_ci[pair][stat]["ci_low"] > 0.0
        for pair in ("B_minus_A", "C_minus_A")
        for stat in ("spearman_rho", "kendall_tau")
    }

    # N8 re-run on relaxed
    obelix_dev_compositions = load_obelix_dev_compositions()
    scaler = TrainFitCompositionScaler().fit(obelix_dev_compositions)
    x_ref = scaler.transform(obelix_dev_compositions)
    relaxed_cohort = pd.read_csv(RELAXED_COHORT_PATH)
    x_query = scaler.transform(relaxed_cohort["reduced_formula"])
    ad_stats = fit_standardization(x_ref)
    ad_scores = nearest_neighbor_ad_score(x_ref, x_query, ad_stats)
    cohort_index = {rf: i for i, rf in enumerate(relaxed_cohort["reduced_formula"])}

    relaxed_ad_per_model: dict[str, Any] = {}
    for model_name, payload in n6["results"].items():
        relaxed_pred = payload["relaxed"]["predictions"]
        y_true = np.array(relaxed_pred["y_true"])
        y_pred = np.array(relaxed_pred["y_pred"])
        row_ids = relaxed_pred["reduced_formula"]
        order = np.array([cohort_index[rf] for rf in row_ids])
        model_ad_scores = ad_scores[order]
        abs_residual = np.abs(y_pred - y_true)
        correlation = ad_score_error_correlation(model_ad_scores, abs_residual)
        risk_coverage = risk_coverage_mae_at_retained_fraction(
            model_ad_scores, abs_residual, RETAINED_FRACTIONS
        )
        relaxed_ad_per_model[model_name] = {
            "spearman_ad_vs_abs_error": correlation,
            "risk_coverage_mae": risk_coverage,
        }

    n_positive_relaxed = sum(
        1
        for m in relaxed_ad_per_model.values()
        if not np.isnan(m["spearman_ad_vs_abs_error"]["spearman_r"])
        and m["spearman_ad_vs_abs_error"]["spearman_r"] > 0
    )

    # Model ranking stability
    strict_ranking = sorted(strict_mae, key=lambda m: strict_mae[m])
    relaxed_ranking = sorted(relaxed_mae, key=lambda m: relaxed_mae[m])

    return {
        "strict_mae": strict_mae,
        "relaxed_mae": relaxed_mae,
        "strict_vs_relaxed_mae_delta": {m: relaxed_mae[m] - strict_mae[m] for m in strict_mae},
        "strict_model_ranking": strict_ranking,
        "relaxed_model_ranking": relaxed_ranking,
        "ranking_identical": strict_ranking == relaxed_ranking,
        "relaxed_n7_per_regime_summary": relaxed_summary,
        "relaxed_n7_h1_support": relaxed_h1,
        "relaxed_n7_h2_support": relaxed_h2,
        "relaxed_n8_per_model": relaxed_ad_per_model,
        "relaxed_n8_n_models_positive_spearman": n_positive_relaxed,
        "relaxed_n8_n_models_total": len(relaxed_ad_per_model),
    }


def sensitivity_b_missing_family_excluded(n4_raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, regimes in n4_raw["results"].items():
        out[model] = {}
        for regime, payload in regimes.items():
            per_family = payload["per_family_mae"]
            with_missing = float(np.mean([v["mae"] for v in per_family.values()]))
            without_missing_mae = macro_family_mae_excluding(per_family, MISSING_FAMILY_LABEL)
            out[model][regime] = {
                "macro_family_mae_with_missing": with_missing,
                "macro_family_mae_without_missing": without_missing_mae,
                "delta": without_missing_mae - with_missing,
                "missing_family_mae": per_family.get(MISSING_FAMILY_LABEL, {}).get("mae"),
                "missing_family_n": per_family.get(MISSING_FAMILY_LABEL, {}).get("n"),
            }
    return out


def main() -> None:
    n6 = json.loads(N6_PATH.read_text(encoding="utf-8"))
    internal_by_regime, n4_raw = load_n4_internal()

    sensitivity_a = sensitivity_a_relaxed_vs_strict(n6, internal_by_regime)
    sensitivity_b = sensitivity_b_missing_family_excluded(n4_raw)

    manifest = {
        "phase": "N9_sensitivity",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "lock_file_sha256": n6["lock_file_sha256"],
        "sensitivity_a_strict_vs_relaxed": sensitivity_a,
        "sensitivity_b_missing_family_excluded": sensitivity_b,
    }
    guard_output(OUTPUT_PATH).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(manifest)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {REPORT_PATH}")


def _write_report(manifest: dict[str, Any]) -> None:
    a = manifest["sensitivity_a_strict_vs_relaxed"]
    b = manifest["sensitivity_b_missing_family_excluded"]
    lines: list[str] = []
    lines.append("# N9 — Predeclared Sensitivity Analyses\n\n")
    lines.append("Status: **DONE (only the two authorized analyses; no post-hoc model hunting)**\n")
    lines.append(f"Lock file hash: `{manifest['lock_file_sha256']}`\n\n")

    lines.append("## A. Strict vs relaxed external cohort\n\n")
    lines.append("### A.1 External MAE, strict vs relaxed\n\n")
    lines.append("| Model | Strict MAE | Relaxed MAE | Delta |\n")
    lines.append("|---|---|---|---|\n")
    for m in sorted(a["strict_mae"]):
        s, r = a["strict_mae"][m], a["relaxed_mae"][m]
        lines.append(f"| {m} | {s:.4f} | {r:.4f} | {r - s:+.4f} |\n")
    lines.append(
        f"\nModel ranking by external MAE identical between strict and relaxed: "
        f"**{a['ranking_identical']}**\n"
    )
    if not a["ranking_identical"]:
        lines.append(f"- Strict ranking: {a['strict_model_ranking']}\n")
        lines.append(f"- Relaxed ranking: {a['relaxed_model_ranking']}\n")

    lines.append("\n### A.2 N7 H1/H2 re-run on relaxed cohort\n\n")
    lines.append(
        "H1 (family-aware regime's median |MAE gap| reliably smaller than random CV's):\n\n"
    )
    for pair, supported in a["relaxed_n7_h1_support"].items():
        lines.append(f"- {pair}: **{'SUPPORTED' if supported else 'NOT SUPPORTED'}**\n")
    lines.append(
        "\nH2 (family-aware regime's rank agreement reliably higher than random CV's):\n\n"
    )
    for key, supported in a["relaxed_n7_h2_support"].items():
        lines.append(f"- {key}: **{'SUPPORTED' if supported else 'NOT SUPPORTED'}**\n")

    lines.append("\n### A.3 N8 AD/risk-coverage re-run on relaxed cohort\n\n")
    lines.append(
        f"{a['relaxed_n8_n_models_positive_spearman']}/{a['relaxed_n8_n_models_total']} models "
        "show positive Spearman(AD, |error|) on the relaxed cohort "
        f"(strict cohort: {sum(1 for m in a['relaxed_n8_per_model'])}/8 comparison in N8's own "
        "report).\n\n"
    )
    lines.append("| Model | Relaxed Spearman r | Relaxed MAE@100% | Relaxed MAE@25% |\n")
    lines.append("|---|---|---|---|\n")
    for model, payload in a["relaxed_n8_per_model"].items():
        r = payload["spearman_ad_vs_abs_error"]["spearman_r"]
        r_str = f"{r:.4f}" if not np.isnan(r) else "nan"
        cov = payload["risk_coverage_mae"]
        lines.append(f"| {model} | {r_str} | {cov[0]['mae']:.4f} | {cov[-1]['mae']:.4f} |\n")

    lines.append(
        "\n## B. `__MISSING__` family excluded from N4's chemical-family-level summary\n\n"
    )
    lines.append(
        "Cohort membership is unaffected (this is a family-summary-table axis only, not a "
        "cohort-exclusion rule). Comparing macro-family MAE with vs. without the `__MISSING__` "
        "group across all 8 models x 3 regimes:\n\n"
    )
    lines.append(
        "| Model | Regime | With `__MISSING__` | Without | Delta | `__MISSING__` MAE (n) |\n"
    )
    lines.append("|---|---|---|---|---|---|\n")
    for model, regimes in b.items():
        for regime, payload in regimes.items():
            wm = payload["macro_family_mae_with_missing"]
            wom = payload["macro_family_mae_without_missing"]
            delta = payload["delta"]
            mm_mae = payload["missing_family_mae"]
            mm_n = payload["missing_family_n"]
            mm_str = f"{mm_mae:.4f} (n={mm_n})" if mm_mae is not None else "n/a"
            lines.append(
                f"| {model} | {REGIME_LABELS.get(regime, regime)} | {wm:.4f} | {wom:.4f} | "
                f"{delta:+.4f} | {mm_str} |\n"
            )

    lines.append(
        "\n## Not done in N9 (by design)\n\n"
        "- No other sensitivity analysis was run (the lock authorizes exactly these two).\n"
        "- No model was added, removed, or re-tuned as a result of these findings.\n"
        "- No GO/NO-GO verdict — that is N10.\n"
    )
    guard_output(REPORT_PATH).write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
