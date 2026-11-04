#!/usr/bin/env python3
"""N8 — external applicability-domain / risk analysis (cross-dataset
study). Per artifacts/frozen/external_transfer_lock.json's `predeclared_n8_protocol`:
a composition-space nearest-neighbor AD score, fitted only on the same
449-row OBELiX development population N6 used for its final fit, evaluated
against the strict external cohort for ALL 8 models (not only N6's winner,
to avoid cherry-picking which model's AD story gets told).

Reuses `uncertainty.applicability_domain` UNCHANGED (not a single line
edited) for `fit_standardization`/`nearest_neighbor_ad_score`/
`ad_score_error_correlation`/`ad_score_quantile_error_table`
(APPLICABILITY_DOMAIN_VERSION=1.0, exactly as the lock declares) -- that
file is one of the old (closed) project's `CODE_HASH_FILES`
(scripts/phase2_7_pretest_lock.py), re-verified on every old-project test
run, so it must never change. The one new pure function the lock flagged
as still needed, `risk_coverage_mae_at_retained_fraction`, was added to a
brand-new sibling module (`uncertainty.risk_coverage`) instead, so the
old frozen lock's hash check is untouched. `Family` is never read here (AD
score is composition-only per the reused module's own docstring). No model
is fit in this script; N6's already-frozen predictions/residuals are
reused for the error side.

Run with:
  python scripts/transport_analysis/n8_applicability_domain.py
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
from sse_transfer.features.composition_pipeline import TrainFitCompositionScaler  # noqa: E402
from sse_transfer.uncertainty.applicability_domain import (  # noqa: E402
    APPLICABILITY_DOMAIN_VERSION,
    ad_score_error_correlation,
    ad_score_quantile_error_table,
    fit_standardization,
    nearest_neighbor_ad_score,
)
from sse_transfer.uncertainty.risk_coverage import (  # noqa: E402
    risk_coverage_mae_at_retained_fraction,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
STRICT_COHORT_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_strict.csv"
OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "n8_applicability_domain_corrected.json"
REPORT_PATH = ROOT / "reports" / "rerun" / "n8_applicability_domain.md"

RETAINED_FRACTIONS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def load_obelix_dev_compositions() -> pd.Series:
    """Same 449-row OBELiX development population (train UNION validation)
    and the same `True Composition` column N6 fit its final feature scaler
    on -- 'source training data' per N8's requirement."""
    dataset = load_regression_dataset()
    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)
    raw_by_id = raw.set_index("ID")
    dev_ids = pd.concat([dataset.train["ID"], dataset.validation["ID"]], ignore_index=True)
    return raw_by_id.loc[dev_ids, "True Composition"].reset_index(drop=True)


def main() -> None:
    n6 = json.loads(N6_PATH.read_text(encoding="utf-8"))
    strict_cohort = pd.read_csv(STRICT_COHORT_PATH)

    obelix_dev_compositions = load_obelix_dev_compositions()
    scaler = TrainFitCompositionScaler().fit(obelix_dev_compositions)
    x_ref = scaler.transform(obelix_dev_compositions)
    x_query = scaler.transform(strict_cohort["reduced_formula"])

    ad_stats = fit_standardization(x_ref)
    ad_scores = nearest_neighbor_ad_score(x_ref, x_query, ad_stats)

    per_model: dict[str, Any] = {}
    for model_name, payload in n6["results"].items():
        strict = payload["strict"]
        y_true = np.array(strict["predictions"]["y_true"])
        y_pred = np.array(strict["predictions"]["y_pred"])
        row_ids = strict["predictions"]["reduced_formula"]
        # AD scores were computed in `strict_cohort`'s row order; re-align
        # to N6's saved row order in case the two files ever diverge in
        # ordering (defensive, since both derive from the same source file).
        cohort_index = {rf: i for i, rf in enumerate(strict_cohort["reduced_formula"])}
        order = np.array([cohort_index[rf] for rf in row_ids])
        model_ad_scores = ad_scores[order]
        abs_residual = np.abs(y_pred - y_true)

        correlation = ad_score_error_correlation(model_ad_scores, abs_residual)
        quantile_table = ad_score_quantile_error_table(model_ad_scores, abs_residual, n_bins=4)
        risk_coverage = risk_coverage_mae_at_retained_fraction(
            model_ad_scores, abs_residual, RETAINED_FRACTIONS
        )
        per_model[model_name] = {
            "spearman_ad_vs_abs_error": correlation,
            "ad_quartile_error_table": quantile_table,
            "risk_coverage_mae": risk_coverage,
        }

    n_positive_spearman = sum(
        1
        for m in per_model.values()
        if not np.isnan(m["spearman_ad_vs_abs_error"]["spearman_r"])
        and m["spearman_ad_vs_abs_error"]["spearman_r"] > 0
    )
    n_defined_spearman = sum(
        1 for m in per_model.values() if not np.isnan(m["spearman_ad_vs_abs_error"]["spearman_r"])
    )

    best_model_by_external_mae = min(n6["results"], key=lambda m: n6["results"][m]["strict"]["mae"])
    best_model_coverage = per_model[best_model_by_external_mae]["risk_coverage_mae"]
    coverage_maes = [row["mae"] for row in best_model_coverage]
    n_non_monotonic_steps = sum(
        1 for i in range(1, len(coverage_maes)) if coverage_maes[i] > coverage_maes[i - 1]
    )
    best_model_spearman_r = per_model[best_model_by_external_mae]["spearman_ad_vs_abs_error"][
        "spearman_r"
    ]
    h3_supported_for_best_model = (
        not np.isnan(best_model_spearman_r)
        and best_model_spearman_r > 0
        and n_non_monotonic_steps <= 1
    )

    manifest = {
        "phase": "N8_applicability_domain",
        "applicability_domain_version": APPLICABILITY_DOMAIN_VERSION,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "lock_file_sha256": n6["lock_file_sha256"],
        "ad_reference_population_n": len(obelix_dev_compositions),
        "ad_query_population_n": len(strict_cohort),
        "retained_fractions": list(RETAINED_FRACTIONS),
        "per_model": per_model,
        "n_models_with_positive_spearman": n_positive_spearman,
        "n_models_with_defined_spearman": n_defined_spearman,
        "best_model_by_external_mae": best_model_by_external_mae,
        "best_model_n_non_monotonic_coverage_steps": n_non_monotonic_steps,
        "g4_h3_supported_for_best_model": h3_supported_for_best_model,
    }

    guard_output(OUTPUT_PATH).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(manifest)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {REPORT_PATH}")


def _write_report(manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# N8 — External Applicability Domain / Risk Analysis\n\n")
    lines.append("Status: **DONE (all 8 models, no cherry-picking)**\n")
    lines.append(f"Lock file hash: `{manifest['lock_file_sha256']}`\n")
    lines.append(
        f"AD method version: `{manifest['applicability_domain_version']}` (reused unchanged)\n\n"
    )
    lines.append(
        f"AD reference (source) population: {manifest['ad_reference_population_n']} OBELiX "
        f"development rows. Query population: {manifest['ad_query_population_n']} strict "
        "external cohort rows. `Family` never used as an AD input.\n\n"
    )

    lines.append("## 1. Spearman(AD score, |error|) per model\n\n")
    lines.append("| Model | Spearman r | p-value | n |\n")
    lines.append("|---|---|---|---|\n")
    for model, payload in manifest["per_model"].items():
        c = payload["spearman_ad_vs_abs_error"]
        r_str = f"{c['spearman_r']:.4f}" if not np.isnan(c["spearman_r"]) else "nan"
        p_str = f"{c['p_value']:.4g}" if not np.isnan(c["spearman_r"]) else "nan"
        lines.append(f"| {model} | {r_str} | {p_str} | {c['n']} |\n")
    lines.append(
        f"\n{manifest['n_models_with_positive_spearman']}/"
        f"{manifest['n_models_with_defined_spearman']} models with a defined correlation "
        "show a positive Spearman(AD, |error|) (higher AD distance -> larger error, the "
        "expected direction if AD is doing useful work).\n"
    )

    lines.append(
        "\n## 2. Risk-coverage MAE (best-by-external-MAE model: "
        f"`{manifest['best_model_by_external_mae']}`)\n\n"
    )
    lines.append("| Retained fraction | n retained | MAE |\n")
    lines.append("|---|---|---|\n")
    best_model_key = manifest["best_model_by_external_mae"]
    best_model_rows = manifest["per_model"][best_model_key]["risk_coverage_mae"]
    for row in best_model_rows:
        frac, n_ret, mae = row["retained_fraction"], row["n_retained"], row["mae"]
        lines.append(f"| {frac:.2f} | {n_ret} | {mae:.4f} |\n")
    lines.append(
        f"\nNon-monotonic steps (coverage decreasing but MAE increasing) for this model: "
        f"{manifest['best_model_n_non_monotonic_coverage_steps']} "
        f"(tolerance: <=1 permitted by the predeclared H3 rule).\n"
    )

    lines.append("\n## 3. Full risk-coverage table, all 8 models\n\n")
    lines.append("| Model | MAE@100% | MAE@75% | MAE@50% | MAE@25% |\n")
    lines.append("|---|---|---|---|---|\n")
    for model, payload in manifest["per_model"].items():
        maes = [f"{row['mae']:.4f}" for row in payload["risk_coverage_mae"]]
        lines.append(f"| {model} | {' | '.join(maes)} |\n")

    lines.append("\n## 4. H3 / G4 predeclared verdict\n\n")
    best_model_name = manifest["best_model_by_external_mae"]
    h3_verdict = "SUPPORTED" if manifest["g4_h3_supported_for_best_model"] else "NOT SUPPORTED"
    lines.append(
        "H3 (AD distance predicts external absolute error and supports useful selective "
        f"prediction) for the best-by-external-MAE model (`{best_model_name}`): "
        f"**{h3_verdict}** (positive Spearman AND <=1 non-monotonic risk-coverage step, "
        "per the lock's predeclared rule).\n\n"
    )
    lines.append(
        "This section reports the input G4 needs (Spearman positivity across the 8-model panel, "
        "and the best model's monotonicity) — the final G4 PASS/FAIL/PREFERRED verdict itself is "
        "N10's job, combining this with G1-G3.\n"
    )

    lines.append(
        "\n## 5. Not done in N8 (by design)\n\n"
        "- No conformal interval (optional per the lock; not attempted here — reported as "
        "not attempted, not as a negative result).\n"
        "- No sensitivity analysis (strict vs relaxed cohort, `__MISSING__` handling) — "
        "that is N9.\n"
        "- No GO/NO-GO verdict — that is N10.\n"
    )

    guard_output(REPORT_PATH).write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
