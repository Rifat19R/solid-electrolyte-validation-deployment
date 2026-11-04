#!/usr/bin/env python3
"""Cross-dataset study, N21 Section C: primary external performance
context (reviewer-closure task, Section C).

Pure post-hoc analysis over the already-frozen N6 external predictions
(`artifacts/frozen/n6_external_transfer_primary.json`, strict cohort, n=204) and the
already-frozen N4b internal-validation development-population pool (used
only to read its already-observed target values -- no fitting, no new data
read, no Hargreaves label used for anything except the same one-shot
evaluation N6 already performed). No model is retrained or refit; no new
external inference occurs.

Adds, for the strict n=204 cohort, per model: MAE, RMSE, R2, Pearson r,
Spearman rho, and a simple MAE skill score against two naive constant
baselines (predict the OBELiX development-population's own mean / median
log10(sigma_RT) for every external row -- never the external cohort's own
mean/median, which would let external labels leak into the baseline
"model"). Skill score = 1 - MAE_model / MAE_mean_baseline (and, separately,
/ MAE_median_baseline).

Run:
    python scripts/transport_analysis/n21c_primary_external_context.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.utils.output_guard import guard_output  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "transport_analysis"))

import n4b_nested_internal_validation as n4b  # noqa: E402

N6_ARTIFACT = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
OUTPUT_JSON = ROOT / "artifacts" / "rerun" / "n21c_primary_external_context_corrected.json"
REPORT_MD = ROOT / "reports" / "rerun" / "n21c_primary_external_context.md"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def main() -> None:
    n6 = json.loads(N6_ARTIFACT.read_text(encoding="utf-8"))
    pool, _ = n4b.load_dev_population()
    dev_mean = float(np.mean(pool["y"].to_numpy()))
    dev_median = float(np.median(pool["y"].to_numpy()))

    # every model's strict-cohort y_true set is identical (same 204-row cohort); compute the
    # two naive-baseline MAEs once, from the first model's y_true, and cross-check on every
    # subsequent model below rather than recomputing.
    first_y_true = np.array(
        next(iter(n6["results"].values()))["strict"]["predictions"]["y_true"], dtype=float
    )
    mean_baseline_mae = float(np.mean(np.abs(first_y_true - dev_mean)))
    median_baseline_mae = float(np.mean(np.abs(first_y_true - dev_median)))

    per_model: dict[str, Any] = {}
    for model, payload in n6["results"].items():
        strict = payload["strict"]
        y_true = np.array(strict["predictions"]["y_true"], dtype=float)
        y_pred = np.array(strict["predictions"]["y_pred"], dtype=float)
        n = len(y_true)

        # sanity: every model shares the same strict-cohort y_true set
        assert abs(float(np.mean(np.abs(y_true - dev_mean))) - mean_baseline_mae) < 1e-9

        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        pearson_r = float(scipy_stats.pearsonr(y_true, y_pred).statistic)
        spearman_rho = float(scipy_stats.spearmanr(y_true, y_pred).statistic)

        # cross-check against already-frozen N6 metrics (same numbers, independently recomputed)
        assert abs(mae - float(strict["mae"])) < 1e-6, f"{model}: MAE mismatch vs frozen N6"
        assert abs(rmse - float(strict["rmse"])) < 1e-6, f"{model}: RMSE mismatch vs frozen N6"

        per_model[model] = {
            "n": n,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "pearson_r": pearson_r,
            "spearman_rho": spearman_rho,
            "mae_skill_vs_mean_baseline": 1.0 - mae / mean_baseline_mae,
            "mae_skill_vs_median_baseline": 1.0 - mae / median_baseline_mae,
        }

    manifest = {
        "phase": "N21_section_c_primary_external_context",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "n6_source": str(N6_ARTIFACT.relative_to(ROOT)),
        "n6_rerun": False,
        "cohort": "strict",
        "n": per_model[next(iter(per_model))]["n"],
        "naive_baseline_definition": (
            "constant prediction = OBELiX 449-row development-population mean/median "
            "log10(sigma_RT/S cm^-1), evaluated once against the strict external cohort's "
            "already-frozen true values -- never the external cohort's own mean/median."
        ),
        "dev_population_mean_y": dev_mean,
        "dev_population_median_y": dev_median,
        "mean_baseline_mae": mean_baseline_mae,
        "median_baseline_mae": median_baseline_mae,
        "per_model": per_model,
    }
    guard_output(OUTPUT_JSON).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    _write_report(manifest)
    print(f"wrote {REPORT_MD.relative_to(ROOT)}")


def _write_report(manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# N21 Section C — Primary External Performance Context\n\n")
    lines.append(f"Git commit: `{manifest['git_commit']}` on branch `{manifest['git_branch']}`\n\n")
    lines.append(
        f"Strict cohort, n={manifest['n']}. Naive baselines predict the OBELiX "
        f"development-population mean ({manifest['dev_population_mean_y']:.4f}) or median "
        f"({manifest['dev_population_median_y']:.4f}) log10(sigma_RT) for every external row -- "
        "the external cohort's own labels are read once for evaluation, exactly as for every "
        "other model, never used to fit the baseline itself. Mean-baseline MAE = "
        f"{manifest['mean_baseline_mae']:.4f}; median-baseline MAE = "
        f"{manifest['median_baseline_mae']:.4f}. No model was retrained or refit; N6 was not "
        "rerun.\n\n"
    )
    lines.append(
        "| Model | MAE | RMSE | R2 | Pearson r | Spearman rho | Skill vs mean baseline | "
        "Skill vs median baseline |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for model, d in sorted(manifest["per_model"].items(), key=lambda kv: kv[1]["mae"]):
        lines.append(
            f"| {model} | {d['mae']:.4f} | {d['rmse']:.4f} | {d['r2']:.4f} | "
            f"{d['pearson_r']:.4f} | {d['spearman_rho']:.4f} | "
            f"{d['mae_skill_vs_mean_baseline']:.4f} | {d['mae_skill_vs_median_baseline']:.4f} |\n"
        )
    lines.append(
        "\nA positive skill score means the model beats the naive constant baseline; a "
        "negative skill score means it does worse. All MAE/RMSE values above were independently "
        "recomputed from the frozen N6 per-row predictions and cross-checked to match the "
        "already-frozen N6 `mae`/`rmse` fields to within 1e-6 (assertion in the script).\n"
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_MD).write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
