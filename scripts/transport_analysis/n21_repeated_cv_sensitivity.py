#!/usr/bin/env python3
"""Cross-dataset study, N21 Section B: repeated random-5-fold CV
fold-randomness sensitivity (reviewer-closure task, Section B).

**Why this script exists:** every regime-comparison result reported so far
(N4b/N7b primary 8-model, N15/N9b six-model-stable sensitivity) used ONE
realization of regime A (random 5-fold CV, seed=20260921). This script asks
how much that single realization's internal MAE and the four downstream
regime-utility statistics (median |MAE gap|, Spearman rho, Kendall tau,
top-model-selection regret vs. the FIXED external N6 numbers) would have
varied under a different random fold assignment, using the exact same
nested per-fold Optuna tuning protocol as N4b (same search spaces,
`N_TRIALS`/`TORCH_N_TRIALS` budgets, inner-tuning-split fraction).

Only regime A (random 5-fold CV) is repeated -- regimes B (Family-GroupKFold)
and C (LOMFO) are deterministic given the family taxonomy and are NOT
re-run here; their already-frozen N4b numbers are used as fixed comparators,
exactly as the task brief specifies. External N6 predictions are NEVER
rerun and the frozen 113-row OBELiX `test` split is never opened.

10 deterministic seeds are used: 20260921 (identical to N4b's original
regime-A seed -- included so the original single-seed result is one member
of this script's own repeated distribution, not a separate computation) and
20260922..20260930 (9 new seeds, arbitrary increments of the original,
chosen before any result was seen).

Run per (seed, model) as its own OS process (same OOM-avoidance rationale
as N4b/N20b):
    python scripts/transport_analysis/n21_repeated_cv_sensitivity.py --seed <seed> --model <name>
Then merge:
    python scripts/transport_analysis/n21_repeated_cv_sensitivity.py --merge
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "transport_analysis"))

import n4b_nested_internal_validation as n4b  # noqa: E402

from sse_transfer.evaluation.regime_utility import (  # noqa: E402
    REGIME_UTILITY_VERSION,
    regime_summary,
)
from sse_transfer.models.ladder import MODEL_NAMES  # noqa: E402
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

SEEDS = [20260921 + i for i in range(10)]
N4B_ARTIFACT = (
    ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
)
N6_ARTIFACT = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
FRAGMENTS_DIR = ROOT / "artifacts" / "fragments" / "n21"
# Recorded fragments (read by --merge) are immutable; --model reruns write here.
FRAGMENT_OUT_DIR = ROOT / "artifacts" / "rerun" / "fragments" / "n21"
ARTIFACT_JSON = ROOT / "artifacts" / "rerun" / "n21_repeated_cv_sensitivity_corrected.json"
REPORT_MD = ROOT / "reports" / "rerun" / "n21b_repeated_cv_sensitivity.md"
STABLE_MODELS = [m for m in MODEL_NAMES if m not in ("ridge", "shallow_mlp")]
STAT_NAMES = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def load_external_mae() -> dict[str, float]:
    data = json.loads(N6_ARTIFACT.read_text(encoding="utf-8"))
    return {model: float(payload["strict"]["mae"]) for model, payload in data["results"].items()}


def run_one(seed: int, model_name: str) -> None:
    t0 = time.time()
    pool, _ = n4b.load_dev_population()
    n4b.CV_SEED = seed  # drives fold construction, inner-tuning split, and fit/tune seed alike
    result = n4b.run_regime_a(pool, model_name)
    runtime = time.time() - t0
    fragment = {
        "seed": seed,
        "model_name": model_name,
        "regime_a": result,
        "runtime_seconds": runtime,
    }
    FRAGMENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    frag_path = FRAGMENT_OUT_DIR / f"seed{seed}_{model_name}.json"
    guard_output(frag_path).write_text(
        json.dumps(fragment, indent=2, default=str), encoding="utf-8"
    )
    print(f"seed={seed} model={model_name} pooled_mae={result['pooled_mae']:.4f} ({runtime:.1f}s)")
    print(f"wrote {frag_path.relative_to(ROOT)}")


def _percentile_ci(values: list[float], lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def _a_favored(stat: str, a_val: float, b_val: float, c_val: float) -> bool:
    # median_abs_mae_gap and top_model_selection_regret: smaller = better for A.
    # spearman_rho and kendall_tau: larger = better for A (stronger rank agreement).
    if stat in ("median_abs_mae_gap", "top_model_selection_regret"):
        return a_val < b_val and a_val < c_val
    return a_val > b_val and a_val > c_val


def _subset_analysis(
    label: str,
    models: list[str],
    internal_by_seed_model: dict[int, dict[str, float]],
    external_mae: dict[str, float],
    b_internal: dict[str, float],
    c_internal: dict[str, float],
) -> dict[str, Any]:
    """Everything in Section B's spec (per-model MAE distribution, the four
    regime-utility statistics' distribution across seeds, and the
    strict-overlap verdict) computed for one model subset (`models`). Reuses
    the SAME already-computed per-(seed, model) regime-A MAE values -- no
    new fitting -- so a second subset (e.g. the six-model stable sensitivity)
    is free to add once the primary 8-model fragments already exist."""
    b_subset = {m: b_internal[m] for m in models}
    c_subset = {m: c_internal[m] for m in models}
    external_subset = {m: external_mae[m] for m in models}
    b_summary = regime_summary(b_subset, external_subset)
    c_summary = regime_summary(c_subset, external_subset)

    per_model_across_seeds: dict[str, Any] = {}
    for model_name in models:
        vals = [internal_by_seed_model[seed][model_name] for seed in SEEDS]
        lo, hi = _percentile_ci(vals)
        per_model_across_seeds[model_name] = {
            "seed_values": vals,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "sd": float(np.std(vals, ddof=1)),
            "ci95_low": lo,
            "ci95_high": hi,
            "original_seed_value": internal_by_seed_model[SEEDS[0]][model_name],
        }

    per_seed_summary = {
        seed: regime_summary({m: internal_by_seed_model[seed][m] for m in models}, external_subset)
        for seed in SEEDS
    }
    stat_distribution: dict[str, Any] = {}
    for stat in STAT_NAMES:
        vals = [per_seed_summary[seed][stat] for seed in SEEDS]
        lo, hi = _percentile_ci(vals)
        stat_distribution[stat] = {
            "seed_values": vals,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "sd": float(np.std(vals, ddof=1)),
            "ci95_low": lo,
            "ci95_high": hi,
            "original_seed_value": per_seed_summary[SEEDS[0]][stat],
            "b_fixed_value": b_summary[stat],
            "c_fixed_value": c_summary[stat],
        }

    verdicts: dict[str, Any] = {}
    for stat in STAT_NAMES:
        d = stat_distribution[stat]
        interval_overlaps_b = d["ci95_low"] <= d["b_fixed_value"] <= d["ci95_high"]
        interval_overlaps_c = d["ci95_low"] <= d["c_fixed_value"] <= d["ci95_high"]
        median_favors_a = _a_favored(stat, d["median"], d["b_fixed_value"], d["c_fixed_value"])
        verdicts[stat] = {
            "median_favors_a_over_b_and_c": median_favors_a,
            "ci95_overlaps_b_fixed_value": interval_overlaps_b,
            "ci95_overlaps_c_fixed_value": interval_overlaps_c,
            "claim_a_superiority_supported": bool(
                median_favors_a and not interval_overlaps_b and not interval_overlaps_c
            ),
        }

    return {
        "label": label,
        "models": models,
        "n_models": len(models),
        "per_model_internal_mae_across_seeds": per_model_across_seeds,
        "regime_b_fixed_summary": b_summary,
        "regime_c_fixed_summary": c_summary,
        "per_seed_regime_a_summary": per_seed_summary,
        "regime_a_stat_distribution_across_seeds": stat_distribution,
        "verdicts": verdicts,
    }


def merge_fragments() -> None:
    internal_by_seed_model: dict[int, dict[str, float]] = {seed: {} for seed in SEEDS}
    total_runtime = 0.0
    for seed in SEEDS:
        for model_name in MODEL_NAMES:
            frag_path = FRAGMENTS_DIR / f"seed{seed}_{model_name}.json"
            if not frag_path.exists():
                raise FileNotFoundError(
                    f"Missing fragment {frag_path}; run --seed {seed} --model {model_name} first."
                )
            frag = json.loads(frag_path.read_text(encoding="utf-8"))
            internal_by_seed_model[seed][model_name] = float(frag["regime_a"]["pooled_mae"])
            total_runtime += float(frag["runtime_seconds"])

    external_mae = load_external_mae()
    n4b_data = json.loads(N4B_ARTIFACT.read_text(encoding="utf-8"))
    b_internal = {
        m: float(n4b_data["results"][m]["B_family_group_kfold"]["pooled_mae"]) for m in MODEL_NAMES
    }
    c_internal = {
        m: float(n4b_data["results"][m]["C_lomfo"]["pooled_mae_over_major_family_rows"])
        for m in MODEL_NAMES
    }

    primary = _subset_analysis(
        "primary_8_model", MODEL_NAMES, internal_by_seed_model, external_mae, b_internal, c_internal
    )
    stable_six = _subset_analysis(
        "stable_6_model_sensitivity",
        STABLE_MODELS,
        internal_by_seed_model,
        external_mae,
        b_internal,
        c_internal,
    )

    manifest = {
        "phase": "N21_section_b_repeated_cv_sensitivity",
        "regime_utility_version": REGIME_UTILITY_VERSION,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "n_outer_splits_per_seed": 5,
        "development_population_only": True,
        "n6_external_rerun": False,
        "obelix_test_split_opened": False,
        "n4b_source": str(N4B_ARTIFACT.relative_to(ROOT)),
        "n6_source": str(N6_ARTIFACT.relative_to(ROOT)),
        "primary_8_model": primary,
        "stable_6_model_sensitivity": stable_six,
        "total_runtime_seconds": total_runtime,
    }
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    guard_output(ARTIFACT_JSON).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {ARTIFACT_JSON.relative_to(ROOT)}")
    _write_report(manifest)
    print(f"wrote {REPORT_MD.relative_to(ROOT)}")
    print(f"total runtime (summed across per-seed-model processes): {total_runtime:.1f}s")


def _write_report(manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# N21 Section B — Repeated Random-5-Fold CV Fold-Randomness Sensitivity\n\n")
    lines.append(
        f"Git commit: `{manifest['git_commit']}` on branch `{manifest['git_branch']}`\n\n"
        f"10 deterministic seeds ({manifest['seeds'][0]}..{manifest['seeds'][-1]}), 5 outer "
        "folds each, same nested per-fold Optuna tuning protocol as N4b. Only regime A "
        "(random 5-fold CV) "
        "was repeated; regimes B and C use their already-frozen N4b values as fixed comparators. "
        "N6 external predictions were not rerun; the frozen 113-row OBELiX test split was not "
        f"opened. Total compute: {manifest['total_runtime_seconds']:.0f}s "
        f"({manifest['total_runtime_seconds'] / 3600:.2f}h).\n\n"
    )

    for subset_key, heading in (
        ("primary_8_model", "Primary 8-model comparison"),
        ("stable_6_model_sensitivity", "Six-model stable-architecture sensitivity"),
    ):
        subset = manifest[subset_key]
        lines.append(f"\n## {heading} ({subset['n_models']} models)\n\n")
        lines.append("### Per-model internal (regime A) MAE across 10 seeds\n\n")
        lines.append("| Model | Mean | Median | SD | 95% interval | Original seed value |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for model, d in subset["per_model_internal_mae_across_seeds"].items():
            lines.append(
                f"| {model} | {d['mean']:.4f} | {d['median']:.4f} | {d['sd']:.4f} | "
                f"[{d['ci95_low']:.4f}, {d['ci95_high']:.4f}] | {d['original_seed_value']:.4f} |\n"
            )

        lines.append(
            "\n### Regime-A regime-utility statistics: distribution across 10 seeds vs. "
            "fixed B/C values\n\n"
        )
        lines.append(
            "| Statistic | A mean | A median | A SD | A 95% interval | A original-seed | "
            "B (fixed) | C (fixed) |\n"
        )
        lines.append("|---|---|---|---|---|---|---|---|\n")
        for stat, d in subset["regime_a_stat_distribution_across_seeds"].items():
            lines.append(
                f"| {stat} | {d['mean']:.4f} | {d['median']:.4f} | {d['sd']:.4f} | "
                f"[{d['ci95_low']:.4f}, {d['ci95_high']:.4f}] | {d['original_seed_value']:.4f} | "
                f"{d['b_fixed_value']:.4f} | {d['c_fixed_value']:.4f} |\n"
            )

        lines.append(
            "\n### Verdicts (does A's 10-seed distribution support a superiority claim?)\n\n"
        )
        lines.append(
            "| Statistic | Median favors A | 95% CI overlaps B | 95% CI overlaps C | "
            "Superiority claim supported |\n"
        )
        lines.append("|---|---|---|---|---|\n")
        for stat, v in subset["verdicts"].items():
            lines.append(
                f"| {stat} | {v['median_favors_a_over_b_and_c']} | "
                f"{v['ci95_overlaps_b_fixed_value']} | "
                f"{v['ci95_overlaps_c_fixed_value']} | {v['claim_a_superiority_supported']} |\n"
            )

        n_supported = sum(
            1 for v in subset["verdicts"].values() if v["claim_a_superiority_supported"]
        )
        lines.append(
            f"\n**{n_supported}/4 statistics support a fold-randomness-robust claim that regime "
            "A (random 5-fold CV) is favorable, under a strict no-material-overlap standard "
            "(median favors A AND the 95% CI does not overlap either fixed comparator), for the "
            f"{heading.lower()}.** Per the task instruction, superiority is NOT claimed for any "
            "statistic where the 95% interval materially overlaps a fixed B/C value, even if the "
            "median point estimate favors A.\n"
        )

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_MD).write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_fragments()
        return
    if args.seed is not None and args.model is not None:
        run_one(args.seed, args.model)
        return
    parser.error("pass either --seed <seed> --model <name>, or --merge")


if __name__ == "__main__":
    main()
