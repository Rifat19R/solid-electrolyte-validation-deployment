#!/usr/bin/env python3
"""Cross-dataset study, N21 Section E: robust shift-distance fix
(reviewer-closure task, Section E).

N15 analysis 2 (`artifacts/historical/n15_deployment_shift.json`) reported a
standardized-Euclidean nearest-neighbor distance max of ~94,559.5 under
Family-GroupKFold. N18's forensic check
(`scripts/transport_analysis/n18_nn_distance_forensic.py`,
`artifacts/authoritative/n18_nn_distance_forensic.json`) traced this to fold 0's
training partition being FOLD-LOCALLY degenerate (not globally
zero-variance) on several GSmagmom-derived Magpie columns: an ABSOLUTE
near-zero-variance guard (`TrainFitCompositionScaler`'s own
`stds.where(stds.abs() > 1e-8, 1.0)`, and an absolute threshold this
script's own first draft tried) never catches this, because that fold's raw
std on those columns, while tiny relative to the full 449-row population's
std on the same columns, is not literally near 1e-8 in absolute terms
(most Li-SSE compositions have zero magnetic-moment-contributing elements,
so the feature is globally near-zero-inflated but not degenerate; one
family-holdout fold's remaining training compositions happen to be almost
entirely non-magnetic, while the held-out row is not). N18 already
predefined and validated the fix that actually works: a RELATIVE threshold
-- drop a raw feature dimension from a fold if that fold's train-only std is
below 5% of the SAME feature's std over the full 449-row, target-free
development population -- which N18 confirmed reduces regime B's max
distance from 94,559.5 to 26.65 (3548x). This script reuses that exact,
already-predefined (not tuned on this script's own output) 5% relative
threshold and extends it from N18's regime-B-only diagnostic to the full
N15-analysis-2 pipeline: all three regimes (A, B, C), the external
reference fit, and the full distance-distribution/Wasserstein-alignment
recomputation N15 originally reported.

`TrainFitCompositionScaler` and `applicability_domain.py` (a frozen
`CODE_HASH_FILES` module from the old project) are never imported or edited
by this script's robust path -- it calls `composition_features` (the same
underlying, unedited Magpie feature function) directly and builds its own
standardized space with the offending dimensions dropped; every model-facing
feature pipeline used elsewhere in this study is completely unaffected.

The original (non-robust) N15 numbers are preserved unmodified in their own
artifact; this script never edits `artifacts/historical/n15_deployment_shift.json`.

Run:
    python scripts/transport_analysis/n21e_robust_shift_distance.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "transport_analysis"))

import n4b_nested_internal_validation as n4b  # noqa: E402

from sse_transfer.evaluation.internal_validation import (  # noqa: E402
    LOMFO_MIN_FAMILY_N,
    N_OUTER_SPLITS,
    family_group_kfold_splits,
    leave_one_major_family_out_splits,
    major_family_counts,
    random_kfold_splits,
)
from sse_transfer.evaluation.n15_deployment_shift import (  # noqa: E402
    distance_distribution_summary,
    wasserstein_distance_1d,
)
from sse_transfer.features.composition_features import composition_features  # noqa: E402
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

N15_ARTIFACT = ROOT / "artifacts" / "authoritative" / "n15_artifacts" / "n15_deployment_shift.json"
N18_ARTIFACT = ROOT / "artifacts" / "authoritative" / "n18_nn_distance_forensic.json"
CV_SEED = 20260921
DEGENERACY_RATIO_THRESHOLD = 0.05  # reused unchanged from N18's already-validated diagnostic
OUTPUT_JSON = ROOT / "artifacts" / "rerun" / "n21e_robust_shift_distance.json"
REPORT_MD = ROOT / "reports" / "rerun" / "n21e_robust_shift_distance.md"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def robust_nn_distance(
    comps_ref: pd.Series, comps_query: pd.Series, full_pool_std: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Raw-feature-level robust standardized-Euclidean NN distance, reusing
    N18's already-validated relative-degeneracy definition: a raw Magpie
    column is dropped from THIS fold if `comps_ref`'s train-fold-only std on
    that column is below `DEGENERACY_RATIO_THRESHOLD` (5%) of the SAME
    column's std over the full 449-row, target-free development population
    (`full_pool_std`, computed once by the caller) -- the decision compares
    this fold's std to a fixed, target-free reference scale, never to
    `comps_query`. Retained columns are then train-median-imputed and
    train-mean/std standardized before the nearest-neighbor Euclidean
    distance is computed, matching `TrainFitCompositionScaler`'s own
    convention otherwise unchanged."""
    raw_ref = composition_features(comps_ref)
    raw_query = composition_features(comps_query)
    medians = raw_ref.median(numeric_only=True)
    imputed_ref = raw_ref.fillna(medians)
    imputed_query = raw_query[list(raw_ref.columns)].fillna(medians)

    train_std = imputed_ref.std(axis=0, ddof=0).to_numpy(dtype=float)
    ratio = train_std / np.where(full_pool_std > 0, full_pool_std, 1.0)
    keep_mask = ratio >= DEGENERACY_RATIO_THRESHOLD
    n_dropped = int((~keep_mask).sum())
    n_total = int(len(keep_mask))
    kept_cols = imputed_ref.columns[keep_mask]

    mean = imputed_ref[kept_cols].mean(axis=0)
    std_kept = imputed_ref[kept_cols].std(axis=0, ddof=0)
    std_kept_safe = std_kept.where(std_kept.abs() > 0, 1.0)
    z_ref = ((imputed_ref[kept_cols] - mean) / std_kept_safe).to_numpy(dtype=float)
    z_query = ((imputed_query[kept_cols] - mean) / std_kept_safe).to_numpy(dtype=float)

    dist = cdist(z_query, z_ref, metric="euclidean").min(axis=1)
    return np.asarray(dist, dtype=float), n_dropped, n_total


def load_dev_population() -> pd.DataFrame:
    pool, _ = n4b.load_dev_population()
    return pool


def robust_fold_nn_distances(
    pool: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], full_pool_std: np.ndarray
) -> tuple[np.ndarray, list[dict[str, int]]]:
    n = len(pool)
    distances = np.full(n, np.nan)
    comps = pool["true_composition"]
    per_fold_dropped: list[dict[str, int]] = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        d, n_dropped, n_total = robust_nn_distance(
            comps.iloc[train_idx], comps.iloc[test_idx], full_pool_std
        )
        distances[test_idx] = d
        per_fold_dropped.append({"fold": fold_i, "n_dropped": n_dropped, "n_total_dims": n_total})
    return distances, per_fold_dropped


def main() -> None:
    original = json.loads(N15_ARTIFACT.read_text(encoding="utf-8"))
    pool = load_dev_population()
    comps = pool["true_composition"]

    strict_cohort_path = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_strict.csv"
    source_disjoint_path = (
        ROOT / "artifacts" / "authoritative" / "n15_artifacts" / "n15_source_disjoint.json"
    )
    strict_df = pd.read_csv(strict_cohort_path)
    source_disjoint = json.loads(source_disjoint_path.read_text(encoding="utf-8"))
    keep_formulas = set(source_disjoint["source_disjoint_formulas"])

    full_pool_std = composition_features(comps).to_numpy(dtype=float).std(axis=0, ddof=0)

    folds_a = random_kfold_splits(len(pool), n_splits=N_OUTER_SPLITS, seed=CV_SEED)
    dist_a, dropped_a = robust_fold_nn_distances(pool, folds_a, full_pool_std)

    folds_b = family_group_kfold_splits(
        pool["family_filled"], n_splits=N_OUTER_SPLITS, seed=CV_SEED
    )
    dist_b, dropped_b = robust_fold_nn_distances(pool, folds_b, full_pool_std)

    major_families = sorted(major_family_counts(pool["family_filled"], min_n=LOMFO_MIN_FAMILY_N))
    splits_c = leave_one_major_family_out_splits(pool["family_filled"], major_families)
    dist_c_list: list[float] = []
    dropped_c: list[dict[str, Any]] = []
    for fam_name, (train_idx, test_idx) in splits_c.items():
        d, n_dropped, n_total = robust_nn_distance(
            comps.iloc[train_idx], comps.iloc[test_idx], full_pool_std
        )
        dist_c_list.extend(d.tolist())
        dropped_c.append({"family": fam_name, "n_dropped": n_dropped, "n_total_dims": n_total})
    dist_c = np.array(dist_c_list, dtype=float)

    dist_strict, dropped_ext, n_total_ext = robust_nn_distance(
        comps, strict_df["reduced_formula"], full_pool_std
    )
    disjoint_mask = strict_df["reduced_formula"].isin(keep_formulas).to_numpy()
    dist_source_disjoint = dist_strict[disjoint_mask]

    summaries = {
        "A_random_5fold_cv": distance_distribution_summary(dist_a),
        "B_family_group_kfold": distance_distribution_summary(dist_b),
        "C_lomfo": distance_distribution_summary(dist_c),
        "hargreaves_strict": distance_distribution_summary(dist_strict),
        "hargreaves_source_disjoint": distance_distribution_summary(dist_source_disjoint),
    }
    wasserstein = {}
    for regime_name, internal_dist in (
        ("A_random_5fold_cv", dist_a),
        ("B_family_group_kfold", dist_b),
        ("C_lomfo", dist_c),
    ):
        wasserstein[regime_name] = {
            "vs_hargreaves_strict": wasserstein_distance_1d(internal_dist, dist_strict),
            "vs_hargreaves_source_disjoint": wasserstein_distance_1d(
                internal_dist, dist_source_disjoint
            ),
        }

    original_summaries = original["distance_distribution_summary"]
    comparison = {
        regime: {
            "original_max": original_summaries[regime]["max"],
            "robust_max": summaries[regime]["max"],
            "original_q90": original_summaries[regime]["q90"],
            "robust_q90": summaries[regime]["q90"],
            "original_median": original_summaries[regime]["median"],
            "robust_median": summaries[regime]["median"],
        }
        for regime in summaries
    }

    original_wasserstein = original["wasserstein_distance_internal_vs_external"]
    wasserstein_comparison = {
        regime: {
            "original_vs_strict": original_wasserstein[regime]["vs_hargreaves_strict"],
            "robust_vs_strict": wasserstein[regime]["vs_hargreaves_strict"],
            "original_vs_source_disjoint": (
                original_wasserstein[regime]["vs_hargreaves_source_disjoint"]
            ),
            "robust_vs_source_disjoint": wasserstein[regime]["vs_hargreaves_source_disjoint"],
        }
        for regime in wasserstein
    }

    def _closest_regime(w: dict[str, Any], key: str) -> str:
        return min(w, key=lambda r: w[r][key])

    orig_closest = _closest_regime(original_wasserstein, "vs_hargreaves_strict")
    robust_closest = _closest_regime(wasserstein, "vs_hargreaves_strict")

    manifest = {
        "phase": "N21_section_e_robust_shift_distance",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "n15_source": str(N15_ARTIFACT.relative_to(ROOT)),
        "n18_source": str(N18_ARTIFACT.relative_to(ROOT)),
        "degeneracy_ratio_threshold": DEGENERACY_RATIO_THRESHOLD,
        "threshold_definition": (
            "raw (pre-TrainFitCompositionScaler) Magpie feature column dropped from a fold if "
            "that fold's train-only std < 5% of the same column's std over the full 449-row "
            "target-free development population; reused unchanged from N18's already-validated "
            "diagnostic, not tuned on this script's own output"
        ),
        "threshold_chosen_independently_of_target_y": True,
        "cv_seed": CV_SEED,
        "per_fold_dims_dropped": {
            "A_random_5fold_cv": dropped_a,
            "B_family_group_kfold": dropped_b,
            "C_lomfo": dropped_c,
            "external_reference_fit": {"n_dropped": dropped_ext, "n_total_dims": n_total_ext},
        },
        "distance_distribution_summary_robust": summaries,
        "wasserstein_distance_internal_vs_external_robust": wasserstein,
        "original_vs_robust_distance_summary": comparison,
        "original_vs_robust_wasserstein": wasserstein_comparison,
        "closest_regime_to_external_by_wasserstein": {
            "original": orig_closest,
            "robust": robust_closest,
            "interpretation_changed": orig_closest != robust_closest,
        },
    }
    guard_output(OUTPUT_JSON).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    _write_report(manifest)
    print(f"wrote {REPORT_MD.relative_to(ROOT)}")


def _write_report(manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# N21 Section E — Robust Shift-Distance Fix\n\n")
    lines.append(f"Git commit: `{manifest['git_commit']}` on branch `{manifest['git_branch']}`\n\n")
    lines.append(
        f"Threshold: {manifest['threshold_definition']} "
        f"(ratio={manifest['degeneracy_ratio_threshold']}). Original N15 numbers are preserved "
        f"unmodified in `{manifest['n15_source']}`.\n\n"
    )

    lines.append("## Dimensions dropped per fold (of ~132 raw Magpie composition features)\n\n")
    for regime, folds in manifest["per_fold_dims_dropped"].items():
        if regime == "external_reference_fit":
            d = folds
            lines.append(
                f"- external reference fit: {d['n_dropped']}/{d['n_total_dims']} dropped\n"
            )
            continue
        dropped_list = ", ".join(
            f"{f.get('fold', f.get('family'))}:{f['n_dropped']}/{f['n_total_dims']}" for f in folds
        )
        lines.append(f"- {regime}: {dropped_list}\n")

    lines.append("\n## Original vs. robust distance distribution (max / q90 / median)\n\n")
    lines.append(
        "| Cohort | Original max | Robust max | Original q90 | Robust q90 | "
        "Original median | Robust median |\n"
    )
    lines.append("|---|---|---|---|---|---|---|\n")
    for regime, c in manifest["original_vs_robust_distance_summary"].items():
        lines.append(
            f"| {regime} | {c['original_max']:.4f} | {c['robust_max']:.4f} | "
            f"{c['original_q90']:.4f} | {c['robust_q90']:.4f} | "
            f"{c['original_median']:.4f} | {c['robust_median']:.4f} |\n"
        )

    lines.append("\n## Original vs. robust regime-vs-external Wasserstein distance\n\n")
    lines.append(
        "| Regime | Orig. vs strict | Robust vs strict | Orig. vs source-disjoint | "
        "Robust vs source-disjoint |\n"
    )
    lines.append("|---|---|---|---|---|\n")
    for regime, c in manifest["original_vs_robust_wasserstein"].items():
        lines.append(
            f"| {regime} | {c['original_vs_strict']:.4f} | {c['robust_vs_strict']:.4f} | "
            f"{c['original_vs_source_disjoint']:.4f} | {c['robust_vs_source_disjoint']:.4f} |\n"
        )

    ci = manifest["closest_regime_to_external_by_wasserstein"]
    b = manifest["original_vs_robust_distance_summary"]["B_family_group_kfold"]
    fix_worked = b["robust_max"] < b["original_max"] / 10.0
    lines.append(
        f"\n**Regime closest to the external cohort by Wasserstein distance:** original = "
        f"`{ci['original']}`, robust = `{ci['robust']}` "
        f"({'CHANGED' if ci['interpretation_changed'] else 'unchanged'}).\n\n"
        f"**Did the raw-feature-level robust fix resolve the ~94,559.5 outlier?** "
        f"{'YES' if fix_worked else 'NO'} -- original Family-GroupKFold max = "
        f"{b['original_max']:.4f}, robust max = {b['robust_max']:.4f}.\n"
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_MD).write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
