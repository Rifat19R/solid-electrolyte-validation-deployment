#!/usr/bin/env python3
"""Reviewer remediation Issue 05: strictly fold-local robust shift-distance
metric.

Spec: reports/remediation/specs/issue05_fold_local_shift_distance.md (hashed before
this script was written).

**What was wrong.** `n21e_robust_shift_distance.py`'s near-degenerate-
dimension guard drops a fold's feature column when that fold's train-only
std is below 5% of the SAME column's std over the full 449-row development
population (`full_pool_std`, computed once over ALL rows and reused
identically for every fold of every regime). That reference includes the
current outer fold's own held-out/test rows' feature values -- not labels
(the diagnostic is and remains fully target-free), but it is not strictly
fold-local either, and for regime B's fold 0 in particular (which excludes
213/449 = 47.4% of the pool), that fold's own huge exclusion measurably
shifts the very population statistic used to judge it.

**Why a truly fold-blind reference is mathematically impossible here (this
matters, so it is documented rather than glossed over).** For any K-fold
partition of a fixed pool, "not in fold k's test set" is BY DEFINITION
identical to "in fold k's training set" -- so a reference computed only
from data outside fold k's test rows is either (a) exactly fold k's own
training partition, making the ratio check trivially 1.0 and useless, or
(b) built from other folds, which (again by K-fold construction) train on
fold k's test rows -- meaning ANY non-trivial population-style reference
necessarily incorporates fold k's rows somewhere. There is no escape from
this within a single K-fold partition of one dataset.

**The fix actually implemented.** Replace the single shared `full_pool_std`
with, for each fold k, the MEDIAN across every OTHER fold j != k (same
regime) of fold j's own train-only std. This does not achieve theoretically
perfect fold-k-blindness (shown above to be impossible), but it removes the
specific, meaningful problem this issue is about: fold k's OWN exclusion no
longer single-handedly determines the reference it is judged against. A
fold with an unusually large exclusion (regime B fold 0) is now judged
against a reference built from the OTHER folds' (much smaller, more
typical) exclusions, not against a population statistic it itself
dominates. Threshold ratio (5%) and every other detail of N18/N21e's
already-validated method are unchanged.

Run: python scripts/analysis/issue05_fold_local_shift_distance.py
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

N21E_ARTIFACT = ROOT / "artifacts" / "authoritative" / "n21e_robust_shift_distance.json"
CV_SEED = 20260921
DEGENERACY_RATIO_THRESHOLD = 0.05  # unchanged from N18/N21e
OUTPUT_JSON = ROOT / "artifacts" / "rerun" / "issue05_fold_local_shift_distance.json"
REPORT_MD = ROOT / "reports" / "rerun" / "issue05_fold_local_shift_distance.md"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def _train_std(comps_train: pd.Series, ref_columns: list[str] | None = None) -> pd.Series:
    raw = composition_features(comps_train)
    if ref_columns is not None:
        raw = raw[ref_columns]
    medians = raw.median(numeric_only=True)
    imputed = raw.fillna(medians)
    return imputed.std(axis=0, ddof=0)


def fold_local_nn_distance(
    comps_ref: pd.Series, comps_query: pd.Series, reference_std: pd.Series
) -> tuple[np.ndarray, int, int]:
    """Identical to n21e's `robust_nn_distance` except the denominator is
    `reference_std` (a leave-fold-out median of sibling folds' train stds),
    not a single shared full-population std."""
    raw_ref = composition_features(comps_ref)
    raw_query = composition_features(comps_query)
    medians = raw_ref.median(numeric_only=True)
    imputed_ref = raw_ref.fillna(medians)
    imputed_query = raw_query[list(raw_ref.columns)].fillna(medians)

    train_std = imputed_ref.std(axis=0, ddof=0)
    ref_std_aligned = reference_std.reindex(train_std.index)
    ratio = train_std / ref_std_aligned.where(ref_std_aligned > 0, 1.0)
    keep_mask = (ratio >= DEGENERACY_RATIO_THRESHOLD).to_numpy()
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


def leave_fold_out_reference_stds(
    comps: pd.Series, folds: list[tuple[np.ndarray, np.ndarray]]
) -> list[pd.Series]:
    """For each fold k, the median (over every OTHER fold j != k) of fold
    j's own train-only std, per feature column. len(folds) must be >= 3 for
    this median to be meaningful (>=2 other folds); this study's regimes
    all have >=5 outer partitions (A/B: 5 folds; C: 9 major-family
    exclusions), so this always holds."""
    if len(folds) < 3:
        raise ValueError("leave_fold_out_reference_stds: need >=3 folds for a meaningful median")
    all_train_stds = [_train_std(comps.iloc[train_idx]) for train_idx, _ in folds]
    refs: list[pd.Series] = []
    for k in range(len(folds)):
        others = [all_train_stds[j] for j in range(len(folds)) if j != k]
        refs.append(pd.concat(others, axis=1).median(axis=1))
    return refs


def robust_fold_nn_distances(
    pool: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, list[dict[str, int]]]:
    n = len(pool)
    distances = np.full(n, np.nan)
    comps = pool["true_composition"]
    reference_stds = leave_fold_out_reference_stds(comps, folds)
    per_fold_dropped: list[dict[str, int]] = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        d, n_dropped, n_total = fold_local_nn_distance(
            comps.iloc[train_idx], comps.iloc[test_idx], reference_stds[fold_i]
        )
        distances[test_idx] = d
        per_fold_dropped.append({"fold": fold_i, "n_dropped": n_dropped, "n_total_dims": n_total})
    return distances, per_fold_dropped


def main() -> None:
    original = json.loads(N21E_ARTIFACT.read_text(encoding="utf-8"))
    pool, _ = n4b.load_dev_population()
    comps = pool["true_composition"]

    strict_cohort_path = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_strict.csv"
    source_disjoint_path = (
        ROOT / "artifacts" / "authoritative" / "n15_artifacts" / "n15_source_disjoint.json"
    )
    strict_df = pd.read_csv(strict_cohort_path)
    source_disjoint = json.loads(source_disjoint_path.read_text(encoding="utf-8"))
    keep_formulas = set(source_disjoint["source_disjoint_formulas"])

    folds_a = random_kfold_splits(len(pool), n_splits=N_OUTER_SPLITS, seed=CV_SEED)
    dist_a, dropped_a = robust_fold_nn_distances(pool, folds_a)

    folds_b = family_group_kfold_splits(
        pool["family_filled"], n_splits=N_OUTER_SPLITS, seed=CV_SEED
    )
    dist_b, dropped_b = robust_fold_nn_distances(pool, folds_b)

    major_families = sorted(major_family_counts(pool["family_filled"], min_n=LOMFO_MIN_FAMILY_N))
    splits_c = leave_one_major_family_out_splits(pool["family_filled"], major_families)
    folds_c_list = list(splits_c.items())
    folds_c_only = [v for _, v in folds_c_list]
    reference_stds_c = leave_fold_out_reference_stds(comps, folds_c_only)
    dist_c_list: list[float] = []
    dropped_c: list[dict[str, Any]] = []
    for i, (fam_name, (train_idx, test_idx)) in enumerate(folds_c_list):
        d, n_dropped, n_total = fold_local_nn_distance(
            comps.iloc[train_idx], comps.iloc[test_idx], reference_stds_c[i]
        )
        dist_c_list.extend(d.tolist())
        dropped_c.append({"family": fam_name, "n_dropped": n_dropped, "n_total_dims": n_total})
    dist_c = np.array(dist_c_list, dtype=float)

    # External reference fit: not fold-specific (fits on the full 449-row
    # pool as a single reference, no outer-test exclusion of its own), so
    # the original full-population std is the correct and only sensible
    # denominator here -- this is NOT a per-fold decision, so Issue 05's
    # concern does not apply to it. Kept identical to n21e.
    full_pool_std = composition_features(comps).to_numpy(dtype=float).std(axis=0, ddof=0)
    raw_ref = composition_features(comps)
    medians_ext = raw_ref.median(numeric_only=True)
    imputed_ref_ext = raw_ref.fillna(medians_ext)
    train_std_ext = imputed_ref_ext.std(axis=0, ddof=0).to_numpy(dtype=float)
    ratio_ext = train_std_ext / np.where(full_pool_std > 0, full_pool_std, 1.0)
    keep_ext = ratio_ext >= DEGENERACY_RATIO_THRESHOLD
    n_dropped_ext = int((~keep_ext).sum())
    n_total_ext = int(len(keep_ext))
    kept_cols_ext = imputed_ref_ext.columns[keep_ext]
    mean_ext = imputed_ref_ext[kept_cols_ext].mean(axis=0)
    std_ext = imputed_ref_ext[kept_cols_ext].std(axis=0, ddof=0)
    std_ext_safe = std_ext.where(std_ext.abs() > 0, 1.0)
    z_ref_ext = ((imputed_ref_ext[kept_cols_ext] - mean_ext) / std_ext_safe).to_numpy(dtype=float)
    raw_query_strict = composition_features(strict_df["reduced_formula"])
    imputed_query_strict = raw_query_strict[list(raw_ref.columns)].fillna(medians_ext)
    z_query_strict = ((imputed_query_strict[kept_cols_ext] - mean_ext) / std_ext_safe).to_numpy(
        dtype=float
    )
    dist_strict = cdist(z_query_strict, z_ref_ext, metric="euclidean").min(axis=1)
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

    n21e_summaries = original["distance_distribution_summary_robust"]
    comparison = {
        regime: {
            "n21e_shared_pop_ref_max": n21e_summaries[regime]["max"],
            "acs05_fold_local_ref_max": summaries[regime]["max"],
            "n21e_shared_pop_ref_q90": n21e_summaries[regime]["q90"],
            "acs05_fold_local_ref_q90": summaries[regime]["q90"],
            "n21e_shared_pop_ref_median": n21e_summaries[regime]["median"],
            "acs05_fold_local_ref_median": summaries[regime]["median"],
        }
        for regime in summaries
    }
    n21e_wasserstein = original["wasserstein_distance_internal_vs_external_robust"]
    wasserstein_comparison = {
        regime: {
            "n21e_vs_strict": n21e_wasserstein[regime]["vs_hargreaves_strict"],
            "acs05_vs_strict": wasserstein[regime]["vs_hargreaves_strict"],
            "n21e_vs_source_disjoint": n21e_wasserstein[regime]["vs_hargreaves_source_disjoint"],
            "acs05_vs_source_disjoint": wasserstein[regime]["vs_hargreaves_source_disjoint"],
        }
        for regime in wasserstein
    }

    def _closest_regime(w: dict[str, Any], key: str) -> str:
        return min(w, key=lambda r: w[r][key])

    n21e_closest = _closest_regime(n21e_wasserstein, "vs_hargreaves_strict")
    acs05_closest = _closest_regime(wasserstein, "vs_hargreaves_strict")

    manifest = {
        "phase": "acs_issue05_fold_local_shift_distance",
        "spec_path": "reports/remediation/specs/issue05_fold_local_shift_distance.md",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "n21e_source": str(N21E_ARTIFACT.relative_to(ROOT)),
        "degeneracy_ratio_threshold": DEGENERACY_RATIO_THRESHOLD,
        "threshold_definition_acs05": (
            "raw Magpie feature column dropped from fold k if fold k's train-only std < 5% of "
            "the MEDIAN, across every OTHER fold j!=k of the same regime, of fold j's own "
            "train-only std for that column -- replaces n21e's single shared full-449-row-"
            "population reference, which included every fold's own held-out rows' feature "
            "values. Still fully target-free. See script docstring for why a theoretically "
            "perfect fold-k-blind reference is not constructible for a K-fold partition."
        ),
        "threshold_chosen_independently_of_target_y": True,
        "per_fold_dims_dropped": {
            "A_random_5fold_cv": dropped_a,
            "B_family_group_kfold": dropped_b,
            "C_lomfo": dropped_c,
            "external_reference_fit": {"n_dropped": n_dropped_ext, "n_total_dims": n_total_ext},
        },
        "distance_distribution_summary_acs05": summaries,
        "wasserstein_distance_internal_vs_external_acs05": wasserstein,
        "n21e_vs_acs05_distance_summary": comparison,
        "n21e_vs_acs05_wasserstein": wasserstein_comparison,
        "closest_regime_to_external_by_wasserstein": {
            "n21e_shared_population_reference": n21e_closest,
            "acs05_fold_local_reference": acs05_closest,
            "interpretation_changed": n21e_closest != acs05_closest,
        },
    }
    guard_output(OUTPUT_JSON).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {OUTPUT_JSON}")
    print(f"closest regime: n21e={n21e_closest}, acs05={acs05_closest}")

    b = comparison["B_family_group_kfold"]
    lines = [
        "# Issue 05 — strictly fold-local robust shift-distance metric",
        "",
        "**Spec:** `reports/remediation/specs/issue05_fold_local_shift_distance.md`.",
        "",
        "Confirmed real gap: n21e's 5% degeneracy-guard denominator was the full "
        "449-row population std (shared, identical across every fold), which includes "
        "each fold's own held-out rows' feature values. Fixed by using, for each fold k, "
        "the median of every OTHER fold's own train-only std as the reference "
        "(a theoretically perfect fold-k-blind reference is not constructible for a "
        "K-fold partition -- see script docstring for the set-theoretic reason).",
        "",
        "## Distance summary: n21e (shared population reference) vs. this fix "
        "(leave-fold-out reference)",
        "",
        "| Regime | n21e max | acs05 max | n21e q90 | acs05 q90 | n21e median | acs05 median |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for regime, c in comparison.items():
        lines.append(
            f"| {regime} | {c['n21e_shared_pop_ref_max']:.4f} | "
            f"{c['acs05_fold_local_ref_max']:.4f} | "
            f"{c['n21e_shared_pop_ref_q90']:.4f} | {c['acs05_fold_local_ref_q90']:.4f} | "
            f"{c['n21e_shared_pop_ref_median']:.4f} | {c['acs05_fold_local_ref_median']:.4f} |"
        )
    lines.append("")
    lines.append(
        f"**Regime closest to external by Wasserstein distance:** n21e = "
        f"`{n21e_closest}`, acs05 = `{acs05_closest}` "
        f"({'CHANGED' if n21e_closest != acs05_closest else 'unchanged'})."
    )
    lines.append("")
    lines.append(
        f"**Family-GroupKFold max distance:** n21e (shared-population reference) = "
        f"{b['n21e_shared_pop_ref_max']:.4f}, acs05 (leave-fold-out reference) = "
        f"{b['acs05_fold_local_ref_max']:.4f}."
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
