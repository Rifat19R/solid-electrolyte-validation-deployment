#!/usr/bin/env python3
"""Cross-dataset study, N18: forensic sanity check of the
Family-GroupKFold (regime B) max standardized composition-space NN distance
(~94,559.5, first reported in reports/cross_dataset_study/n15_reviewer_remediation.md).

STATUS: N18 FINAL-REFEREE DIAGNOSTIC ONLY. This is a metric sanity check,
not a new hypothesis search: it re-derives the exact same regime-B
fold splits and NN distances that scripts/transport_analysis/n15_reviewer_remediation.py
already computed and reported (same seed, same deterministic
GroupKFold/TrainFitCompositionScaler/fit_standardization call sequence), and
additionally (a) attributes the max distance to its originating fold/row and
dominant feature dimension(s), and (b) runs ONE diagnostic sensitivity that
recomputes regime B's NN distances after dropping zero/near-zero-variance
training-fold feature dimensions, to check whether the extreme is
scale-driven. No model is fit/retrained, no target y is read, and the
frozen OBELiX 113-row official test split is never opened -- identical
constraints to N15.

Run with: python scripts/transport_analysis/n18_nn_distance_forensic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.data.obelix_loader import load_obelix  # noqa: E402
from sse_transfer.data.regression_dataset import load_regression_dataset  # noqa: E402
from sse_transfer.evaluation.internal_validation import (  # noqa: E402
    N_OUTER_SPLITS,
    canonicalize_and_fill_family,
    family_group_kfold_splits,
)
from sse_transfer.features.composition_pipeline import (  # noqa: E402
    TrainFitCompositionScaler,
    composition_features,
)
from sse_transfer.uncertainty.applicability_domain import fit_standardization  # noqa: E402
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

ARTIFACTS_DIR = ROOT / "artifacts"
OUT_PATH = ROOT / "artifacts" / "rerun" / "n18_nn_distance_forensic.json"
CV_SEED = 20260921
NEAR_ZERO_ABS_THRESHOLD = 1e-8  # matches TrainFitCompositionScaler's own guard
# Relative threshold for the diagnostic sensitivity: a raw (pre-scaler) composition
# feature is treated as "degenerate-variance in this fold's training pool" if that
# fold's train-only std is less than 5% of the SAME feature's std computed over the
# full 449-row development population (target-free; no y is read). This is a much
# more informative test than the absolute 1e-8 guard already built into
# TrainFitCompositionScaler/fit_standardization, which only catches *exactly*
# zero-variance columns and misses columns that are merely near-degenerate within
# one particular family-holdout fold (see report: the true mechanism here).
DEGENERACY_RATIO_THRESHOLD = 0.05


def load_dev_population() -> pd.DataFrame:
    dataset = load_regression_dataset()
    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)
    raw_by_id = raw.set_index("ID")
    dev = pd.concat([dataset.train, dataset.validation], ignore_index=True)
    pool = dev[["ID", "y"]].copy()
    pool["true_composition"] = raw_by_id.loc[pool["ID"], "True Composition"].to_numpy()
    pool["family_filled"] = canonicalize_and_fill_family(
        raw_by_id.loc[pool["ID"], "Family"].reset_index(drop=True)
    ).to_numpy()
    return pool.reset_index(drop=True)


def main() -> None:
    pool = load_dev_population()
    comps = pool["true_composition"]
    families = pool["family_filled"].to_numpy()

    folds_b = family_group_kfold_splits(
        pool["family_filled"], n_splits=N_OUTER_SPLITS, seed=CV_SEED
    )

    per_row_records: list[dict[str, Any]] = []
    global_max = -1.0
    global_max_record: dict[str, Any] | None = None

    for fold_i, (train_idx, test_idx) in enumerate(folds_b):
        scaler = TrainFitCompositionScaler().fit(comps.iloc[train_idx])
        x_ref = scaler.transform(comps.iloc[train_idx])  # scaler-standardized (layer 1)
        x_query = scaler.transform(comps.iloc[test_idx])
        stats = fit_standardization(x_ref)  # second standardization layer, over x_ref
        cols = list(stats["columns"])
        mean = np.asarray(stats["mean"])
        std = np.asarray(stats["std"])  # already has the >0-else-1.0 guard applied

        # Raw (pre-fit_standardization, i.e. layer-1-only) train-fold std, to see
        # which columns were zero/near-zero variance *before* fit_standardization's
        # own guard kicked in a second time.
        raw_std_layer1 = x_ref.to_numpy(dtype=float).std(axis=0, ddof=0)
        near_zero_mask = raw_std_layer1 < NEAR_ZERO_ABS_THRESHOLD

        z_ref = (x_ref.to_numpy(dtype=float) - mean) / std
        z_query = (x_query.to_numpy(dtype=float) - mean) / std

        # nearest-neighbor distance + which train row is nearest + per-dim contribution
        for qi_local, qi_global in enumerate(test_idx):
            diffs = z_ref - z_query[qi_local]  # (n_train, n_features)
            sq = diffs**2
            dist_to_each_train = np.sqrt(sq.sum(axis=1))
            nn_pos = int(np.argmin(dist_to_each_train))
            dist = float(dist_to_each_train[nn_pos])
            contrib = sq[nn_pos]  # per-feature squared contribution to nearest neighbor
            top_order = np.argsort(-contrib)[:5]
            top_features = [
                {
                    "feature": cols[j],
                    "sq_contribution": float(contrib[j]),
                    "pct_of_total_sq": float(contrib[j] / max(sq[nn_pos].sum(), 1e-30) * 100.0),
                    "train_fold_mean_layer2": float(mean[j]),
                    "train_fold_std_layer2": float(std[j]),
                    "train_fold_raw_std_layer1_pre_guard": float(raw_std_layer1[j]),
                    "is_near_zero_variance_layer1": bool(near_zero_mask[j]),
                    "held_out_row_value_layer1_scaled": float(
                        x_query.to_numpy(dtype=float)[qi_local, j]
                    ),
                    "held_out_row_z_value": float(z_query[qi_local, j]),
                    "nearest_train_row_layer1_scaled": float(
                        x_ref.to_numpy(dtype=float)[nn_pos, j]
                    ),
                }
                for j in top_order
            ]
            rec = {
                "fold": fold_i,
                "pool_row_index": int(qi_global),
                "obelix_id": str(pool["ID"].iloc[qi_global]),
                "family": str(families[qi_global]),
                "distance": dist,
                "n_near_zero_variance_dims_in_train_fold": int(near_zero_mask.sum()),
                "n_total_dims": int(len(cols)),
                "top_contributing_features": top_features,
            }
            per_row_records.append(rec)
            if dist > global_max:
                global_max = dist
                global_max_record = rec

    assert global_max_record is not None
    reported_max = 94559.5361
    match_reported = abs(global_max - reported_max) < 0.5

    # Full-pool (all 449 rows, target-free) raw per-feature std, used only as a
    # fixed reference scale for the degeneracy-ratio diagnostic below.
    full_pool_raw = composition_features(comps)
    full_pool_std = full_pool_raw.to_numpy(dtype=float).std(axis=0, ddof=0)

    # --- ONE diagnostic sensitivity: for each fold, drop any raw composition
    # feature whose TRAIN-FOLD-ONLY std is < 5% of that feature's full-pool std
    # (i.e. this fold's family exclusion left that dimension functionally
    # degenerate) rather than falling back to scale=1.0/dividing by a tiny
    # nonzero std, then recompute the NN distance on the remaining dimensions.
    from scipy.spatial.distance import cdist

    sens_all_dist: list[float] = []
    sens_n_dropped_by_fold: dict[int, int] = {}
    for fold_i, (train_idx, test_idx) in enumerate(folds_b):
        raw_train = composition_features(comps.iloc[train_idx]).to_numpy(dtype=float)
        raw_query = composition_features(comps.iloc[test_idx]).to_numpy(dtype=float)
        medians = np.nanmedian(raw_train, axis=0)
        raw_train = np.where(np.isnan(raw_train), medians, raw_train)
        raw_query = np.where(np.isnan(raw_query), medians, raw_query)
        train_std = raw_train.std(axis=0, ddof=0)
        ratio = train_std / np.where(full_pool_std > 0, full_pool_std, 1.0)
        keep_mask = ratio >= DEGENERACY_RATIO_THRESHOLD
        sens_n_dropped_by_fold[fold_i] = int((~keep_mask).sum())

        x_ref_kept = raw_train[:, keep_mask]
        x_query_kept = raw_query[:, keep_mask]
        mean_k = x_ref_kept.mean(axis=0)
        std_k = x_ref_kept.std(axis=0, ddof=0)
        std_k_safe = np.where(std_k > 0, std_k, 1.0)
        z_ref_k = (x_ref_kept - mean_k) / std_k_safe
        z_query_k = (x_query_kept - mean_k) / std_k_safe
        d = cdist(z_query_k, z_ref_k, metric="euclidean").min(axis=1)
        sens_all_dist.extend(d.tolist())

    sens_arr = np.array(sens_all_dist, dtype=float)
    sensitivity = {
        "description": (
            "Regime B NN distance recomputed after DROPPING (not scale=1.0 "
            "fallback for) any raw composition-feature dimension whose "
            f"train-fold-only std is < {DEGENERACY_RATIO_THRESHOLD:.0%} of that "
            "same feature's std over the full 449-row target-free development "
            "population (a fold-local degenerate-variance dimension), per fold."
        ),
        "n_dims_dropped_by_fold": sens_n_dropped_by_fold,
        "n": int(sens_arr.size),
        "median": float(np.median(sens_arr)),
        "q90": float(np.quantile(sens_arr, 0.90)),
        "max": float(sens_arr.max()),
        "original_max_reported_n15": reported_max,
        "max_reduction_factor": float(reported_max / sens_arr.max())
        if sens_arr.max() > 0
        else None,
    }

    out = {
        "analysis": "N18_nn_distance_forensic_sanity_check",
        "cv_seed": CV_SEED,
        "reported_n15_max": reported_max,
        "recomputed_max": global_max,
        "recomputed_matches_reported": match_reported,
        "max_record": global_max_record,
        "n_folds": len(folds_b),
        "n_rows_scored": len(per_row_records),
        "diagnostic_sensitivity_drop_near_zero_variance_dims": sensitivity,
        "note": (
            "Diagnostic sanity check only, per N18 instruction item 3. Does not "
            "alter N15's reported numbers, does not rerun N6, does not read any "
            "target y, and does not open the frozen OBELiX 113-row test split."
        ),
    }
    guard_output(OUT_PATH).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Recomputed regime-B max distance: {global_max:.4f} (N15 reported: {reported_max})")
    print(f"Match: {match_reported}")
    print(
        f"Max record: fold={global_max_record['fold']}, "
        f"obelix_id={global_max_record['obelix_id']}, "
        f"family={global_max_record['family']}"
    )
    print("Top contributing features:")
    for f in global_max_record["top_contributing_features"]:
        print(
            f"  {f['feature']}: {f['pct_of_total_sq']:.2f}% of sq-distance, "
            f"near_zero_var={f['is_near_zero_variance_layer1']}, "
            f"train_mean={f['train_fold_mean_layer2']:.6g}, "
            f"train_std_layer2={f['train_fold_std_layer2']:.6g}, "
            f"raw_std_layer1_pre_guard={f['train_fold_raw_std_layer1_pre_guard']:.6g}, "
            f"held_out_value={f['held_out_row_value_layer1_scaled']:.6g}, "
            f"held_out_z={f['held_out_row_z_value']:.6g}"
        )
    print()
    print(
        f"Sensitivity (drop near-zero-var dims): max={sensitivity['max']:.4f}, "
        f"q90={sensitivity['q90']:.4f}, reduction_factor={sensitivity['max_reduction_factor']}"
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
