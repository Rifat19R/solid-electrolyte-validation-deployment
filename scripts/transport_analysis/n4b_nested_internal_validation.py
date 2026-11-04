#!/usr/bin/env python3
"""Cross-dataset study, N4b: NESTED internal validation regimes
(post-lock methodological remediation of N4).

**Why this script exists (leakage finding, not a new experiment):** N3
tuned each of the 8 portable models' hyperparameters ONCE against the
OBELiX train(359)/validation(90) split, then froze them. N4 reused those
same frozen hyperparameters across every outer-CV fold built from
train UNION validation (449 rows). Because N3's 90 "validation" rows
influenced which hyperparameters were frozen, and many of those same rows
later served as OUTER-TEST rows in N4's regimes A/B/C, N4's internal-CV
numbers carry an optimism bias: outer-test labels had already partially
shaped model selection before being scored. This is a within-project
methodological correction, run once, before any external (Hargreaves)
label is touched -- N6's one-shot external inference/results are NOT
rerun here; N4b only recomputes the internal (OBELiX-only) side of the
old-vs-nested comparison the task requires.

**Nested fix:** for every outer fold in every regime, hyperparameters are
now tuned using an inner (inner-train, inner-validation) split carved out
of THAT FOLD'S OWN outer-train partition only (`tune_fold.py`, N3's
exact search spaces/builders/`N_TRIALS`/`TORCH_N_TRIALS` budgets and
tuning-seed convention) -- the outer-test fold is never seen by tuning,
and for regime C (LOMFO) the held-out major family is excluded from
outer-train entirely, so it cannot appear in inner tuning either.

Regimes (identical definitions/thresholds to N4 -- only the hyperparameter
source changed):
  A. Random 5-fold CV
  B. Canonical Family-GroupKFold
  C. Leave-one-major-family-out (families with >=15 rows)

Same leakage boundary as N3/N4: the frozen 113-row OBELiX `test` split is
read only for its length; no Hargreaves file is opened at all.

Run per-model (own OS process, same OOM-avoidance rationale as N4):
    python scripts/transport_analysis/n4b_nested_internal_validation.py --model <name>
Then merge:
    python scripts/transport_analysis/n4b_nested_internal_validation.py --merge
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.data.obelix_loader import load_obelix  # noqa: E402
from sse_transfer.data.regression_dataset import load_regression_dataset  # noqa: E402
from sse_transfer.evaluation.family_robustness import (  # noqa: E402
    macro_family_mae,
    median_family_mae,
    per_family_mae,
    sample_weighted_mae,
)
from sse_transfer.evaluation.internal_validation import (  # noqa: E402
    LOMFO_MIN_FAMILY_N,
    N_OUTER_SPLITS,
    canonicalize_and_fill_family,
    family_group_kfold_splits,
    inner_early_stop_split,
    leave_one_major_family_out_splits,
    major_family_counts,
    random_kfold_splits,
)
from sse_transfer.evaluation.metrics import regression_metrics  # noqa: E402
from sse_transfer.models.ladder import MODEL_NAMES, fit_predict_model  # noqa: E402
from sse_transfer.training.tune_fold import tune_hyperparameters  # noqa: E402
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

ARTIFACT_JSON = ROOT / "artifacts" / "rerun" / "n4b_nested_internal_validation_corrected.json"
REPORT_MD = ROOT / "reports" / "rerun" / "n4b_nested_internal_validation.md"
FRAGMENTS_DIR = ROOT / "artifacts" / "fragments" / "n4b"
# Recorded fragments (read by --merge) are immutable; --model reruns write here.
FRAGMENT_OUT_DIR = ROOT / "artifacts" / "rerun" / "fragments" / "n4b"
OLD_N4_ARTIFACT = ROOT / "artifacts" / "frozen" / "n4_internal_validation.json"

CV_SEED = 20260921  # same fixed seed as N3 (tuning) / N4 (fold construction).
INNER_TUNING_STOP_FRACTION = 0.2  # same fraction as fit_predict_model's own
# torch early-stopping carve-out; reused here for the tuning inner-split too.


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def load_dev_population() -> tuple[pd.DataFrame, int]:
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
    return pool.reset_index(drop=True), len(dataset.test)


def _oof_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, family_filled: np.ndarray
) -> dict[str, Any]:
    base = regression_metrics(y_true, y_pred)
    per_fam = per_family_mae(y_true, y_pred, family_filled)
    return {
        "pooled_mae": base["mae"],
        "rmse": base["rmse"],
        "r2": base["r2"],
        "spearman_rho": base["spearman_rho"],
        "median_absolute_error": base["median_absolute_error"],
        "sample_weighted_mae_from_per_family": sample_weighted_mae(per_fam),
        "macro_family_mae": macro_family_mae(per_fam),
        "median_family_mae": median_family_mae(per_fam),
        "n_families_covered": len(per_fam),
        "per_family_mae": per_fam,
    }


def _nested_fold_predict(
    model_name: str,
    train_composition: pd.Series,
    train_y: np.ndarray,
    test_composition: pd.Series,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """One outer fold's nested cycle: inner-split the fold's OWN outer-train
    partition, tune hyperparameters against that inner split only, then
    refit on the full outer-train with those hyperparameters and predict
    the outer-test partition. The outer-test rows never appear in the
    inner split passed to `tune_hyperparameters`."""
    train_composition = train_composition.reset_index(drop=True)
    n_train = len(train_composition)
    inner_train_idx, inner_val_idx = inner_early_stop_split(
        n_train, stop_fraction=INNER_TUNING_STOP_FRACTION, seed=seed
    )
    inner_train_comp = train_composition.iloc[inner_train_idx].reset_index(drop=True)
    inner_val_comp = train_composition.iloc[inner_val_idx].reset_index(drop=True)
    y_inner_train = np.asarray(train_y)[inner_train_idx]
    y_inner_val = np.asarray(train_y)[inner_val_idx]

    hyperparameters = tune_hyperparameters(
        model_name,
        inner_train_comp,
        y_inner_train,
        inner_val_comp,
        y_inner_val,
        seed=seed,
    )
    preds = fit_predict_model(
        model_name,
        hyperparameters,
        train_composition,
        train_y,
        test_composition,
        seed=seed,
    )
    return preds, hyperparameters


def run_regime_a(pool: pd.DataFrame, model_name: str) -> dict[str, Any]:
    n = len(pool)
    folds = random_kfold_splits(n, n_splits=N_OUTER_SPLITS, seed=CV_SEED)
    oof_pred = np.full(n, np.nan)
    fold_hyperparameters: list[dict[str, Any]] = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        preds, hp = _nested_fold_predict(
            model_name,
            pool["true_composition"].iloc[train_idx],
            pool["y"].to_numpy()[train_idx],
            pool["true_composition"].iloc[test_idx],
            seed=CV_SEED,
        )
        oof_pred[test_idx] = preds
        fold_hyperparameters.append(hp)
        print(f"    [A fold {fold_i}] done")
    if np.isnan(oof_pred).any():
        raise RuntimeError("run_regime_a: some pool row never appeared in a test fold")
    metrics = _oof_metrics(pool["y"].to_numpy(), oof_pred, pool["family_filled"].to_numpy())
    metrics["regime"] = "A_random_5fold_cv_nested"
    metrics["n_folds"] = len(folds)
    metrics["seed"] = CV_SEED
    metrics["per_fold_hyperparameters"] = fold_hyperparameters
    metrics["note"] = (
        "Folds are not family-stratified; per-family/macro-family MAE here is "
        "purely descriptive, not the regime's grouping axis. Hyperparameters were "
        "tuned per fold using an inner split of that fold's own outer-train "
        "partition only (nested; see module docstring)."
    )
    return metrics


def run_regime_b(pool: pd.DataFrame, model_name: str) -> dict[str, Any]:
    folds = family_group_kfold_splits(pool["family_filled"], n_splits=N_OUTER_SPLITS, seed=CV_SEED)
    n = len(pool)
    oof_pred = np.full(n, np.nan)
    fold_hyperparameters: list[dict[str, Any]] = []
    for fold_i, (train_idx, test_idx) in enumerate(folds):
        preds, hp = _nested_fold_predict(
            model_name,
            pool["true_composition"].iloc[train_idx],
            pool["y"].to_numpy()[train_idx],
            pool["true_composition"].iloc[test_idx],
            seed=CV_SEED,
        )
        oof_pred[test_idx] = preds
        fold_hyperparameters.append(hp)
        print(f"    [B fold {fold_i}] done")
    if np.isnan(oof_pred).any():
        raise RuntimeError("run_regime_b: some pool row never appeared in a test fold")
    metrics = _oof_metrics(pool["y"].to_numpy(), oof_pred, pool["family_filled"].to_numpy())
    metrics["regime"] = "B_family_group_kfold_nested"
    metrics["n_folds"] = len(folds)
    metrics["seed"] = CV_SEED
    metrics["per_fold_hyperparameters"] = fold_hyperparameters
    return metrics


def run_regime_c(pool: pd.DataFrame, model_name: str, major_families: list[str]) -> dict[str, Any]:
    splits = leave_one_major_family_out_splits(pool["family_filled"], major_families)
    y_all = pool["y"].to_numpy()
    held_indices: list[int] = []
    held_preds: list[float] = []
    per_family: dict[str, dict[str, float]] = {}
    fold_hyperparameters: dict[str, dict[str, Any]] = {}
    for fam_name, (train_idx, test_idx) in splits.items():
        preds, hp = _nested_fold_predict(
            model_name,
            pool["true_composition"].iloc[train_idx],
            y_all[train_idx],
            pool["true_composition"].iloc[test_idx],
            seed=CV_SEED,
        )
        held_indices.extend(test_idx.tolist())
        held_preds.extend(preds.tolist())
        mae = float(np.mean(np.abs(preds - y_all[test_idx])))
        per_family[fam_name] = {"mae": mae, "n": int(len(test_idx))}
        fold_hyperparameters[fam_name] = hp
        print(f"    [C {fam_name}] done, MAE={mae:.4f}")

    held_idx_arr = np.array(held_indices)
    held_pred_arr = np.array(held_preds)
    held_true_arr = y_all[held_idx_arr]
    base = regression_metrics(held_true_arr, held_pred_arr)
    return {
        "regime": "C_lomfo_nested",
        "seed": CV_SEED,
        "major_families": major_families,
        "n_major_families": len(major_families),
        "n_held_out_rows_total": int(len(held_idx_arr)),
        "pooled_mae_over_major_family_rows": base["mae"],
        "rmse_over_major_family_rows": base["rmse"],
        "spearman_rho_over_major_family_rows": base["spearman_rho"],
        "sample_weighted_mae_from_per_family": sample_weighted_mae(per_family),
        "macro_family_mae": macro_family_mae(per_family),
        "median_family_mae": median_family_mae(per_family),
        "per_family_mae": per_family,
        "per_fold_hyperparameters": fold_hyperparameters,
        "note": (
            "Coverage is intentionally partial: only rows belonging to a major "
            "family (n>=15) are ever held out and scored here. Hyperparameters "
            "were tuned per family-fold using an inner split of that fold's own "
            "outer-train partition (which already excludes the held-out family "
            "entirely), so the held-out family cannot leak into inner tuning."
        ),
    }


def _pool_meta(pool: pd.DataFrame) -> dict[str, Any]:
    family_counts_all = pool["family_filled"].value_counts()
    major_counts = major_family_counts(pool["family_filled"], min_n=LOMFO_MIN_FAMILY_N)
    return {
        "family_counts_all": {str(k): int(v) for k, v in family_counts_all.items()},
        "major_families": major_counts,
    }


def run_one_model(model_name: str) -> None:
    t0 = time.time()
    pool, n_test_untouched = load_dev_population()
    meta = _pool_meta(pool)
    major_families = sorted(meta["major_families"])

    print(f"=== NESTED {model_name} ===")
    a = run_regime_a(pool, model_name)
    print(f"  A nested random-5fold pooled MAE={a['pooled_mae']:.4f}")
    b = run_regime_b(pool, model_name)
    print(f"  B nested family-groupkfold pooled MAE={b['pooled_mae']:.4f}")
    c = run_regime_c(pool, model_name, major_families)
    print(f"  C nested lomfo pooled MAE={c['pooled_mae_over_major_family_rows']:.4f}")
    runtime = time.time() - t0
    print(f"  ({runtime:.1f}s)")

    fragment = {
        "model_name": model_name,
        "obelix_dev_population_n": len(pool),
        "obelix_test_untouched_row_count": n_test_untouched,
        "pool_meta": meta,
        "A_random_5fold_cv": a,
        "B_family_group_kfold": b,
        "C_lomfo": c,
        "runtime_seconds": runtime,
    }
    FRAGMENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    frag_path = FRAGMENT_OUT_DIR / f"{model_name}.json"
    guard_output(frag_path).write_text(
        json.dumps(fragment, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {frag_path.relative_to(ROOT)}")


def merge_fragments() -> None:
    fragments: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        frag_path = FRAGMENTS_DIR / f"{model_name}.json"
        if not frag_path.exists():
            raise FileNotFoundError(
                f"Missing fragment {frag_path}; run `python {Path(__file__).name} "
                f"--model {model_name}` first."
            )
        fragments[model_name] = json.loads(frag_path.read_text(encoding="utf-8"))

    first = fragments[MODEL_NAMES[0]]
    for model_name, frag in fragments.items():
        if frag["obelix_dev_population_n"] != first["obelix_dev_population_n"]:
            raise RuntimeError(
                f"Fragment mismatch: {model_name} used a different dev-population size."
            )
        if frag["pool_meta"] != first["pool_meta"]:
            raise RuntimeError(f"Fragment mismatch: {model_name} used different family metadata.")

    results = {
        name: {
            "A_random_5fold_cv": frag["A_random_5fold_cv"],
            "B_family_group_kfold": frag["B_family_group_kfold"],
            "C_lomfo": frag["C_lomfo"],
        }
        for name, frag in fragments.items()
    }
    total_runtime = sum(frag["runtime_seconds"] for frag in fragments.values())

    old = (
        json.loads(OLD_N4_ARTIFACT.read_text(encoding="utf-8"))
        if OLD_N4_ARTIFACT.exists()
        else None
    )

    manifest = {
        "phase": "N4b_nested_internal_validation_regimes",
        "supersedes": "N4 (reports/cross_dataset_study/n4_internal_validation.md) -- post-lock "
        "remediation",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "obelix_dev_population_n": first["obelix_dev_population_n"],
        "obelix_test_untouched_row_count": first["obelix_test_untouched_row_count"],
        "hargreaves_data_read": False,
        "n6_external_rerun": False,
        "cv_seed": CV_SEED,
        "n_outer_splits": N_OUTER_SPLITS,
        "lomfo_min_family_n": LOMFO_MIN_FAMILY_N,
        "inner_tuning_stop_fraction": INNER_TUNING_STOP_FRACTION,
        "family_counts_all": first["pool_meta"]["family_counts_all"],
        "major_families": first["pool_meta"]["major_families"],
        "results": results,
        "total_runtime_seconds": total_runtime,
    }
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    guard_output(ARTIFACT_JSON).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {ARTIFACT_JSON.relative_to(ROOT)}")

    _write_report(manifest, old)
    print(f"total runtime (summed across per-model processes): {total_runtime:.1f}s")


def _write_report(manifest: dict[str, Any], old: dict[str, Any] | None) -> None:
    lines: list[str] = []
    lines.append("# N4b — Nested Internal Validation Regimes (post-lock remediation)\n")
    lines.append(
        "Status: **DONE (post-lock methodological audit/remediation of N4's internal-CV "
        "hyperparameter leakage; N6 external inference was NOT rerun)**\n"
    )
    lines.append(f"Git commit: `{manifest['git_commit']}` on branch `{manifest['git_branch']}`\n")
    lines.append("\n## 1. What was wrong with N4\n\n")
    lines.append(
        "N3 tuned each model's hyperparameters once against OBELiX train(359)/validation(90), "
        "then froze them. N4 reused those frozen hyperparameters unchanged across every outer-CV "
        "fold built from train UNION validation (449 rows). Because the 90 validation rows "
        "shaped which hyperparameters were frozen, and most of those rows later served as "
        "OUTER-TEST rows in N4's regimes, N4's internal-CV metrics carry an optimism bias: "
        "outer-test labels had already partially influenced model selection before being "
        "scored on it. This script recomputes all three regimes with **nested** model "
        "selection: for every outer fold, hyperparameters are tuned using an inner split of "
        "that fold's own outer-train partition ONLY, using N3's exact search spaces, "
        f"builders, trial budgets, and tuning-seed convention (seed=`{manifest['cv_seed']}` "
        f"throughout). Inner tuning split fraction: {manifest['inner_tuning_stop_fraction']}.\n"
    )
    lines.append("\n## 2. Leakage boundary (unchanged from N4)\n\n")
    lines.append(
        f"- OBELiX frozen `test` split ({manifest['obelix_test_untouched_row_count']} rows) read "
        "only for its length.\n"
        "- No Hargreaves file opened in this script.\n"
        "- N6's one-shot external transfer/results were **not** rerun; this script only "
        "recomputes the internal (OBELiX-only) side of the comparison.\n"
    )
    lines.append("\n## 3. Old (N4, non-nested) vs. new (N4b, nested) pooled MAE\n\n")
    lines.append("| Model | Regime | N4 MAE (leaky) | N4b MAE (nested) | Δ (nested − leaky) |\n")
    lines.append("|---|---|---|---|---|\n")
    regime_keys = {
        "A_random_5fold_cv": ("A random-5fold", "pooled_mae", "pooled_mae"),
        "B_family_group_kfold": ("B family-groupkfold", "pooled_mae", "pooled_mae"),
        "C_lomfo": (
            "C lomfo",
            "pooled_mae_over_major_family_rows",
            "pooled_mae_over_major_family_rows",
        ),
    }
    deltas: dict[str, dict[str, float]] = {}
    for model_name in MODEL_NAMES:
        new_r = manifest["results"][model_name]
        old_r = old["results"][model_name] if old is not None else None
        deltas[model_name] = {}
        for regime_key, (label, old_key, new_key) in regime_keys.items():
            new_mae = new_r[regime_key][new_key]
            if old_r is not None:
                old_mae = old_r[regime_key][old_key]
                delta = new_mae - old_mae
                deltas[model_name][regime_key] = delta
                lines.append(
                    f"| {model_name} | {label} | {old_mae:.4f} | {new_mae:.4f} | {delta:+.4f} |\n"
                )
            else:
                lines.append(f"| {model_name} | {label} | N/A | {new_mae:.4f} | N/A |\n")
    lines.append(
        "\nFull per-fold hyperparameters and per-family breakdowns: "
        "`artifacts/historical/n4b_nested_internal_validation.json`.\n"
    )
    lines.append("\n## 4. Whether H1/H2 conclusions change\n\n")
    lines.append(
        "See `reports/cross_dataset_study/n7_regime_utility.md` for the original H1/H2 statistics "
        "computed "
        "from N4's (leaky) numbers, and the accompanying manuscript/report addendum for the "
        "recomputed H1/H2 statistics using N4b's nested numbers against the same "
        "already-frozen N6 external results. This report only supplies the corrected internal "
        "numbers; the H1/H2 regime-comparison recompute itself lives in the N7 remediation "
        "addendum so the two computations are not duplicated.\n"
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_MD).write_text("".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_MD.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_fragments()
        return
    if args.model is not None:
        run_one_model(args.model)
        return
    parser.error("pass either --model <name> or --merge")


if __name__ == "__main__":
    main()
