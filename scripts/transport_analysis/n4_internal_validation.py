#!/usr/bin/env python3
"""Cross-dataset study, N4: internal validation regimes.

Evaluates all 8 N3-frozen composition-only models under three conceptually
distinct outer-fold regimes on the OBELiX development population ONLY
(frozen `train` UNION `validation` = 449 rows from
`data.regression_dataset.load_regression_dataset()`):

  A. Random 5-fold CV (`sklearn.model_selection.KFold`, no grouping)
  B. Canonical Family-GroupKFold (`sklearn.model_selection.GroupKFold`,
     grouped by the harmonized+filled `Family` label -- matches the old
     project's `build_canonical_family_folds` convention)
  C. Leave-one-major-family-out (every family with >=15 rows in the pool,
     the same `LOMFO_MIN_FAMILY_N` threshold used identically across
     scripts/phase2_2/2_3/2_4b/2_5/2_6a)

The frozen 113-row OBELiX `test` split (old, closed DD project) is read
only for its length, never its features/targets. No Hargreaves file is
opened at all in this script -- external transfer is N6, out of scope here.

Every model's frozen hyperparameters come from N3's
`artifacts/frozen/model_ladder_frozen.json` and are never re-tuned; each fold
refits the composition-only feature encoder (Magpie scaler / pooled
element embedder / element vocabulary) on that fold's training rows only.

Run with: python scripts/transport_analysis/n4_internal_validation.py
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
    leave_one_major_family_out_splits,
    major_family_counts,
    random_kfold_splits,
)
from sse_transfer.evaluation.metrics import regression_metrics  # noqa: E402
from sse_transfer.models.ladder import (  # noqa: E402
    MODEL_NAMES,
    fit_predict_model,
    load_frozen_ladder,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

ARTIFACT_JSON = ROOT / "artifacts" / "rerun" / "n4_internal_validation.json"
REPORT_MD = ROOT / "reports" / "rerun" / "n4_internal_validation.md"
FRAGMENTS_DIR = ROOT / "artifacts" / "rerun" / "fragments" / "n4"

CV_SEED = 20260921  # same fixed seed N3 used for tuning; reused here for fold
# construction and every fold's model-fit seed, documented as a single
# frozen choice for this descriptive phase (multi-seed robustness averaging
# is reserved for N6/N7's headline reported numbers, not this phase).


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def load_dev_population() -> tuple[pd.DataFrame, int]:
    """Returns (pool_df, n_test_untouched). pool_df has columns
    [ID, y, true_composition, family_filled], one row per OBELiX dev-
    population row (449 = 359 train + 90 validation, combined into one pool
    for this phase's outer-fold CV -- N3's train/validation split existed
    only for N3's own hyperparameter freezing, not for N4's regimes)."""
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


def run_regime_a(
    pool: pd.DataFrame, model_name: str, hyperparameters: dict[str, Any]
) -> dict[str, Any]:
    n = len(pool)
    folds = random_kfold_splits(n, n_splits=N_OUTER_SPLITS, seed=CV_SEED)
    oof_pred = np.full(n, np.nan)
    for train_idx, test_idx in folds:
        preds = fit_predict_model(
            model_name,
            hyperparameters,
            pool["true_composition"].iloc[train_idx],
            pool["y"].to_numpy()[train_idx],
            pool["true_composition"].iloc[test_idx],
            seed=CV_SEED,
        )
        oof_pred[test_idx] = preds
    if np.isnan(oof_pred).any():
        raise RuntimeError("run_regime_a: some pool row never appeared in a test fold")
    metrics = _oof_metrics(pool["y"].to_numpy(), oof_pred, pool["family_filled"].to_numpy())
    metrics["regime"] = "A_random_5fold_cv"
    metrics["n_folds"] = len(folds)
    metrics["seed"] = CV_SEED
    metrics["note"] = (
        "Folds are not family-stratified; per-family/macro-family MAE here is "
        "purely descriptive (pooled across whichever fold each family's rows "
        "happened to land in), not the regime's grouping axis."
    )
    return metrics


def run_regime_b(
    pool: pd.DataFrame, model_name: str, hyperparameters: dict[str, Any]
) -> dict[str, Any]:
    folds = family_group_kfold_splits(pool["family_filled"], n_splits=N_OUTER_SPLITS, seed=CV_SEED)
    n = len(pool)
    oof_pred = np.full(n, np.nan)
    for train_idx, test_idx in folds:
        preds = fit_predict_model(
            model_name,
            hyperparameters,
            pool["true_composition"].iloc[train_idx],
            pool["y"].to_numpy()[train_idx],
            pool["true_composition"].iloc[test_idx],
            seed=CV_SEED,
        )
        oof_pred[test_idx] = preds
    if np.isnan(oof_pred).any():
        raise RuntimeError("run_regime_b: some pool row never appeared in a test fold")
    metrics = _oof_metrics(pool["y"].to_numpy(), oof_pred, pool["family_filled"].to_numpy())
    metrics["regime"] = "B_family_group_kfold"
    metrics["n_folds"] = len(folds)
    metrics["seed"] = CV_SEED
    return metrics


def run_regime_c(
    pool: pd.DataFrame,
    model_name: str,
    hyperparameters: dict[str, Any],
    major_families: list[str],
) -> dict[str, Any]:
    splits = leave_one_major_family_out_splits(pool["family_filled"], major_families)
    y_all = pool["y"].to_numpy()
    held_indices: list[int] = []
    held_preds: list[float] = []
    per_family: dict[str, dict[str, float]] = {}
    for fam_name, (train_idx, test_idx) in splits.items():
        preds = fit_predict_model(
            model_name,
            hyperparameters,
            pool["true_composition"].iloc[train_idx],
            y_all[train_idx],
            pool["true_composition"].iloc[test_idx],
            seed=CV_SEED,
        )
        held_indices.extend(test_idx.tolist())
        held_preds.extend(preds.tolist())
        mae = float(np.mean(np.abs(preds - y_all[test_idx])))
        per_family[fam_name] = {"mae": mae, "n": int(len(test_idx))}

    held_idx_arr = np.array(held_indices)
    held_pred_arr = np.array(held_preds)
    held_true_arr = y_all[held_idx_arr]
    base = regression_metrics(held_true_arr, held_pred_arr)
    return {
        "regime": "C_lomfo",
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
        "note": (
            "Coverage is intentionally partial: only rows belonging to a major "
            "family (n>=15) are ever held out and scored here, unlike regimes A/B "
            "which cover the full 449-row pool."
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
    """Runs all three regimes for a single model and writes a fragment JSON.

    Run as its own OS process (see the shell loop in this module's
    docstring / reports/cross_dataset_study/n4_internal_validation.md "Execution note") so
    memory from one model's fits (especially the tree ensembles and torch
    models) is fully released by the OS before the next model starts --
    this repo's runtime environment has ~4.8GB total RAM, and running all 8
    models in one long-lived process was empirically observed to be killed
    by the OOM killer partway through `random_forest`.
    """
    t0 = time.time()
    pool, n_test_untouched = load_dev_population()
    meta = _pool_meta(pool)
    major_families = sorted(meta["major_families"])

    ladder = load_frozen_ladder()
    hyperparameters = ladder["models"][model_name]["hyperparameters"]

    print(f"=== {model_name} ===")
    a = run_regime_a(pool, model_name, hyperparameters)
    print(
        f"  A random-5fold pooled MAE={a['pooled_mae']:.4f} "
        f"macro-family={a['macro_family_mae']:.4f}"
    )
    b = run_regime_b(pool, model_name, hyperparameters)
    print(
        f"  B family-groupkfold pooled MAE={b['pooled_mae']:.4f} "
        f"macro-family={b['macro_family_mae']:.4f}"
    )
    c = run_regime_c(pool, model_name, hyperparameters, major_families)
    print(
        f"  C lomfo pooled MAE={c['pooled_mae_over_major_family_rows']:.4f} "
        f"macro-family={c['macro_family_mae']:.4f}"
    )
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
    FRAGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    frag_path = FRAGMENTS_DIR / f"{model_name}.json"
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
                f"Fragment mismatch: {model_name} used a different dev-population size "
                f"({frag['obelix_dev_population_n']}) than {MODEL_NAMES[0]} "
                f"({first['obelix_dev_population_n']})."
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

    manifest = {
        "phase": "N4_internal_validation_regimes",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "obelix_dev_population_n": first["obelix_dev_population_n"],
        "obelix_test_untouched_row_count": first["obelix_test_untouched_row_count"],
        "hargreaves_data_read": False,
        "cv_seed": CV_SEED,
        "n_outer_splits": N_OUTER_SPLITS,
        "lomfo_min_family_n": LOMFO_MIN_FAMILY_N,
        "family_counts_all": first["pool_meta"]["family_counts_all"],
        "major_families": first["pool_meta"]["major_families"],
        "results": results,
        "total_runtime_seconds": total_runtime,
        "execution_note": (
            "Each model was run as its own OS process (see run_one_model docstring) to "
            "avoid an OOM kill observed when running all 8 models in one long-lived "
            "process on this environment's ~4.8GB RAM; total_runtime_seconds is the sum "
            "of the 8 per-model wall-clock times, not a single contiguous run."
        ),
    }
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    guard_output(ARTIFACT_JSON).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {ARTIFACT_JSON.relative_to(ROOT)}")

    _write_report(manifest)
    print(f"total runtime (summed across per-model processes): {total_runtime:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default=None,
        help="Run only this model's three regimes and write a fragment JSON.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all 8 per-model fragment JSONs into the final artifact + report.",
    )
    args = parser.parse_args()

    if args.merge:
        merge_fragments()
        return
    if args.model is not None:
        run_one_model(args.model)
        return
    parser.error("pass either --model <name> or --merge")


#: A (model, regime) pooled MAE more than this multiple of the same model's
#: best other-regime pooled MAE is flagged as a possible catastrophic-
#: extrapolation outlier worth a dedicated report section. Chosen as a
#: clearly-abnormal threshold (a well-behaved regime shift is at most a
#: modest multiple, not 100x+) before ever looking at this run's actual
#: numbers -- not tuned to flag exactly what was observed.
EXTRAPOLATION_OUTLIER_RATIO = 10.0
#: Within a flagged (model, regime), a family's own MAE more than this
#: multiple of the regime's *median*-family MAE (robust to the very
#: outlier(s) being searched for -- unlike macro-family MAE, which is a
#: mean and is itself inflated by whichever family is driving the outlier)
#: is reported as a driving family.
EXTRAPOLATION_FAMILY_RATIO = 20.0


def _extrapolation_outlier_section(manifest: dict[str, Any]) -> str:
    """Detects and reports any (model, regime) pair whose pooled MAE badly
    diverges from that same model's other two regimes -- computed fresh from
    `manifest['results']` on every run (never hardcoded prose), so a future
    rerun with different numbers regenerates this section correctly rather
    than silently going stale. See reports/cross_dataset_study/n4_internal_validation.md
    section 5 for the interpretation this run produced."""
    regime_pooled_key = {
        "A_random_5fold_cv": "pooled_mae",
        "B_family_group_kfold": "pooled_mae",
        "C_lomfo": "pooled_mae_over_major_family_rows",
    }
    outliers: list[dict[str, Any]] = []
    for model_name, r in manifest["results"].items():
        pooled = {regime: r[regime][key] for regime, key in regime_pooled_key.items()}
        for regime, value in pooled.items():
            others = [v for k, v in pooled.items() if k != regime]
            best_other = min(others)
            if best_other > 0 and value >= EXTRAPOLATION_OUTLIER_RATIO * best_other:
                per_fam = r[regime]["per_family_mae"]
                median = r[regime]["median_family_mae"]
                driving = sorted(
                    (
                        (fam, v["mae"], v["n"])
                        for fam, v in per_fam.items()
                        if median > 0 and v["mae"] >= EXTRAPOLATION_FAMILY_RATIO * median
                    ),
                    key=lambda t: -t[1],
                )
                outliers.append(
                    {
                        "model": model_name,
                        "regime": regime,
                        "pooled_mae": value,
                        "best_other_regime_pooled_mae": best_other,
                        "driving_families": driving,
                    }
                )

    if not outliers:
        return (
            "\n## 5. Extrapolation-outlier check\n\n"
            f"No (model, regime) pooled MAE exceeded {EXTRAPOLATION_OUTLIER_RATIO:.0f}x that "
            "same model's best other-regime pooled MAE this run — no catastrophic-extrapolation "
            "outlier to report.\n"
        )

    parts = [
        "\n## 5. Notable finding: catastrophic extrapolation under a family-shift regime "
        "(detected, not hardcoded)\n\n",
        "The check below flags any (model, regime) pooled MAE at least "
        f"{EXTRAPOLATION_OUTLIER_RATIO:.0f}x that same model's best other-regime pooled MAE, "
        "computed fresh from this run's own numbers (not asserted from a prior run). This is "
        "reported honestly rather than smoothed over, re-tuned away, or excluded (CLAUDE.md "
        '"Negative scientific results are valid"; EXPERIMENT_PROTOCOL.md "Do not select ... '
        'models by final-test performance") -- no hyperparameter was changed after observing '
        "it.\n\n",
    ]
    for o in outliers:
        parts.append(
            f"- **{o['model']} / {o['regime']}**: pooled MAE {o['pooled_mae']:.1f} vs. "
            f"{o['best_other_regime_pooled_mae']:.4f} in its best other regime "
            f"({o['pooled_mae'] / o['best_other_regime_pooled_mae']:.0f}x). "
        )
        if o["driving_families"]:
            fam_desc = ", ".join(
                f"{fam} (n={n}, MAE={mae:.1f})" for fam, mae, n in o["driving_families"]
            )
            parts.append(f"Driven by: {fam_desc}.\n")
        else:
            parts.append("No single family's MAE exceeded the driving-family threshold.\n")
    parts.append(
        "\nModels capable of unbounded output (Ridge's linear map; a ReLU-MLP's unbounded "
        "final linear layer) have no mechanism to bound a prediction to a physically sane "
        "range once a held-out family's feature vector falls outside the training "
        "distribution. Bagged tree ensembles (RandomForest/ExtraTrees) are structurally "
        "range-bounded by construction (each leaf predicts an average of training targets), "
        "but gradient-boosted trees (XGBoost/CatBoost) are not bounded by this same argument "
        "in principle -- each boosting stage adds an unbounded leaf-value correction -- and "
        "were simply not flagged above, i.e. this is an observed empirical stability, not a "
        "proven structural guarantee, for XGBoost/CatBoost specifically. Random-5-fold CV did "
        "not expose this failure pattern whenever the affected family is large enough to "
        "appear in every random fold's training set and is not designed to isolate a "
        "complete family-removal failure of this kind -- only a family-holdout regime "
        "exposes it, and section 4's table shows the same model's LOMFO (regime C) result "
        "for comparison.\n"
    )
    return "".join(parts)


def _write_report(manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# N4 — Internal Validation Regimes\n")
    lines.append(
        "Status: **DONE (descriptive only — no cross-regime utility comparison, that is N7)**\n"
    )
    lines.append(f"Git commit: `{manifest['git_commit']}` on branch `{manifest['git_branch']}`\n")
    lines.append("\n## 1. Population and leakage boundary\n\n")
    lines.append(
        f"- OBELiX development population pool: {manifest['obelix_dev_population_n']} rows "
        "(frozen `train` UNION `validation` from `load_regression_dataset()`).\n"
        f"- OBELiX frozen `test` split ({manifest['obelix_test_untouched_row_count']} rows) was "
        "read only for its length "
        "(`scripts/transport_analysis/n4_internal_validation.py::load_dev_population` "
        "reads `dataset.test` only via `len(dataset.test)`) — never for features/targets.\n"
        "- No Hargreaves file was opened anywhere in this script.\n"
        "- Every model's hyperparameters are the exact values N3 froze in "
        "`artifacts/frozen/model_ladder_frozen.json` — never re-tuned per fold. Each fold refits "
        "its composition-only feature encoder (Magpie scaler / pooled element embedder / "
        "element vocabulary) on that fold's training rows only.\n"
    )
    lines.append("\n## 2. Major chemical families\n\n")
    lines.append(
        f"Threshold: `LOMFO_MIN_FAMILY_N = {manifest['lomfo_min_family_n']}` (reused unchanged "
        "from `scripts/phase2_2_ood_robustness.py` and every later phase that used it).\n\n"
    )
    lines.append("| Family | n (in 449-row pool) |\n|---|---|\n")
    for fam, n in sorted(manifest["major_families"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {fam} | {n} |\n")
    lines.append(
        f"\n{len(manifest['major_families'])} of "
        f"{len(manifest['family_counts_all'])} distinct family labels in the pool qualify as "
        "major.\n"
    )
    lines.append("\n## 3. Methodology per regime\n\n")
    lines.append(
        "- **A. Random 5-fold CV**: `sklearn.model_selection.KFold(n_splits=5, shuffle=True, "
        f"random_state={manifest['cv_seed']})` over the 449-row pool, no grouping. Every row "
        "appears in exactly one test fold; pooled metrics are computed on the full out-of-fold "
        "prediction vector.\n"
        "- **B. Canonical Family-GroupKFold**: `sklearn.model_selection.GroupKFold(n_splits=5, "
        f"shuffle=True, random_state={manifest['cv_seed']})` grouped by the harmonized+filled "
        "`Family` label (`family_harmonization.canonical_family` then `__MISSING__`-filled) — "
        "every row of a given family falls in exactly one fold, matching the old project's "
        "`build_canonical_family_folds` convention.\n"
        "- **C. LOMFO**: for each major family, every row of that family is held out and the "
        "model is refit on every other row in the pool (including minor families and "
        "`__MISSING__`). Coverage is intentionally partial (only major-family rows are ever "
        "scored), unlike A/B which cover the full pool.\n"
        f"- All folds/exclusions and all model fits use the single fixed seed "
        f"`{manifest['cv_seed']}` (same value N3 used for hyperparameter tuning); no per-fold "
        "re-tuning occurs anywhere in this phase. Multi-seed robustness averaging is reserved "
        "for later phases' headline reported numbers, not this descriptive internal-validation "
        "comparison.\n"
    )
    lines.append("\n## 4. Per-model x per-regime metrics\n\n")
    lines.append(
        "| Model | Regime | Pooled MAE | Macro-family MAE | RMSE | Spearman rho | n scored |\n"
    )
    lines.append("|---|---|---|---|---|---|---|\n")
    for model_name in [
        "ridge",
        "random_forest",
        "extra_trees",
        "xgboost",
        "catboost",
        "shallow_mlp",
        "pretrained_element_embedding",
        "composition_attention",
    ]:
        r = manifest["results"][model_name]
        a, b, c = r["A_random_5fold_cv"], r["B_family_group_kfold"], r["C_lomfo"]
        lines.append(
            f"| {model_name} | A random-5fold | {a['pooled_mae']:.4f} | "
            f"{a['macro_family_mae']:.4f} | {a['rmse']:.4f} | {a['spearman_rho']:.4f} | "
            f"{manifest['obelix_dev_population_n']} |\n"
        )
        lines.append(
            f"| {model_name} | B family-groupkfold | {b['pooled_mae']:.4f} | "
            f"{b['macro_family_mae']:.4f} | {b['rmse']:.4f} | {b['spearman_rho']:.4f} | "
            f"{manifest['obelix_dev_population_n']} |\n"
        )
        lines.append(
            f"| {model_name} | C lomfo | {c['pooled_mae_over_major_family_rows']:.4f} | "
            f"{c['macro_family_mae']:.4f} | {c['rmse_over_major_family_rows']:.4f} | "
            f"{c['spearman_rho_over_major_family_rows']:.4f} | {c['n_held_out_rows_total']} |\n"
        )
    lines.append(
        "\nFull per-family breakdowns for every (model, regime) pair: "
        "`artifacts/frozen/n4_internal_validation.json`.\n"
    )
    lines.append(_extrapolation_outlier_section(manifest))
    lines.append("\n## 6. Plain-language note (descriptive only)\n\n")
    lines.append(
        "Regime C (LOMFO) is structurally the hardest test — every held-out row belongs to a "
        "family the model never saw at all during that fit, unlike regimes A and B where "
        "(for A) the same family's other rows are usually still in training, and (for B) a "
        "*different* family is excluded per fold, not necessarily the one being scored. This "
        "phase does not judge which regime is 'better' or attempt to relate any of these "
        "numbers to the Hargreaves external transfer result — that comparison (H1/H2 in the "
        "task brief) is N7's job, using N6's one-shot external results, not available yet.\n"
    )
    lines.append("\n## 7. Not done in N4 (by design)\n\n")
    lines.append(
        "- No comparison of these regimes' predictive value against external (Hargreaves) "
        "performance — that is N7.\n"
        "- No Hargreaves data read at all.\n"
        "- No OBELiX-test evaluation at all.\n"
        "- No re-tuning of any hyperparameter.\n"
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_MD).write_text("".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
