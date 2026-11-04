#!/usr/bin/env python3
"""Reviewer remediation Issue 08b: chemical-system-grouped sensitivity
control (regime E).

Spec: reports/remediation/specs/issue08b_chemical_system_groupkfold.md (hashed
before this script was written).

Exact structural copy of scripts/transport_analysis/n20b_formula_group_kfold.py (itself a
copy of N4b's regime B) with the grouping vector changed from canonical
reduced formula to chemical system (pymatgen element set, via the existing
`chemical_system()` helper in
scripts/transport_analysis/phase1_0_1_predeclare_and_glass_split.py -- not a new
definition). Same nested per-fold Optuna tuning protocol, same CV_SEED,
same 8-model ladder, same one-model-per-process convention as N4b/N20b.

Run per-model (own OS process):
    python scripts/analysis/issue08b_chemical_system_groupkfold.py --model <name>
Then merge:
    python scripts/analysis/issue08b_chemical_system_groupkfold.py --merge
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
sys.path.insert(0, str(ROOT / "scripts" / "transport_analysis"))

import phase1_0_1_predeclare_and_glass_split as phase1  # noqa: E402

from sse_transfer.data.obelix_loader import load_obelix  # noqa: E402
from sse_transfer.data.regression_dataset import load_regression_dataset  # noqa: E402
from sse_transfer.evaluation.family_robustness import (  # noqa: E402
    macro_family_mae,
    median_family_mae,
    per_family_mae,
    sample_weighted_mae,
)
from sse_transfer.evaluation.internal_validation import (  # noqa: E402
    N_OUTER_SPLITS,
    canonicalize_and_fill_family,
    family_group_kfold_splits,
    inner_early_stop_split,
)
from sse_transfer.evaluation.metrics import regression_metrics  # noqa: E402
from sse_transfer.models.ladder import MODEL_NAMES, fit_predict_model  # noqa: E402
from sse_transfer.training.tune_fold import tune_hyperparameters  # noqa: E402
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

ARTIFACT_JSON = ROOT / "artifacts" / "rerun" / "issue08b_chemical_system_groupkfold_corrected.json"
REPORT_MD = ROOT / "reports" / "rerun" / "issue08b_chemical_system_groupkfold.md"
FRAGMENTS_DIR = ROOT / "artifacts" / "fragments" / "issue08b"
# Recorded fragments (read by --merge) are immutable; --model reruns write here.
FRAGMENT_OUT_DIR = ROOT / "artifacts" / "rerun" / "fragments" / "issue08b"

CV_SEED = 20260921  # identical to N3/N4/N4b/N20b throughout
INNER_TUNING_STOP_FRACTION = 0.2  # identical to N4b


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def load_dev_population() -> tuple[pd.DataFrame, int]:
    """Identical to N4b/N20b's loader, plus a `chemical_system` column."""
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
    pool["chemical_system"] = pool["true_composition"].apply(phase1.chemical_system).to_numpy()
    if pool["chemical_system"].isna().any():
        raise RuntimeError("load_dev_population: unparseable composition in dev pool")
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
    """Identical nested-tuning cycle to N4b/N20b's `_nested_fold_predict`."""
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
        model_name, inner_train_comp, y_inner_train, inner_val_comp, y_inner_val, seed=seed
    )
    preds = fit_predict_model(
        model_name, hyperparameters, train_composition, train_y, test_composition, seed=seed
    )
    return preds, hyperparameters


def run_regime_chemical_system_group_kfold(pool: pd.DataFrame, model_name: str) -> dict[str, Any]:
    """Chemical-system GroupKFold: every row sharing a pymatgen element set
    falls entirely in one outer fold (reuses `family_group_kfold_splits`
    with `chemical_system` as the grouping vector)."""
    folds = family_group_kfold_splits(
        pool["chemical_system"], n_splits=N_OUTER_SPLITS, seed=CV_SEED
    )
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
        print(f"    [chemical-system-groupkfold fold {fold_i}] done, test size={len(test_idx)}")
    if np.isnan(oof_pred).any():
        raise RuntimeError(
            "run_regime_chemical_system_group_kfold: some pool row never appeared in a test fold"
        )
    metrics = _oof_metrics(pool["y"].to_numpy(), oof_pred, pool["family_filled"].to_numpy())
    metrics["regime"] = "E_chemical_system_group_kfold"
    metrics["n_folds"] = len(folds)
    metrics["seed"] = CV_SEED
    metrics["per_fold_hyperparameters"] = fold_hyperparameters
    metrics["note"] = (
        "ACS/JCIM remediation Issue 08b: grouping axis is the pymatgen element set "
        "(chemical system), not exact reduced formula (regime D) or chemistry family "
        "(regime B) -- tests whether redundancy broader than exact-formula duplication "
        "(e.g. dopant/solid-solution series sharing an element set) drives any part of "
        "the primary result."
    )
    return metrics


def run_one_model(model_name: str) -> None:
    t0 = time.time()
    pool, n_test_untouched = load_dev_population()
    n_unique_systems = int(pool["chemical_system"].nunique())

    print(f"=== ACS08b chemical-system-groupkfold {model_name} ===")
    result = run_regime_chemical_system_group_kfold(pool, model_name)
    print(f"  pooled MAE={result['pooled_mae']:.4f}")
    runtime = time.time() - t0
    print(f"  ({runtime:.1f}s)")

    fragment = {
        "model_name": model_name,
        "obelix_dev_population_n": len(pool),
        "obelix_test_untouched_row_count": n_test_untouched,
        "n_unique_chemical_systems": n_unique_systems,
        "chemical_system_group_kfold": result,
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
        if frag["n_unique_chemical_systems"] != first["n_unique_chemical_systems"]:
            raise RuntimeError(
                f"Fragment mismatch: {model_name} used a different chemical-system set."
            )

    results = {name: frag["chemical_system_group_kfold"] for name, frag in fragments.items()}
    total_runtime = sum(frag["runtime_seconds"] for frag in fragments.values())

    manifest = {
        "phase": "acs_issue08b_chemical_system_groupkfold",
        "spec_path": "reports/remediation/specs/issue08b_chemical_system_groupkfold.md",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "obelix_dev_population_n": first["obelix_dev_population_n"],
        "obelix_test_untouched_row_count": first["obelix_test_untouched_row_count"],
        "n_unique_chemical_systems": first["n_unique_chemical_systems"],
        "hargreaves_data_read": False,
        "n6_external_rerun": False,
        "cv_seed": CV_SEED,
        "n_outer_splits": N_OUTER_SPLITS,
        "inner_tuning_stop_fraction": INNER_TUNING_STOP_FRACTION,
        "results": results,
        "total_runtime_seconds": total_runtime,
    }
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    guard_output(ARTIFACT_JSON).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {ARTIFACT_JSON.relative_to(ROOT)}")
    _write_report(manifest)


def _write_report(manifest: dict[str, Any]) -> None:
    n4b_path = (
        ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
    )
    n4b = json.loads(n4b_path.read_text(encoding="utf-8")) if n4b_path.exists() else None

    lines: list[str] = []
    lines.append("# Issue 08b — Chemical-System GroupKFold Sensitivity (regime E)\n")
    lines.append(
        "\n**Spec:** `reports/remediation/specs/issue08b_chemical_system_groupkfold.md`.\n"
    )
    lines.append(f"\nGit commit: `{manifest['git_commit']}` on branch `{manifest['git_branch']}`\n")
    lines.append(
        f"\n{manifest['n_unique_chemical_systems']} unique chemical systems over "
        f"{manifest['obelix_dev_population_n']} rows.\n"
    )
    lines.append(
        "\n## Pooled MAE: random-5fold (regime A) vs chemical-system-GroupKFold (regime E)\n\n"
    )
    lines.append("| Model | A: random-5fold MAE | E: chem-system-GroupKFold MAE | Delta |\n")
    lines.append("|---|---|---|---|\n")
    for model_name in MODEL_NAMES:
        new_mae = manifest["results"][model_name]["pooled_mae"]
        if n4b is not None:
            old_mae = n4b["results"][model_name]["A_random_5fold_cv"]["pooled_mae"]
            delta = new_mae - old_mae
            lines.append(f"| {model_name} | {old_mae:.4f} | {new_mae:.4f} | {delta:+.4f} |\n")
        else:
            lines.append(f"| {model_name} | N/A | {new_mae:.4f} | N/A |\n")
    lines.append(
        "\nFull per-fold hyperparameters and per-family breakdowns: "
        "`artifacts/historical/issue08b_chemical_system_groupkfold.json`.\n"
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
