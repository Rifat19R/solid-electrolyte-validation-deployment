#!/usr/bin/env python3
"""Reviewer remediation Issue 09, Variant B: treat each __MISSING__ row as
its own singleton group under Family-GroupKFold (regime B').

Spec: reports/remediation/specs/issue09_missing_family_sensitivity.md (hashed
before this script was written).

Exact structural copy of scripts/transport_analysis/n20b_formula_group_kfold.py /
scripts/analysis/issue08b_chemical_system_groupkfold.py with the grouping vector
changed: every row whose harmonized family is MISSING_FAMILY
("__MISSING__") gets a unique synthetic per-row group id instead of
sharing one "__MISSING__" label; every other row keeps its real family
label unchanged. This is a genuine rerun (not a recombination like Variant
A) because GroupKFold's fold assignment for every OTHER row can shift once
18 rows stop being one shared group. Regime A (random CV) is untouched by
family-grouping changes and is NOT rerun here -- reused unmodified from
N4b.

LOMFO (regime C) is only trivially affected: __MISSING__ automatically
drops out of `major_families` once it is 18 singleton groups (no group
has >=15 rows), which needs no rerun beyond removing it from the sum this
script's own `main()` already computes as part of Issue 09 Variant A's
common-support-style recombination -- not repeated here.

Run per-model (own OS process):
    python scripts/analysis/issue09b_missing_singleton_groupkfold.py --model <name>
Then merge:
    python scripts/analysis/issue09b_missing_singleton_groupkfold.py --merge
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
    MISSING_FAMILY,
    N_OUTER_SPLITS,
    canonicalize_and_fill_family,
    family_group_kfold_splits,
    inner_early_stop_split,
)
from sse_transfer.evaluation.metrics import regression_metrics  # noqa: E402
from sse_transfer.models.ladder import MODEL_NAMES, fit_predict_model  # noqa: E402
from sse_transfer.training.tune_fold import tune_hyperparameters  # noqa: E402
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

ARTIFACT_JSON = (
    ROOT / "artifacts" / "rerun" / "issue09b_missing_singleton_groupkfold_corrected.json"
)
REPORT_MD = ROOT / "reports" / "rerun" / "issue09b_missing_singleton_groupkfold.md"
FRAGMENTS_DIR = ROOT / "artifacts" / "fragments" / "issue09b"
# Recorded fragments (read by --merge) are immutable; --model reruns write here.
FRAGMENT_OUT_DIR = ROOT / "artifacts" / "rerun" / "fragments" / "issue09b"

CV_SEED = 20260921  # identical to N3/N4/N4b throughout
INNER_TUNING_STOP_FRACTION = 0.2  # identical to N4b


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def load_dev_population() -> tuple[pd.DataFrame, int]:
    """Identical to N4b's loader, plus a `family_missing_singleton` column:
    __MISSING__ rows get a unique synthetic group id ("__MISSING__::<row
    position>"); every other row keeps its real family_filled label."""
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
    pool = pool.reset_index(drop=True)
    is_missing = pool["family_filled"] == MISSING_FAMILY
    singleton_group = pool["family_filled"].astype(object).to_numpy()
    missing_positions = np.flatnonzero(is_missing.to_numpy())
    for pos in missing_positions:
        singleton_group[pos] = f"{MISSING_FAMILY}::{pos}"
    pool["family_missing_singleton"] = singleton_group
    n_missing = int(is_missing.sum())
    if n_missing != 18:
        raise RuntimeError(f"load_dev_population: expected 18 __MISSING__ rows, got {n_missing}")
    return pool, len(dataset.test)


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


def run_regime_missing_singleton_group_kfold(pool: pd.DataFrame, model_name: str) -> dict[str, Any]:
    folds = family_group_kfold_splits(
        pool["family_missing_singleton"], n_splits=N_OUTER_SPLITS, seed=CV_SEED
    )
    n = len(pool)
    oof_pred = np.full(n, np.nan)
    fold_hyperparameters: list[dict[str, Any]] = []
    fold_sizes: list[int] = []
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
        fold_sizes.append(len(test_idx))
        print(f"    [missing-singleton-groupkfold fold {fold_i}] done, test size={len(test_idx)}")
    if np.isnan(oof_pred).any():
        raise RuntimeError(
            "run_regime_missing_singleton_group_kfold: some pool row never appeared in a test fold"
        )
    metrics = _oof_metrics(pool["y"].to_numpy(), oof_pred, pool["family_filled"].to_numpy())
    metrics["regime"] = "B_prime_missing_singleton_group_kfold"
    metrics["n_folds"] = len(folds)
    metrics["fold_sizes"] = fold_sizes
    metrics["seed"] = CV_SEED
    metrics["per_fold_hyperparameters"] = fold_hyperparameters
    metrics["note"] = (
        "ACS/JCIM remediation Issue 09 Variant B: identical to regime B (Family-GroupKFold) "
        "except the 18 __MISSING__ rows are each their own singleton group instead of one "
        "shared 18-row group -- tests whether Family-GroupKFold's fold assignment for OTHER "
        "rows, and hence the primary result, is sensitive to how __MISSING__ is grouped."
    )
    return metrics


def run_one_model(model_name: str) -> None:
    t0 = time.time()
    pool, n_test_untouched = load_dev_population()

    print(f"=== ACS09b missing-singleton-groupkfold {model_name} ===")
    result = run_regime_missing_singleton_group_kfold(pool, model_name)
    print(f"  pooled MAE={result['pooled_mae']:.4f}")
    runtime = time.time() - t0
    print(f"  ({runtime:.1f}s)")

    fragment = {
        "model_name": model_name,
        "obelix_dev_population_n": len(pool),
        "obelix_test_untouched_row_count": n_test_untouched,
        "missing_singleton_group_kfold": result,
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

    results = {name: frag["missing_singleton_group_kfold"] for name, frag in fragments.items()}
    total_runtime = sum(frag["runtime_seconds"] for frag in fragments.values())

    # LOMFO side-effect: with __MISSING__ as 18 singletons, no group reaches
    # the LOMFO_MIN_FAMILY_N=15 threshold from __MISSING__ alone, so it
    # drops out of major_families automatically; every OTHER major family's
    # membership/rows are unaffected by this change (LOMFO holds out one
    # major family at a time, and __MISSING__'s internal grouping is
    # irrelevant to any fold that isn't the __MISSING__ exclusion fold,
    # which no longer exists under this scheme). N4b's C_lomfo results for
    # the 8 other major families are therefore reused unchanged; only the
    # sum over major_families changes (394 - 18 = 378 fewer rows scored).
    n4b_path = (
        ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
    )
    n4b = json.loads(n4b_path.read_text(encoding="utf-8"))
    lomfo_major_families_excl_missing = {
        k: v for k, v in n4b["major_families"].items() if k != "__MISSING__"
    }

    manifest = {
        "phase": "acs_issue09b_missing_singleton_groupkfold",
        "spec_path": "reports/remediation/specs/issue09_missing_family_sensitivity.md",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "obelix_dev_population_n": first["obelix_dev_population_n"],
        "obelix_test_untouched_row_count": first["obelix_test_untouched_row_count"],
        "n_missing_rows_now_singleton": 18,
        "hargreaves_data_read": False,
        "n6_external_rerun": False,
        "cv_seed": CV_SEED,
        "n_outer_splits": N_OUTER_SPLITS,
        "inner_tuning_stop_fraction": INNER_TUNING_STOP_FRACTION,
        "results": results,
        "lomfo_major_families_excluding_missing": lomfo_major_families_excl_missing,
        "lomfo_common_support_n_excluding_missing": sum(lomfo_major_families_excl_missing.values()),
        "total_runtime_seconds": total_runtime,
    }
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    guard_output(ARTIFACT_JSON).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {ARTIFACT_JSON.relative_to(ROOT)}")
    _write_report(manifest, n4b)


def _write_report(manifest: dict[str, Any], n4b: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Issue 09 Variant B — __MISSING__ rows as singleton groups\n")
    lines.append("\n**Spec:** `reports/remediation/specs/issue09_missing_family_sensitivity.md`.\n")
    lines.append(f"\nGit commit: `{manifest['git_commit']}` on branch `{manifest['git_branch']}`\n")
    lines.append(
        "\n## Pooled MAE: regime B (Family-GroupKFold, __MISSING__ as one group) vs "
        "regime B' (__MISSING__ as 18 singleton groups)\n\n"
    )
    lines.append("| Model | B: one __MISSING__ group | B': 18 singleton groups | Delta |\n")
    lines.append("|---|---|---|---|\n")
    for model_name in MODEL_NAMES:
        new_mae = manifest["results"][model_name]["pooled_mae"]
        old_mae = n4b["results"][model_name]["B_family_group_kfold"]["pooled_mae"]
        delta = new_mae - old_mae
        lines.append(f"| {model_name} | {old_mae:.4f} | {new_mae:.4f} | {delta:+.4f} |\n")
    lines.append(
        f"\nFold sizes under B' for ridge (representative): "
        f"{manifest['results']['ridge']['fold_sizes']}.\n"
    )
    lines.append(
        "\nFull per-fold hyperparameters and per-family breakdowns: "
        "`artifacts/historical/issue09b_missing_singleton_groupkfold.json`.\n"
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
