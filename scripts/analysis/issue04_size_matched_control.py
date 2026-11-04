#!/usr/bin/env python3
"""Reviewer remediation, Issue 04: training-size-matched random control for
regime B (Family-GroupKFold).

**Why this script exists:** regime B's outer folds remove a large family
from training entirely, confounding two effects: (1) training-set size
reduction, and (2) a distribution shift from excluding an entire chemical
family. This script isolates effect (1) by building, for each of B's 5
folds, a plain random row-level holdout of the IDENTICAL test-set size
(ignoring family membership), then running the SAME nested tuning pipeline
N4b used (inner-split tune_hyperparameters on that fold's own outer-train
partition, then fit_predict_model) -- full nested retuning, not a reuse of
B's frozen hyperparameters, so this control is judged under the same
protocol as the reference regime it is compared against.

Spec: reports/remediation/specs/issue04_training_size_matched_control.md (hashed;
see reports/remediation/specs/HASHES.txt). This script's compute was already in
flight (started by an out-of-scope background process, see
the since-removed quarantine directory) when that spec was written;
the spec was independently drafted, found methodologically identical to
this script's own inline docstring/spec reference
(artifacts/frozen/remediation_spec.json), and the results were independently
re-verified by rerunning one (model, seed) pair from scratch after the
sweep completed (byte-identical output except wall-clock runtime) before
being trusted -- see reports/remediation/issue04_training_size_matched_control.md
for the full disclosure. Relocated from scripts/transport_analysis/ to scripts/analysis/ after
verification, since this is reviewer-remediation work.

Scope decision (disclosed in the spec): 3 deterministic control seeds,
regime B only (not C/LOMFO), to bound compute to the agreed budget.

Does NOT touch N6 (external transfer, frozen) or any N4b/N6
artifact. Reads the OBELiX dev population exactly as N4b does; the frozen
113-row OBELiX test split is opened only for its length.

Run per (seed, model) (own OS process, same OOM-avoidance rationale as N4b):
    python scripts/analysis/issue04_size_matched_control.py --seed <seed> --model <name>
Then merge:
    python scripts/analysis/issue04_size_matched_control.py --merge
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
    N_OUTER_SPLITS,
    canonicalize_and_fill_family,
    family_group_kfold_splits,
    inner_early_stop_split,
    size_matched_random_splits,
)
from sse_transfer.evaluation.metrics import regression_metrics  # noqa: E402
from sse_transfer.models.ladder import MODEL_NAMES, fit_predict_model  # noqa: E402
from sse_transfer.training.tune_fold import tune_hyperparameters  # noqa: E402
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

ARTIFACT_JSON = ROOT / "artifacts" / "rerun" / "issue04_size_matched_control_corrected.json"
REPORT_MD = ROOT / "reports" / "rerun" / "issue04_size_matched_control_raw.md"
FRAGMENTS_DIR = ROOT / "artifacts" / "fragments" / "issue04"
# Recorded fragments (read by --merge) are immutable; --model reruns write here.
FRAGMENT_OUT_DIR = ROOT / "artifacts" / "rerun" / "fragments" / "issue04"
N4B_ARTIFACT = (
    ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
)

FOLD_CONSTRUCTION_SEED = 20260921  # same seed N4b used to build regime B's folds
CONTROL_SEEDS = (20260930, 20260931, 20260932)  # 3 deterministic control seeds (agreed scope)
INNER_TUNING_STOP_FRACTION = 0.2  # identical to N4b


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def load_dev_population() -> tuple[pd.DataFrame, int]:
    """Identical to N4b's loader -- same dev population, same ID ordering."""
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
    """Identical nested-tuning cycle to N4b's `_nested_fold_predict`."""
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


def run_control_seed(pool: pd.DataFrame, model_name: str, control_seed: int) -> dict[str, Any]:
    reference_b_folds = family_group_kfold_splits(
        pool["family_filled"], n_splits=N_OUTER_SPLITS, seed=FOLD_CONSTRUCTION_SEED
    )
    n = len(pool)
    matched_folds = size_matched_random_splits(n, reference_b_folds, seed=control_seed)
    oof_pred = np.full(n, np.nan)
    fold_hyperparameters: list[dict[str, Any]] = []
    for fold_i, (train_idx, test_idx) in enumerate(matched_folds):
        preds, hp = _nested_fold_predict(
            model_name,
            pool["true_composition"].iloc[train_idx],
            pool["y"].to_numpy()[train_idx],
            pool["true_composition"].iloc[test_idx],
            seed=control_seed,
        )
        oof_pred[test_idx] = preds
        fold_hyperparameters.append(hp)
        print(f"    [seed {control_seed} fold {fold_i}] done, n_test={len(test_idx)}")
    if np.isnan(oof_pred).any():
        raise RuntimeError("run_control_seed: some pool row never appeared in a test fold")
    metrics = _oof_metrics(pool["y"].to_numpy(), oof_pred, pool["family_filled"].to_numpy())
    metrics["regime"] = "E_size_matched_random_control"
    metrics["reference_regime"] = "B_family_group_kfold"
    metrics["control_seed"] = control_seed
    metrics["fold_construction_seed"] = FOLD_CONSTRUCTION_SEED
    metrics["n_folds"] = len(matched_folds)
    metrics["fold_test_sizes"] = [len(t) for _, t in matched_folds]
    metrics["per_fold_hyperparameters"] = fold_hyperparameters
    metrics["note"] = (
        "Test-fold sizes are IDENTICAL to regime B's 5 folds, but row membership is a "
        "plain random draw, not family-grouped. Hyperparameters were tuned per fold using "
        "the same nested inner-split protocol as N4b (not reused from B's frozen "
        "hyperparameters)."
    )
    return metrics


def run_one(model_name: str, control_seed: int) -> None:
    t0 = time.time()
    pool, n_test_untouched = load_dev_population()
    print(f"=== SIZE-MATCHED CONTROL {model_name} seed={control_seed} ===")
    result = run_control_seed(pool, model_name, control_seed)
    runtime = time.time() - t0
    print(f"  pooled MAE={result['pooled_mae']:.4f} ({runtime:.1f}s)")

    fragment = {
        "model_name": model_name,
        "control_seed": control_seed,
        "obelix_dev_population_n": len(pool),
        "obelix_test_untouched_row_count": n_test_untouched,
        "result": result,
        "runtime_seconds": runtime,
    }
    FRAGMENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    frag_path = FRAGMENT_OUT_DIR / f"{model_name}__seed{control_seed}.json"
    guard_output(frag_path).write_text(
        json.dumps(fragment, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {frag_path.relative_to(ROOT)}")


def merge_fragments() -> None:
    fragments: dict[tuple[str, int], dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        for seed in CONTROL_SEEDS:
            frag_path = FRAGMENTS_DIR / f"{model_name}__seed{seed}.json"
            if not frag_path.exists():
                raise FileNotFoundError(
                    f"Missing fragment {frag_path}; run "
                    f"`python {Path(__file__).name} --model {model_name} --seed {seed}` first."
                )
            fragments[(model_name, seed)] = json.loads(frag_path.read_text(encoding="utf-8"))

    first = next(iter(fragments.values()))
    for (model_name, seed), frag in fragments.items():
        if frag["obelix_dev_population_n"] != first["obelix_dev_population_n"]:
            raise RuntimeError(f"Fragment mismatch: {model_name}/{seed} used a different pop size.")

    n4b = json.loads(N4B_ARTIFACT.read_text(encoding="utf-8"))

    results: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        results[model_name] = {
            "regime_A_random_5fold_cv_pooled_mae": n4b["results"][model_name]["A_random_5fold_cv"][
                "pooled_mae"
            ],
            "regime_B_family_group_kfold_pooled_mae": n4b["results"][model_name][
                "B_family_group_kfold"
            ]["pooled_mae"],
            "size_matched_control_by_seed": {
                str(seed): fragments[(model_name, seed)]["result"] for seed in CONTROL_SEEDS
            },
        }

    total_runtime = sum(frag["runtime_seconds"] for frag in fragments.values())
    manifest = {
        "phase": "ACS_issue04_size_matched_control",
        "spec": "artifacts/frozen/remediation_spec.json#issue_04_training_size_matched_control",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "obelix_dev_population_n": first["obelix_dev_population_n"],
        "obelix_test_untouched_row_count": first["obelix_test_untouched_row_count"],
        "fold_construction_seed": FOLD_CONSTRUCTION_SEED,
        "control_seeds": list(CONTROL_SEEDS),
        "n4b_source": str(N4B_ARTIFACT.relative_to(ROOT)),
        "n6_rerun": False,
        "results": results,
        "total_runtime_seconds": total_runtime,
    }
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    guard_output(ARTIFACT_JSON).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {ARTIFACT_JSON.relative_to(ROOT)}")
    _write_report(manifest)
    print(f"total runtime (summed across per-(model,seed) processes): {total_runtime:.1f}s")


def _write_report(manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# ACS Issue 04 — Training-Size-Matched Random Control\n\n")
    lines.append(f"Git commit: `{manifest['git_commit']}` on branch `{manifest['git_branch']}`\n\n")
    lines.append(
        "Spec: `artifacts/frozen/remediation_spec.json#issue_04_training_size_matched_control`. "
        "N6 external transfer was NOT rerun. Scope: regime B (Family-GroupKFold) only, "
        f"{len(manifest['control_seeds'])} deterministic control seeds "
        f"({', '.join(str(s) for s in manifest['control_seeds'])}), disclosed budget-bounded "
        "scope decision (see spec).\n\n"
    )
    lines.append(
        "## Pooled MAE: regime A (random CV) vs regime B (Family-GroupKFold, actual) vs "
        "size-matched random control (same fold sizes as B, random membership)\n\n"
    )
    lines.append(
        "| Model | A (random CV) | B (Family-GroupKFold, actual) | "
        + " | ".join(f"E seed {s} (size-matched random)" for s in manifest["control_seeds"])
        + " |\n"
    )
    lines.append("|---|---|---|" + "---|" * len(manifest["control_seeds"]) + "\n")
    for model_name, r in manifest["results"].items():
        row = [
            model_name,
            f"{r['regime_A_random_5fold_cv_pooled_mae']:.4f}",
            f"{r['regime_B_family_group_kfold_pooled_mae']:.4f}",
        ]
        for seed in manifest["control_seeds"]:
            row.append(f"{r['size_matched_control_by_seed'][str(seed)]['pooled_mae']:.4f}")
        lines.append("| " + " | ".join(row) + " |\n")
    lines.append(
        "\n## Interpretation guide (per spec's decision rule)\n\n"
        "If a model's size-matched-random MAE (E) sits close to regime A's MAE and far below "
        "regime B's actual MAE, degradation under B is attributable to family-exclusion "
        "distribution shift, not size loss alone. If E is ALSO close to B's blown-up MAE, "
        "size loss alone is sufficient to explain the degradation. No causal claim is made "
        "beyond what this direct three-way comparison shows.\n"
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_MD).write_text("".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_MD.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--seed", type=int, choices=CONTROL_SEEDS, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_fragments()
        return
    if args.model is not None and args.seed is not None:
        run_one(args.model, args.seed)
        return
    parser.error("pass either --model <name> --seed <seed>, or --merge")


if __name__ == "__main__":
    main()
