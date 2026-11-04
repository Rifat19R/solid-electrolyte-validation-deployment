#!/usr/bin/env python3
"""N6: one-shot external transfer.

THIS IS A ONE-SHOT, IRREVERSIBLE-BY-DESIGN PHASE. Executes exactly the
`predeclared_n6_protocol` frozen in `artifacts/frozen/external_transfer_lock.json` --
refit all 8 N3-frozen models on the FULL 449-row OBELiX development
population (no more fold-holdout; this is the final fit), predict once on
the strict (204-row) and relaxed (239-row) Hargreaves external cohorts, and
record MAE/RMSE/Spearman. No hyperparameter, feature, model-list, seed-list,
or cohort-membership choice is made in this script -- every one of those was
already frozen in N0-N5. This script only executes the frozen plan.

Sklearn/GBDT models (1-5): reuses `models.ladder.fit_predict_model`
unchanged -- for these model families it already fits the composition
scaler on exactly the full `train_compositions` passed in, satisfying the
lock's "refit once on the full 449-row pool" requirement as-is.

Torch models (6-8): `fit_predict_model` was written for N4's CV-fold
context, where it carves an internal ~80% early-stopping split and fits the
feature encoder (scaler/embedder/vocabulary) on that inner subset, not on
the full partition passed in -- appropriate there, but it would violate this
lock's explicit "feature scaler / element embedder / element vocabulary all
refit once on the full 449-row pool for this final fit" requirement if reused
unchanged here. Instead this script fits each torch model's feature encoder
on the full 449 rows and refits for exactly `best_epoch` epochs (the epoch
count N3's early-stopping already selected, frozen in
`artifacts/frozen/model_ladder_frozen.json`) via the existing, already-tested
`torch_regressors.train_fixed_epochs` -- the same "refit on the whole
partition at the already-selected epoch count" pattern the old (closed)
project used for its own Phase 2.8 final fit (see
`reports/phase2_7_pretest_lock.md` section 4). No new early-stopping
decision is made here; the stopping point was already fixed in N3.

The primary one-shot result is recorded in
`artifacts/frozen/n6_external_transfer_primary.json` and is never rewritten by
this script. Per-model fits (`--model <name>`) write fragments under
`artifacts/rerun/fragments/n6/`. `--merge` refuses to run while the primary
record exists unless `--force-unauthorized-rerun` is given; a forced merge is
labelled NON-PRIMARY and written only to `artifacts/rerun/n6/`.

Audit rerun (never replaces the primary record):
  python scripts/transport_analysis/n6_external_transfer.py --model <name>   # each model
  python scripts/transport_analysis/n6_external_transfer.py --merge --force-unauthorized-rerun
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.data.obelix_loader import load_obelix  # noqa: E402
from sse_transfer.data.regression_dataset import load_regression_dataset  # noqa: E402
from sse_transfer.evaluation.metrics import regression_metrics  # noqa: E402
from sse_transfer.features.composition_pipeline import TrainFitCompositionScaler  # noqa: E402
from sse_transfer.features.element_embeddings import (  # noqa: E402
    ElementVocabulary,
    TrainFitElementEmbedder,
)
from sse_transfer.models.ladder import MODEL_NAMES, fit_predict_model  # noqa: E402
from sse_transfer.models.torch_regressors import (  # noqa: E402
    CompositionAttentionRegressor,
    ShallowMLPRegressor,
    attention_forward,
    mlp_forward,
    to_tensor,
    train_fixed_epochs,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

LOCK_PATH = ROOT / "artifacts" / "frozen" / "external_transfer_lock.json"
FROZEN_LADDER_PATH = ROOT / "artifacts" / "frozen" / "model_ladder_frozen.json"
STRICT_COHORT = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_strict.csv"
RELAXED_COHORT = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_relaxed.csv"
# Primary one-shot record: read for the authorization check, never written here.
PRIMARY_RECORD = ROOT / "artifacts" / "frozen" / "n6_external_transfer_primary.json"
RERUN_JSON = ROOT / "artifacts" / "rerun" / "n6" / "n6_external_transfer_nonprimary.json"
REPORT_MD = ROOT / "reports" / "rerun" / "n6_external_transfer.md"

SEEDS: list[int] = [17, 29, 43, 71, 101]  # frozen in artifacts/frozen/external_transfer_lock.json
MAX_ELEMENTS = 10  # ElementVocabulary.encode's per-composition cap, same as N3/N4.
_SKLEARN_MODELS = {"ridge", "random_forest", "extra_trees", "xgboost", "catboost"}
_TORCH_MODELS = {"shallow_mlp", "pretrained_element_embedding", "composition_attention"}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_obelix_dev_population() -> tuple[pd.DataFrame, int]:
    """Full 449-row OBELiX development population (train UNION validation),
    columns [ID, y, true_composition]. `dataset.test` is read only for its
    length -- its features/targets are never referenced anywhere else in
    this script (grep for `.test` to verify: it appears exactly once here)."""
    dataset = load_regression_dataset()
    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)
    raw_by_id = raw.set_index("ID")

    dev = pd.concat([dataset.train, dataset.validation], ignore_index=True)
    out = dev[["ID", "y"]].copy()
    out["true_composition"] = raw_by_id.loc[out["ID"], "True Composition"].to_numpy()
    return out.reset_index(drop=True), len(dataset.test)


def load_cohort(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["reduced_formula", "log_target_rt_median"]].copy()


def fit_predict_torch_full_pool(
    model_name: str,
    hyperparameters: dict[str, Any],
    best_epoch: int,
    train_compositions: pd.Series,
    train_y: np.ndarray,
    eval_compositions: pd.Series,
    *,
    seed: int,
) -> np.ndarray:
    """Torch-model final fit: feature encoder fit on the FULL `train_compositions`
    (never an inner subset), trained for exactly `best_epoch` epochs via
    `train_fixed_epochs` (no new early-stopping decision -- N3 already made
    that decision and froze it). Satisfies the lock's "refit once on the full
    449-row pool" requirement for the feature encoder, unlike
    `models.ladder.fit_predict_model` (written for N4's fold context,
    where it correctly carves an inner early-stop split instead)."""
    y_t = to_tensor(np.asarray(train_y, dtype=float))

    if model_name == "shallow_mlp":
        scaler = TrainFitCompositionScaler().fit(train_compositions)
        Xtr = to_tensor(scaler.transform(train_compositions).to_numpy())
        Xeval = to_tensor(scaler.transform(eval_compositions).to_numpy())
        torch.manual_seed(seed)
        model: torch.nn.Module = ShallowMLPRegressor(
            in_dim=Xtr.shape[1],
            hidden_dims=tuple(hyperparameters["hidden_dims"]),
            dropout=hyperparameters["dropout"],
        )
        model = train_fixed_epochs(
            model,
            mlp_forward,
            (Xtr,),
            y_t,
            lr=hyperparameters["lr"],
            weight_decay=hyperparameters["weight_decay"],
            n_epochs=best_epoch,
            loss="mae",
            seed=seed,
        )
        model.eval()
        with torch.no_grad():
            preds = model(Xeval).numpy()
        return np.asarray(preds, dtype=float)

    if model_name == "pretrained_element_embedding":
        embedder = TrainFitElementEmbedder().fit(train_compositions)
        Etr = to_tensor(embedder.transform(train_compositions))
        Eeval = to_tensor(embedder.transform(eval_compositions))
        torch.manual_seed(seed)
        model = ShallowMLPRegressor(
            in_dim=Etr.shape[1],
            hidden_dims=tuple(hyperparameters["hidden_dims"]),
            dropout=hyperparameters["dropout"],
        )
        model = train_fixed_epochs(
            model,
            mlp_forward,
            (Etr,),
            y_t,
            lr=hyperparameters["lr"],
            weight_decay=hyperparameters["weight_decay"],
            n_epochs=best_epoch,
            loss="mae",
            seed=seed,
        )
        model.eval()
        with torch.no_grad():
            preds = model(Eeval).numpy()
        return np.asarray(preds, dtype=float)

    if model_name == "composition_attention":
        vocab = ElementVocabulary().fit(train_compositions)
        idx_tr, frac_tr, mask_tr = vocab.encode(train_compositions, max_elements=MAX_ELEMENTS)
        idx_ev, frac_ev, mask_ev = vocab.encode(eval_compositions, max_elements=MAX_ELEMENTS)
        torch.manual_seed(seed)
        model = CompositionAttentionRegressor(
            vocab_size=vocab.vocab_size,
            d_model=hyperparameters["d_model"],
            n_heads=hyperparameters["n_heads"],
            n_layers=hyperparameters["n_layers"],
            dropout=hyperparameters["dropout"],
        )
        model = train_fixed_epochs(
            model,
            attention_forward,
            (to_tensor(idx_tr, torch.long), to_tensor(frac_tr), to_tensor(mask_tr, torch.bool)),
            y_t,
            lr=hyperparameters["lr"],
            weight_decay=hyperparameters["weight_decay"],
            n_epochs=best_epoch,
            loss="mae",
            seed=seed,
        )
        model.eval()
        with torch.no_grad():
            preds = model(
                to_tensor(idx_ev, torch.long), to_tensor(frac_ev), to_tensor(mask_ev, torch.bool)
            ).numpy()
        return np.asarray(preds, dtype=float)

    raise AssertionError(f"unreachable: unhandled torch model_name {model_name!r}")


def score_cohort(
    predictions_by_model: dict[str, np.ndarray], y_true: np.ndarray
) -> dict[str, dict[str, float]]:
    """Pure scoring function: {model_name: regression_metrics(...)}. No
    fitting, no data loading -- testable with synthetic arrays alone."""
    out: dict[str, dict[str, float]] = {}
    for name, preds in predictions_by_model.items():
        out[name] = regression_metrics(y_true, preds)
    return out


def check_run_authorization(output_path: Path, force_unauthorized_rerun: bool) -> tuple[int, bool]:
    """Returns (run_count, authorized). If `output_path` already holds a
    primary (authorized=True) result and `force_unauthorized_rerun` is not
    set, raises -- this is the "one authorized run" guard. If
    `force_unauthorized_rerun` is set, the new run is explicitly marked
    non-primary (authorized=False) and run_count is incremented; it must
    never silently overwrite the original result as if it were primary."""
    if not output_path.exists():
        return 1, True
    existing = json.loads(output_path.read_text(encoding="utf-8"))
    prior_count = int(existing.get("run_count", 1))
    prior_authorized = bool(existing.get("authorized", True))
    if prior_authorized and not force_unauthorized_rerun:
        raise RuntimeError(
            f"{output_path} already holds an authorized primary N6 result "
            f"(run_count={prior_count}). N6 is a one-shot phase per "
            "artifacts/frozen/external_transfer_lock.json's predeclared_n6_protocol. Re-run with "
            "--force-unauthorized-rerun only if you explicitly intend to produce a "
            "clearly-flagged NON-primary result for audit purposes; it will never "
            "silently replace the primary result."
        )
    return prior_count + 1, False


FRAGMENTS_DIR = ROOT / "artifacts" / "fragments" / "n6"
# Recorded fragments (read by --merge) are immutable; --model reruns write here.
FRAGMENT_OUT_DIR = ROOT / "artifacts" / "rerun" / "fragments" / "n6"


def run_one_model(model_name: str) -> None:
    """Fits/predicts one model in its own OS process and writes a fragment
    JSON. Run as a separate process per model (see `main`'s `--model` path)
    because running all 8 models (sklearn + GBDT + 3 torch candidates) in one
    long-lived process was OOM-killed on this environment's limited RAM --
    the same failure mode N4 hit and fixed the same way
    (`scripts/transport_analysis/n4_internal_validation.py`'s `--model`/`--merge` split)."""
    t0 = __import__("time").time()
    frozen_ladder = json.loads(FROZEN_LADDER_PATH.read_text(encoding="utf-8"))
    model_info = frozen_ladder["models"][model_name]
    hyperparameters = model_info["hyperparameters"]

    dev_df, n_test_untouched = load_obelix_dev_population()
    strict_df = load_cohort(STRICT_COHORT)
    relaxed_df = load_cohort(RELAXED_COHORT)
    train_comp = dev_df["true_composition"]
    train_y = dev_df["y"].to_numpy()

    seed_preds_strict: list[np.ndarray] = []
    seed_preds_relaxed: list[np.ndarray] = []
    for seed in SEEDS:
        if model_name in _SKLEARN_MODELS:
            p_strict = fit_predict_model(
                model_name,
                hyperparameters,
                train_comp,
                train_y,
                strict_df["reduced_formula"],
                seed=seed,
            )
            p_relaxed = fit_predict_model(
                model_name,
                hyperparameters,
                train_comp,
                train_y,
                relaxed_df["reduced_formula"],
                seed=seed,
            )
        else:
            best_epoch = int(model_info["best_epoch"])
            p_strict = fit_predict_torch_full_pool(
                model_name,
                hyperparameters,
                best_epoch,
                train_comp,
                train_y,
                strict_df["reduced_formula"],
                seed=seed,
            )
            p_relaxed = fit_predict_torch_full_pool(
                model_name,
                hyperparameters,
                best_epoch,
                train_comp,
                train_y,
                relaxed_df["reduced_formula"],
                seed=seed,
            )
        seed_preds_strict.append(p_strict)
        seed_preds_relaxed.append(p_relaxed)
        print(f"  [{model_name}] seed={seed} done")

    stack_strict = np.stack(seed_preds_strict)  # (5, n_strict)
    stack_relaxed = np.stack(seed_preds_relaxed)  # (5, n_relaxed)
    mean_strict = stack_strict.mean(axis=0)
    mean_relaxed = stack_relaxed.mean(axis=0)
    std_strict = stack_strict.std(axis=0)
    std_relaxed = stack_relaxed.std(axis=0)

    metrics_strict = regression_metrics(strict_df["log_target_rt_median"].to_numpy(), mean_strict)
    metrics_relaxed = regression_metrics(
        relaxed_df["log_target_rt_median"].to_numpy(), mean_relaxed
    )

    fragment = {
        "model_name": model_name,
        "obelix_dev_population_n": len(dev_df),
        "obelix_test_untouched_row_count": n_test_untouched,
        "strict": {
            **metrics_strict,
            "n": len(strict_df),
            "seed_dispersion_std_mean": float(std_strict.mean()),
            "seed_dispersion_std_median": float(np.median(std_strict)),
            "predictions": {
                "reduced_formula": strict_df["reduced_formula"].tolist(),
                "y_true": strict_df["log_target_rt_median"].tolist(),
                "y_pred": mean_strict.tolist(),
                "residual": (mean_strict - strict_df["log_target_rt_median"].to_numpy()).tolist(),
            },
        },
        "relaxed": {
            **metrics_relaxed,
            "n": len(relaxed_df),
            "seed_dispersion_std_mean": float(std_relaxed.mean()),
            "seed_dispersion_std_median": float(np.median(std_relaxed)),
            "predictions": {
                "reduced_formula": relaxed_df["reduced_formula"].tolist(),
                "y_true": relaxed_df["log_target_rt_median"].tolist(),
                "y_pred": mean_relaxed.tolist(),
                "residual": (mean_relaxed - relaxed_df["log_target_rt_median"].to_numpy()).tolist(),
            },
        },
        "runtime_seconds": __import__("time").time() - t0,
    }
    print(
        f"[{model_name}] external strict MAE={metrics_strict['mae']:.4f} "
        f"relaxed MAE={metrics_relaxed['mae']:.4f}"
    )
    FRAGMENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    frag_path = FRAGMENT_OUT_DIR / f"{model_name}.json"
    guard_output(frag_path).write_text(
        json.dumps(fragment, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {frag_path.relative_to(ROOT)}")


def merge_fragments(
    force_unauthorized_rerun: bool,
    fragments_dir: Path | None = None,
    primary_record: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Merge per-model fragments into a NON-PRIMARY audit result.

    The primary one-shot record already exists; this function can only
    produce a clearly labelled non-primary rerun and can never write the
    primary record (``guard_output`` rejects every protected location).
    """
    fragments_dir = FRAGMENT_OUT_DIR if fragments_dir is None else fragments_dir
    primary_record = PRIMARY_RECORD if primary_record is None else primary_record
    output_path = RERUN_JSON if output_path is None else output_path
    if not primary_record.exists():
        raise RuntimeError(
            f"{primary_record} is missing. The one-shot primary external transfer was "
            "executed once and is recorded there; this repository never creates a new "
            "primary result. Restore the frozen record instead of re-running N6."
        )
    run_count, authorized = check_run_authorization(primary_record, force_unauthorized_rerun)
    if authorized:  # pragma: no cover - unreachable while the primary record exists
        raise RuntimeError("a merge can never be authorized as primary")
    print(f"N6 run_count={run_count} authorized={authorized}")
    lock_hash = _sha256_file(LOCK_PATH)

    fragments: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        frag_path = fragments_dir / f"{model_name}.json"
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

    results = {
        name: {"strict": frag["strict"], "relaxed": frag["relaxed"]}
        for name, frag in fragments.items()
    }
    ranking = sorted(MODEL_NAMES, key=lambda m: results[m]["strict"]["mae"])
    for rank, name in enumerate(ranking, start=1):
        results[name]["strict"]["rank_by_strict_mae"] = rank

    manifest = {
        "phase": "N6_one_shot_external_transfer",
        "status": "NON-PRIMARY audit rerun; the primary result is "
        "artifacts/frozen/n6_external_transfer_primary.json",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "lock_file": "artifacts/frozen/external_transfer_lock.json",
        "lock_file_sha256": lock_hash,
        "run_count": run_count,
        "authorized": authorized,
        "obelix_dev_population_n": first["obelix_dev_population_n"],
        "obelix_test_untouched_row_count": first["obelix_test_untouched_row_count"],
        "seeds": SEEDS,
        "model_ranking_by_strict_mae": ranking,
        "results": results,
        "total_runtime_seconds": sum(frag["runtime_seconds"] for frag in fragments.values()),
        "execution_note": (
            "Each model was fit/predicted in its own OS process (--model <name>), then merged "
            "(--merge), to avoid the OOM kill observed when running all 8 models in one "
            "long-lived process on this environment's limited RAM -- same fix N4 used."
        ),
    }
    guard_output(output_path).parent.mkdir(parents=True, exist_ok=True)
    guard_output(output_path).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {output_path} (NON-PRIMARY)")

    _write_report(manifest)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--force-unauthorized-rerun", action="store_true")
    args = parser.parse_args()

    if args.merge:
        merge_fragments(args.force_unauthorized_rerun)
        return
    if args.model is not None:
        run_one_model(args.model)
        return
    parser.error("pass either --model <name> or --merge")


def _write_report(manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# N6 — One-Shot External Transfer\n\n")
    status = (
        "PRIMARY (authorized)" if manifest["authorized"] else "NON-PRIMARY (unauthorized rerun)"
    )
    lines.append(f"Status: **{status}**, run_count={manifest['run_count']}\n\n")
    lines.append(f"Git commit: `{manifest['git_commit']}` on branch `{manifest['git_branch']}`\n")
    lines.append(f"Lock file hash: `{manifest['lock_file_sha256']}`\n\n")
    lines.append("## 1. Population and leakage boundary\n\n")
    lines.append(
        f"- OBELiX development population: {manifest['obelix_dev_population_n']} rows "
        "(frozen train UNION validation) -- the final-fit population for every model.\n"
        f"- OBELiX frozen test split ({manifest['obelix_test_untouched_row_count']} rows) read "
        "only for its length (`load_obelix_dev_population` in "
        "`scripts/transport_analysis/n6_external_transfer.py` -- grep for `.test`, "
        "appears exactly once).\n"
        "- Hyperparameters: exactly `artifacts/frozen/model_ladder_frozen.json`'s frozen values; "
        "zero retuning, zero Optuna calls in this script.\n"
        "- Seeds: each model fit once per seed in "
        f"{manifest['seeds']} (5 fits/model, 40 total fits); "
        "point prediction = mean across seeds.\n"
    )
    lines.append("\n## 2. External MAE / RMSE / Spearman — strict cohort (primary)\n\n")
    lines.append("| Rank | Model | MAE | RMSE | Spearman rho | Seed-dispersion (mean std) | n |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for name in manifest["model_ranking_by_strict_mae"]:
        r = manifest["results"][name]["strict"]
        lines.append(
            f"| {r['rank_by_strict_mae']} | {name} | {r['mae']:.4f} | {r['rmse']:.4f} | "
            f"{r['spearman_rho']:.4f} | {r['seed_dispersion_std_mean']:.4f} | {r['n']} |\n"
        )
    lines.append(
        "\n## 3. External MAE / RMSE / Spearman — relaxed cohort (companion sensitivity)\n\n"
    )
    lines.append("| Model | MAE | RMSE | Spearman rho | n |\n")
    lines.append("|---|---|---|---|---|\n")
    for name in manifest["model_ranking_by_strict_mae"]:
        r = manifest["results"][name]["relaxed"]
        lines.append(
            f"| {name} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['spearman_rho']:.4f} | {r['n']} |\n"
        )
    lines.append(
        "\n## 4. Not done in N6 (by design)\n\n"
        "- No comparison of these numbers to N4's internal-validation regimes or to H1/H2 -- "
        "that is N7.\n"
        "- No applicability-domain/risk analysis -- that is N8.\n"
        "- No re-tuning of any hyperparameter after these external numbers were computed.\n"
        "- No OBELiX-test evaluation at all.\n"
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_MD).parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_MD).write_text("".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
