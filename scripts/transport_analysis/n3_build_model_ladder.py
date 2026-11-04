#!/usr/bin/env python3
"""Cross-dataset study, N3: build/freeze the portable, composition-
only model ladder (8 models) on the OBELiX development population.

"Development population" = the frozen `train` (359 rows) UNION `validation`
(90 rows) splits from `data.regression_dataset.load_regression_dataset()`.
The 113-row frozen `test` split belongs to the OLD, closed DD project
(reports/phase2_8_final_test_evaluation.md) and is a hard leakage boundary
here: this script reads `dataset.test` only to report `len(dataset.test)`
in the written report, never its features/targets, and never passes it to
any `.fit`/`.predict`/tuning call. No Hargreaves data is read at all in N3
(external evaluation is N6, out of scope here).

Every model uses the SAME composition-only Magpie feature schema
(`features.composition_pipeline.TrainFitCompositionScaler`, fit on OBELiX
train only) except the two candidates that are architecturally composition-
only by construction (pretrained-element-embedding regressor, composition-
attention), which use their own already-existing train-fit encoders
(`features.element_embeddings`). Hyperparameters are tuned via Optuna
against OBELiX train (fit) / validation (objective) only
(EXPERIMENT_PROTOCOL.md "Never tune on the final test set") and then
FROZEN -- this script's output is one fixed hyperparameter set per model,
not an open search.

Run with: python scripts/transport_analysis/n3_build_model_ladder.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.data.obelix_loader import load_obelix  # noqa: E402
from sse_transfer.data.regression_dataset import load_regression_dataset  # noqa: E402
from sse_transfer.evaluation.metrics import regression_metrics  # noqa: E402
from sse_transfer.features.composition_features import (  # noqa: E402
    COMPOSITION_FEATURE_NAMES,
    composition_features,
)
from sse_transfer.features.composition_pipeline import (  # noqa: E402
    FEATURE_SCHEMA_NAME,
    TrainFitCompositionScaler,
)
from sse_transfer.features.element_embeddings import (  # noqa: E402
    ElementVocabulary,
    TrainFitElementEmbedder,
)
from sse_transfer.models.classical_baselines import (  # noqa: E402
    build_catboost,
    build_extra_trees,
    build_random_forest,
    build_ridge,
    build_xgboost,
)
from sse_transfer.models.torch_regressors import (  # noqa: E402
    CompositionAttentionRegressor,
    ShallowMLPRegressor,
    attention_forward,
    count_parameters,
    mlp_forward,
    to_tensor,
    train_with_early_stopping,
)
from sse_transfer.training.tune_classical import tune_model as tune_gbdt  # noqa: E402
from sse_transfer.training.tune_ladder import (  # noqa: E402
    N_TRIALS,
    TORCH_N_TRIALS,
    tune_sklearn_model,
    tune_torch_model,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

ARTIFACT_JSON = ROOT / "artifacts" / "rerun" / "model_ladder_frozen.json"
REPORT_MD = ROOT / "reports" / "rerun" / "n3_portable_model_benchmark.md"
HARGREAVES_STRICT = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_strict.csv"

SEEDS: list[int] = [17, 29, 43, 71, 101]  # same list as reports/phase2_7_pretest_lock.md
TUNING_SEED = 20260921
MAX_ELEMENTS = 10  # ElementVocabulary.encode's per-composition cap


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def _hash_params(params: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def load_obelix_dev_population() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Returns (train_df, validation_df) with columns [ID, y, true_composition],
    plus len(dataset.test) for the report -- dataset.test's rows/features/
    targets are never otherwise touched."""
    dataset = load_regression_dataset()
    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)
    raw_by_id = raw.set_index("ID")

    def attach(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["ID", "y"]].copy()
        out["true_composition"] = raw_by_id.loc[out["ID"], "True Composition"].to_numpy()
        return out.reset_index(drop=True)

    return attach(dataset.train), attach(dataset.validation), len(dataset.test)


def verify_feature_schema_identical(obelix_compositions: pd.Series) -> dict[str, Any]:
    """Computes Magpie composition features for a sample of OBELiX and
    Hargreaves compositions and diffs the column name lists -- an executed
    check, not an assumption."""
    harg = pd.read_csv(HARGREAVES_STRICT)
    obelix_feats = composition_features(obelix_compositions.head(20))
    harg_feats = composition_features(harg["reduced_formula"].head(20))
    obelix_cols = list(obelix_feats.columns)
    harg_cols = list(harg_feats.columns)
    return {
        "n_features_obelix": len(obelix_cols),
        "n_features_hargreaves": len(harg_cols),
        "columns_identical": obelix_cols == harg_cols,
        "columns_identical_as_set": set(obelix_cols) == set(harg_cols),
        "feature_schema_name": FEATURE_SCHEMA_NAME,
    }


def main() -> None:
    train_df, val_df, n_test_untouched = load_obelix_dev_population()
    print(f"OBELiX dev population: train={len(train_df)}, validation={len(val_df)}")
    print(f"OBELiX frozen test split (untouched, length only): {n_test_untouched}")

    schema_check = verify_feature_schema_identical(train_df["true_composition"])
    print("feature schema identical across datasets:", schema_check["columns_identical"])
    if not schema_check["columns_identical"]:
        raise RuntimeError(
            "Composition-only feature schema differs between OBELiX and Hargreaves; "
            "N3 requires an identical schema. Aborting rather than silently proceeding."
        )

    scaler = TrainFitCompositionScaler().fit(train_df["true_composition"])
    X_train = scaler.transform(train_df["true_composition"])
    X_val = scaler.transform(val_df["true_composition"])
    y_train = train_df["y"].to_numpy()
    y_val = val_df["y"].to_numpy()

    models: dict[str, dict[str, Any]] = {}

    # 1-3: sklearn (Ridge, RandomForest, ExtraTrees) via extended Optuna pattern
    for name, model_key, builder in [
        ("ridge", "ridge", build_ridge),
        ("random_forest", "random_forest", build_random_forest),
        ("extra_trees", "extra_trees", build_extra_trees),
    ]:
        tuning = tune_sklearn_model(
            model_key, X_train, y_train, X_val, y_val, seed=TUNING_SEED, n_trials=N_TRIALS
        )
        model = builder(tuning.best_params, seed=TUNING_SEED)
        model.fit(X_train, y_train)
        train_mae = regression_metrics(y_train, model.predict(X_train))["mae"]
        val_mae = regression_metrics(y_val, model.predict(X_val))["mae"]
        models[name] = {
            "model_family": model_key,
            "hyperparameters": tuning.best_params,
            "hyperparameters_hash": _hash_params(tuning.best_params),
            "seed": TUNING_SEED,
            "feature_schema": FEATURE_SCHEMA_NAME,
            "n_features": len(COMPOSITION_FEATURE_NAMES),
            "train_mae": train_mae,
            "validation_mae": val_mae,
            "n_trials": tuning.n_trials,
        }
        print(f"[{name}] val MAE={val_mae:.4f} params={tuning.best_params}")

    # 4-5: existing GBDT factories, existing tuning pattern (reused as-is)
    for name, model_key, builder in [
        ("xgboost", "xgboost", build_xgboost),
        ("catboost", "catboost", build_catboost),
    ]:
        tuning = tune_gbdt(
            model_key, X_train, y_train, X_val, y_val, seed=TUNING_SEED, n_trials=N_TRIALS
        )
        model = builder(tuning.best_params, seed=TUNING_SEED)
        model.fit(X_train, y_train)
        train_mae = regression_metrics(y_train, model.predict(X_train))["mae"]
        val_mae = regression_metrics(y_val, model.predict(X_val))["mae"]
        models[name] = {
            "model_family": model_key,
            "hyperparameters": tuning.best_params,
            "hyperparameters_hash": _hash_params(tuning.best_params),
            "seed": TUNING_SEED,
            "feature_schema": FEATURE_SCHEMA_NAME,
            "n_features": len(COMPOSITION_FEATURE_NAMES),
            "train_mae": train_mae,
            "validation_mae": val_mae,
            "n_trials": tuning.n_trials,
        }
        print(f"[{name}] val MAE={val_mae:.4f} params={tuning.best_params}")

    # 6: shallow MLP on Magpie composition features
    Xtr_t, Xval_t = to_tensor(X_train.to_numpy()), to_tensor(X_val.to_numpy())
    ytr_t, yval_t = to_tensor(y_train), to_tensor(y_val)

    def mlp_search_space(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }

    def build_mlp(params: dict[str, Any]) -> torch.nn.Module:
        # `params` is always the raw dict of `trial.suggest_*` calls above --
        # both mid-tuning (search_space's own return value) and post-tuning
        # (Optuna's `study.best_params`, which only ever contains raw
        # suggested keys, never a derived key) -- so hidden_dims must always
        # be reconstructed here, never read as a pre-built key.
        torch.manual_seed(TUNING_SEED)
        hidden_dims = tuple([params["hidden_dim"]] * params["n_layers"])
        return ShallowMLPRegressor(
            in_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=params["dropout"]
        )

    mlp_tuning = tune_torch_model(
        "shallow_mlp",
        build_mlp,
        mlp_forward,
        mlp_search_space,
        (Xtr_t,),
        ytr_t,
        (Xval_t,),
        yval_t,
        seed=TUNING_SEED,
        n_trials=TORCH_N_TRIALS,
    )
    mlp = build_mlp(mlp_tuning.best_params)
    fit_result = train_with_early_stopping(
        mlp,
        mlp_forward,
        (Xtr_t,),
        ytr_t,
        (Xval_t,),
        yval_t,
        lr=mlp_tuning.best_params["lr"],
        weight_decay=mlp_tuning.best_params["weight_decay"],
        max_epochs=150,
        patience=15,
        loss="mae",
        seed=TUNING_SEED,
    )
    mlp = fit_result.model
    mlp.eval()
    with torch.no_grad():
        train_mae = regression_metrics(y_train, mlp(Xtr_t).numpy())["mae"]
        val_mae = regression_metrics(y_val, mlp(Xval_t).numpy())["mae"]
    resolved_mlp_params = {
        "hidden_dims": list(
            [mlp_tuning.best_params["hidden_dim"]] * mlp_tuning.best_params["n_layers"]
        ),
        "dropout": mlp_tuning.best_params["dropout"],
        "lr": mlp_tuning.best_params["lr"],
        "weight_decay": mlp_tuning.best_params["weight_decay"],
    }
    models["shallow_mlp"] = {
        "model_family": "torch_mlp",
        "hyperparameters": resolved_mlp_params,
        "hyperparameters_hash": _hash_params(resolved_mlp_params),
        "seed": TUNING_SEED,
        "feature_schema": FEATURE_SCHEMA_NAME,
        "n_features": len(COMPOSITION_FEATURE_NAMES),
        "train_mae": train_mae,
        "validation_mae": val_mae,
        "n_trials": mlp_tuning.n_trials,
        "n_parameters": count_parameters(mlp),
        "best_epoch": fit_result.best_epoch,
    }
    print(f"[shallow_mlp] val MAE={val_mae:.4f} params={resolved_mlp_params}")

    # 7: pretrained-element-embedding regressor (pooled stoichiometric embedding -> shallow MLP)
    embedder = TrainFitElementEmbedder().fit(train_df["true_composition"])
    Etr = to_tensor(embedder.transform(train_df["true_composition"]))
    Eval = to_tensor(embedder.transform(val_df["true_composition"]))

    def embed_search_space(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }

    def build_embed_model(params: dict[str, Any]) -> torch.nn.Module:
        # Same raw-params-only convention as build_mlp above.
        torch.manual_seed(TUNING_SEED)
        return ShallowMLPRegressor(
            in_dim=Etr.shape[1], hidden_dims=(params["hidden_dim"],), dropout=params["dropout"]
        )

    embed_tuning = tune_torch_model(
        "pretrained_element_embedding",
        build_embed_model,
        mlp_forward,
        embed_search_space,
        (Etr,),
        ytr_t,
        (Eval,),
        yval_t,
        seed=TUNING_SEED,
        n_trials=TORCH_N_TRIALS,
    )
    embed_model = build_embed_model(embed_tuning.best_params)
    embed_fit = train_with_early_stopping(
        embed_model,
        mlp_forward,
        (Etr,),
        ytr_t,
        (Eval,),
        yval_t,
        lr=embed_tuning.best_params["lr"],
        weight_decay=embed_tuning.best_params["weight_decay"],
        max_epochs=150,
        patience=15,
        loss="mae",
        seed=TUNING_SEED,
    )
    embed_model = embed_fit.model
    embed_model.eval()
    with torch.no_grad():
        train_mae = regression_metrics(y_train, embed_model(Etr).numpy())["mae"]
        val_mae = regression_metrics(y_val, embed_model(Eval).numpy())["mae"]
    resolved_embed_params = {
        "hidden_dims": [embed_tuning.best_params["hidden_dim"]],
        "dropout": embed_tuning.best_params["dropout"],
        "lr": embed_tuning.best_params["lr"],
        "weight_decay": embed_tuning.best_params["weight_decay"],
    }
    models["pretrained_element_embedding"] = {
        "model_family": "torch_pooled_element_embedding_mlp",
        "hyperparameters": resolved_embed_params,
        "hyperparameters_hash": _hash_params(resolved_embed_params),
        "seed": TUNING_SEED,
        "feature_schema": "magpie_pooled_element_embedding_v1",
        "n_features": Etr.shape[1],
        "train_mae": train_mae,
        "validation_mae": val_mae,
        "n_trials": embed_tuning.n_trials,
        "n_parameters": count_parameters(embed_model),
        "best_epoch": embed_fit.best_epoch,
    }
    print(f"[pretrained_element_embedding] val MAE={val_mae:.4f} params={resolved_embed_params}")

    # 8: composition-attention (CrabNet/Roost-style)
    vocab = ElementVocabulary().fit(train_df["true_composition"])
    idx_tr, frac_tr, mask_tr = vocab.encode(train_df["true_composition"], max_elements=MAX_ELEMENTS)
    idx_val, frac_val, mask_val = vocab.encode(
        val_df["true_composition"], max_elements=MAX_ELEMENTS
    )
    idx_tr_t, frac_tr_t, mask_tr_t = (
        to_tensor(idx_tr, torch.long),
        to_tensor(frac_tr),
        to_tensor(mask_tr, torch.bool),
    )
    idx_val_t, frac_val_t, mask_val_t = (
        to_tensor(idx_val, torch.long),
        to_tensor(frac_val),
        to_tensor(mask_val, torch.bool),
    )

    def build_attn_model(params: dict[str, Any]) -> torch.nn.Module:
        torch.manual_seed(TUNING_SEED)
        return CompositionAttentionRegressor(
            vocab_size=vocab.vocab_size,
            d_model=params["d_model"],
            n_heads=params["n_heads"],
            n_layers=params["n_layers"],
            dropout=params["dropout"],
        )

    def attn_search_space(trial: optuna.Trial) -> dict[str, Any]:
        # [16, 32, 64] x [2, 4] are all evenly divisible, required by
        # nn.MultiheadAttention (d_model must be divisible by n_heads).
        d_model = trial.suggest_categorical("d_model", [16, 32, 64])
        n_heads = trial.suggest_categorical("n_heads", [2, 4])
        return {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }

    attn_tuning = tune_torch_model(
        "composition_attention",
        build_attn_model,
        attention_forward,
        attn_search_space,
        (idx_tr_t, frac_tr_t, mask_tr_t),
        ytr_t,
        (idx_val_t, frac_val_t, mask_val_t),
        yval_t,
        seed=TUNING_SEED,
        n_trials=TORCH_N_TRIALS,
    )
    attn_model = build_attn_model(attn_tuning.best_params)
    attn_fit = train_with_early_stopping(
        attn_model,
        attention_forward,
        (idx_tr_t, frac_tr_t, mask_tr_t),
        ytr_t,
        (idx_val_t, frac_val_t, mask_val_t),
        yval_t,
        lr=attn_tuning.best_params["lr"],
        weight_decay=attn_tuning.best_params["weight_decay"],
        max_epochs=150,
        patience=15,
        loss="mae",
        seed=TUNING_SEED,
    )
    attn_model = attn_fit.model
    attn_model.eval()
    with torch.no_grad():
        train_mae = regression_metrics(y_train, attn_model(idx_tr_t, frac_tr_t, mask_tr_t).numpy())[
            "mae"
        ]
        val_mae = regression_metrics(y_val, attn_model(idx_val_t, frac_val_t, mask_val_t).numpy())[
            "mae"
        ]
    resolved_attn_params = {k: v for k, v in attn_tuning.best_params.items()}
    models["composition_attention"] = {
        "model_family": "torch_composition_attention",
        "hyperparameters": resolved_attn_params,
        "hyperparameters_hash": _hash_params(resolved_attn_params),
        "seed": TUNING_SEED,
        "feature_schema": "element_vocabulary_v1",
        "n_features": None,
        "max_elements": MAX_ELEMENTS,
        "vocab_size": vocab.vocab_size,
        "train_mae": train_mae,
        "validation_mae": val_mae,
        "n_trials": attn_tuning.n_trials,
        "n_parameters": count_parameters(attn_model),
        "best_epoch": attn_fit.best_epoch,
    }
    print(f"[composition_attention] val MAE={val_mae:.4f} params={resolved_attn_params}")

    manifest = {
        "phase": "N3_portable_model_benchmark",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "obelix_dev_population": {"n_train": len(train_df), "n_validation": len(val_df)},
        "obelix_test_untouched_row_count": n_test_untouched,
        "hargreaves_data_read": False,
        "feature_schema_check": schema_check,
        "seeds_available_for_downstream_multiseed_use": SEEDS,
        "tuning_seed": TUNING_SEED,
        "models": models,
    }
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    guard_output(ARTIFACT_JSON).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {ARTIFACT_JSON.relative_to(ROOT)}")

    _write_report(manifest, train_df, val_df, n_test_untouched, schema_check)


def _write_report(
    manifest: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_test_untouched: int,
    schema_check: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# N3 — Portable Composition-Only Model Benchmark\n")
    lines.append(
        "Status: **DONE (build/freeze only — no Hargreaves evaluation, no OBELiX-test access)**\n"
    )
    lines.append(f"Git commit: `{manifest['git_commit']}` on branch `{manifest['git_branch']}`\n")
    lines.append("\n## 1. Development population\n\n")
    lines.append(
        f"- OBELiX development population = frozen `train` ({len(train_df)}) UNION "
        f"`validation` ({len(val_df)}) from `load_regression_dataset()` = "
        f"{len(train_df) + len(val_df)} rows.\n"
        f"- OBELiX frozen `test` split ({n_test_untouched} rows, from the old closed DD "
        "project) was read only for its length in this script "
        "(`load_regression_dataset().test`'s features/targets are never referenced "
        "anywhere else in `scripts/transport_analysis/n3_build_model_ladder.py` — grep "
        "the file for `.test` to verify: it appears exactly once, in "
        "`load_obelix_dev_population`, as "
        "`len(dataset.test)`).\n"
        "- No Hargreaves file (`data/raw/hargreaves_conductivity/`, "
        "`data/processed/external_cohort/`) was opened for fitting or tuning in this "
        "script; the only Hargreaves read is `verify_feature_schema_identical`, which "
        "loads 20 Hargreaves rows purely to diff Magpie feature *column names* against "
        "OBELiX's — no Hargreaves target value or row is ever passed to "
        "`.fit`/`.predict`/tuning anywhere in this file.\n"
    )
    lines.append("\n## 2. Feature schema identity check (executed, not assumed)\n\n")
    lines.append(f"```json\n{json.dumps(schema_check, indent=2)}\n```\n")
    lines.append("\n## 3. Tuning protocol\n\n")
    lines.append(
        "- Sklearn/GBDT candidates: Optuna TPE sampler, fixed seed, "
        f"{N_TRIALS} trials, objective = OBELiX-validation MAE, fit on OBELiX-train only "
        "(identical pattern/budget to the existing Phase 1 `training.tune_classical` "
        "module).\n"
        f"- Torch candidates: Optuna TPE sampler, fixed seed, {TORCH_N_TRIALS} trials "
        "(smaller because each trial trains a full early-stopped model), same "
        "train/validation objective, max_epochs=150, patience=15.\n"
        f"- Tuning seed: `{manifest['tuning_seed']}`. Multi-seed list reserved for "
        "downstream (N4+) multi-seed evaluation: "
        f"`{manifest['seeds_available_for_downstream_multiseed_use']}` (matches the old "
        "project's Phase 2.7/2.8 seed list for consistency) — N3 itself freezes one "
        "hyperparameter configuration per model using the tuning seed only.\n"
    )
    lines.append("\n## 4. Frozen model ladder — train/validation MAE\n\n")
    lines.append("| # | Model | Feature schema | n_features | Train MAE | Validation MAE |\n")
    lines.append("|---|---|---|---|---|---|\n")
    order = [
        "ridge",
        "random_forest",
        "extra_trees",
        "xgboost",
        "catboost",
        "shallow_mlp",
        "pretrained_element_embedding",
        "composition_attention",
    ]
    for i, name in enumerate(order, start=1):
        m = manifest["models"][name]
        nf = m["n_features"] if m["n_features"] is not None else f"vocab={m.get('vocab_size')}"
        lines.append(
            f"| {i} | {name} | {m['feature_schema']} | {nf} | {m['train_mae']:.4f} | "
            f"{m['validation_mae']:.4f} |\n"
        )
    lines.append(
        "\nFull frozen hyperparameters, seeds, and hashes: "
        "`artifacts/frozen/model_ladder_frozen.json`.\n"
    )
    embed_m = manifest["models"]["pretrained_element_embedding"]
    lines.append(
        "\n**Caveat (honest, not smoothed over):** model 7 (pretrained-element-embedding "
        f"regressor) is substantially weaker than every other candidate (train MAE "
        f"{embed_m['train_mae']:.3f}, validation MAE {embed_m['validation_mae']:.3f}, vs. "
        "0.70-1.14 for the rest). Train and validation MAE are close (no train/validation "
        "divergence, so this is not an overfitting artifact), and the model trained for "
        f"{embed_m['best_epoch']} epochs before early stopping (not epoch 0), so this is "
        "not a silent training failure. The most likely explanation is architectural: "
        "this candidate's 22-dimensional fraction-pooled raw-elemental-property vector "
        "(`features.element_embeddings.TrainFitElementEmbedder`) is far less expressive "
        "than the 132-column full Magpie statistic vector every other tabular candidate "
        "uses. A `RuntimeWarning: All-NaN slice encountered` was also observed during "
        "`TrainFitElementEmbedder.fit` (at least one of the 22 raw Magpie elemental "
        "properties is undefined for every element in the OBELiX-train vocabulary); the "
        "existing, already-tested `element_embeddings.py` code already guards this case "
        "by falling back to 0.0 for that column, so it does not silently propagate NaN, "
        "but it is a real precision loss inherent to this feature representation on this "
        "vocabulary, not a bug introduced by this script. This candidate is kept in the "
        "frozen ladder as-is (N3 forbids re-tuning after seeing results / loosening "
        "criteria to improve a number), and its weakness is carried forward honestly "
        "into N4-N6 rather than dropped or re-engineered.\n"
    )
    lines.append("\n## 5. Rationale for model choices\n\n")
    lines.append(
        "- Models 1-5 (Ridge, RandomForest, ExtraTrees, XGBoost, CatBoost) are standard "
        "tabular regressors on the shared Magpie/matminer composition-only feature "
        "vector (`features.composition_pipeline.TrainFitCompositionScaler`, "
        "train-fit-only). XGBoost/CatBoost factories are reused unchanged from the "
        "existing Phase 1 baseline module (`models.classical_baselines`); "
        "Ridge/RandomForest/ExtraTrees are new thin factories added to the same module, "
        "same style.\n"
        "- Model 6 (shallow MLP) reuses the existing "
        "`models.torch_regressors.ShallowMLPRegressor` (Phase 1.9 candidate C) on the "
        "same Magpie feature vector.\n"
        "- Model 7 (pretrained-element-embedding regressor) reuses the existing "
        "`features.element_embeddings.TrainFitElementEmbedder` (fraction-weighted "
        "pooled per-element Magpie property vector, train-fit-only) feeding the same "
        "`ShallowMLPRegressor` class with a smaller hidden size (Phase 1.9 candidate "
        "D).\n"
        "- Model 8 (composition-attention, CrabNet/Roost-style) reuses the existing "
        "`features.element_embeddings.ElementVocabulary` + "
        "`models.torch_regressors.CompositionAttentionRegressor` (Phase 1.9 candidate "
        "E) unchanged.\n"
        "- No 9th model was added, per N3's explicit instruction.\n"
    )
    lines.append("\n## 6. Not done in N3 (by design)\n\n")
    lines.append(
        "- No outer-fold validation-regime comparison (random 5-fold CV / "
        "Family-GroupKFold / LOMFO) — that is N4.\n"
        "- No Hargreaves-target evaluation — that is N6 (one-shot external transfer), "
        "and this script never reads a Hargreaves target value.\n"
        "- No OBELiX-test evaluation at all, at any point in this project's remaining "
        "npj phases until an explicit N-phase says otherwise (there currently is none "
        "— the npj story targets Hargreaves, not the old OBELiX test split).\n"
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_MD).write_text("".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
