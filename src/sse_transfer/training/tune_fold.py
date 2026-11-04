"""Per-fold hyperparameter tuning for the nested internal-validation
remediation (`scripts/transport_analysis/n4b_nested_internal_validation.py`).

N3 (`scripts/transport_analysis/n3_build_model_ladder.py`) tuned each of the 8 portable
model families ONCE against the OBELiX train(359)/validation(90) split and
froze the result; N4 then reused those frozen hyperparameters across every
outer-CV fold built from train UNION validation (449 rows). Because the
90 "validation" rows that shaped the frozen hyperparameters later served as
outer-test rows in most folds, N4's regime comparison carried an internal-
CV optimism bias (post-lock methodological finding, not a new experiment).

This module re-tunes each model from scratch on whatever
(inner-train, inner-validation) pair a caller passes in -- exactly N3's
search spaces, builders, `N_TRIALS`/`TORCH_N_TRIALS` budgets, and tuning
seed convention, just parameterized so `n4b` can call it once per
(outer fold, model) using an inner split carved out of that fold's own
outer-train partition only. No OBELiX-test or Hargreaves row is ever
touched here (same invariant as `tune_ladder.py`, `n3`, `n4`).

Returns hyperparameters in the exact same shape N3 froze into
`artifacts/frozen/model_ladder_frozen.json`, so
`sse_transfer.models.ladder.fit_predict_model` can consume the output
unchanged.
"""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd
import torch

from sse_transfer.features.composition_pipeline import TrainFitCompositionScaler
from sse_transfer.features.element_embeddings import ElementVocabulary, TrainFitElementEmbedder
from sse_transfer.models.torch_regressors import (
    CompositionAttentionRegressor,
    ShallowMLPRegressor,
    attention_forward,
    mlp_forward,
    to_tensor,
)
from sse_transfer.training.tune_classical import tune_model as tune_gbdt
from sse_transfer.training.tune_ladder import (
    N_TRIALS,
    TORCH_N_TRIALS,
    tune_sklearn_model,
    tune_torch_model,
)

MAX_ELEMENTS = 10  # same cap as N3/N4/ladder.py.

_SKLEARN_MODEL_NAMES = {"ridge", "random_forest", "extra_trees"}
_GBDT_MODEL_NAMES = {"xgboost", "catboost"}


def tune_hyperparameters(
    model_name: str,
    train_compositions: pd.Series,
    train_y: Any,
    val_compositions: pd.Series,
    val_y: Any,
    *,
    seed: int,
    n_trials: int = N_TRIALS,
    torch_n_trials: int = TORCH_N_TRIALS,
) -> dict[str, Any]:
    """Tunes `model_name` on (train_compositions, train_y) against
    (val_compositions, val_y) only, using N3's exact search space/budget
    for that model family. Every feature encoder (Magpie scaler / pooled
    element embedder / element vocabulary) is fit on `train_compositions`
    only, matching N3/`fit_predict_model`'s train-only-preprocessing
    convention -- `val_compositions` only ever supplies the tuning
    objective, never a fitted statistic."""
    if model_name in _SKLEARN_MODEL_NAMES:
        scaler = TrainFitCompositionScaler().fit(train_compositions)
        x_train = scaler.transform(train_compositions)
        x_val = scaler.transform(val_compositions)
        tuning = tune_sklearn_model(
            model_name, x_train, train_y, x_val, val_y, seed=seed, n_trials=n_trials
        )
        return dict(tuning.best_params)

    if model_name in _GBDT_MODEL_NAMES:
        scaler = TrainFitCompositionScaler().fit(train_compositions)
        x_train = scaler.transform(train_compositions)
        x_val = scaler.transform(val_compositions)
        gbdt_tuning = tune_gbdt(
            model_name, x_train, train_y, x_val, val_y, seed=seed, n_trials=n_trials
        )
        return dict(gbdt_tuning.best_params)

    if model_name == "shallow_mlp":
        scaler = TrainFitCompositionScaler().fit(train_compositions)
        x_train = scaler.transform(train_compositions)
        x_val = scaler.transform(val_compositions)
        xtr_t, xval_t = to_tensor(x_train.to_numpy()), to_tensor(x_val.to_numpy())
        ytr_t, yval_t = to_tensor(train_y), to_tensor(val_y)
        in_dim = x_train.shape[1]

        def search_space(trial: optuna.Trial) -> dict[str, Any]:
            return {
                "n_layers": trial.suggest_int("n_layers", 1, 3),
                "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            }

        def build(params: dict[str, Any]) -> torch.nn.Module:
            torch.manual_seed(seed)
            hidden_dims = tuple([params["hidden_dim"]] * params["n_layers"])
            return ShallowMLPRegressor(
                in_dim=in_dim, hidden_dims=hidden_dims, dropout=params["dropout"]
            )

        tuning = tune_torch_model(
            "shallow_mlp",
            build,
            mlp_forward,
            search_space,
            (xtr_t,),
            ytr_t,
            (xval_t,),
            yval_t,
            seed=seed,
            n_trials=torch_n_trials,
        )
        return {
            "hidden_dims": [tuning.best_params["hidden_dim"]] * tuning.best_params["n_layers"],
            "dropout": tuning.best_params["dropout"],
            "lr": tuning.best_params["lr"],
            "weight_decay": tuning.best_params["weight_decay"],
        }

    if model_name == "pretrained_element_embedding":
        embedder = TrainFitElementEmbedder().fit(train_compositions)
        etr = to_tensor(embedder.transform(train_compositions))
        eval_ = to_tensor(embedder.transform(val_compositions))
        ytr_t, yval_t = to_tensor(train_y), to_tensor(val_y)
        in_dim = etr.shape[1]

        def search_space(trial: optuna.Trial) -> dict[str, Any]:
            return {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            }

        def build(params: dict[str, Any]) -> torch.nn.Module:
            torch.manual_seed(seed)
            return ShallowMLPRegressor(
                in_dim=in_dim, hidden_dims=(params["hidden_dim"],), dropout=params["dropout"]
            )

        tuning = tune_torch_model(
            "pretrained_element_embedding",
            build,
            mlp_forward,
            search_space,
            (etr,),
            ytr_t,
            (eval_,),
            yval_t,
            seed=seed,
            n_trials=torch_n_trials,
        )
        return {
            "hidden_dims": [tuning.best_params["hidden_dim"]],
            "dropout": tuning.best_params["dropout"],
            "lr": tuning.best_params["lr"],
            "weight_decay": tuning.best_params["weight_decay"],
        }

    if model_name == "composition_attention":
        vocab = ElementVocabulary().fit(train_compositions)
        idx_tr, frac_tr, mask_tr = vocab.encode(train_compositions, max_elements=MAX_ELEMENTS)
        idx_val, frac_val, mask_val = vocab.encode(val_compositions, max_elements=MAX_ELEMENTS)
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
        ytr_t, yval_t = to_tensor(train_y), to_tensor(val_y)
        vocab_size = vocab.vocab_size

        def search_space(trial: optuna.Trial) -> dict[str, Any]:
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

        def build(params: dict[str, Any]) -> torch.nn.Module:
            torch.manual_seed(seed)
            return CompositionAttentionRegressor(
                vocab_size=vocab_size,
                d_model=params["d_model"],
                n_heads=params["n_heads"],
                n_layers=params["n_layers"],
                dropout=params["dropout"],
            )

        tuning = tune_torch_model(
            "composition_attention",
            build,
            attention_forward,
            search_space,
            (idx_tr_t, frac_tr_t, mask_tr_t),
            ytr_t,
            (idx_val_t, frac_val_t, mask_val_t),
            yval_t,
            seed=seed,
            n_trials=torch_n_trials,
        )
        return dict(tuning.best_params)

    raise ValueError(f"Unknown model_name {model_name!r}")
