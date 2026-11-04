"""N3 nested/internal hyperparameter tuning for the portable model
ladder (reports/cross_dataset_study/n3_portable_model_benchmark.md).

Extends `training.tune_classical`'s Optuna pattern (train-fit, validation-
objective, fixed TPE sampler seed, fixed documented trial budget) to the
remaining N3 candidates: Ridge/RandomForest/ExtraTrees (sklearn) and the
three torch candidates (shallow MLP, pretrained-element-embedding
regressor, composition-attention). Every `tune_*` function here reads only
the `X_train`/`y_train` (fit) and `X_val`/`y_val` (objective) arrays a
caller passes in; none of them ever opens a file, so a caller that (by
construction) never passes OBELiX-test or Hargreaves rows here cannot leak
them (EXPERIMENT_PROTOCOL.md "Never tune on the final test set").
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn

from sse_transfer.evaluation.metrics import regression_metrics
from sse_transfer.models.classical_baselines import (
    build_extra_trees,
    build_random_forest,
    build_ridge,
)
from sse_transfer.models.torch_regressors import ForwardFn, train_with_early_stopping

optuna.logging.set_verbosity(optuna.logging.WARNING)

#: Identical to `training.tune_classical.N_TRIALS`: EXPERIMENT_PROTOCOL.md
#: "The same search budget should be reasonably comparable across competing
#: neural models" -- kept equal across every N3 candidate, classical or
#: torch, rather than tuned per model.
N_TRIALS = 40
#: Each torch trial trains a full early-stopped model, so this is kept
#: smaller than N_TRIALS -- but still fixed/documented and identical across
#: all 3 torch candidates.
TORCH_N_TRIALS = 20
MAX_EPOCHS = 150
PATIENCE = 15


@dataclass(frozen=True)
class TuningResult:
    model_name: str
    best_params: dict[str, Any]
    best_val_mae: float
    n_trials: int


def _ridge_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}


def _random_forest_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


def _extra_trees_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


_SKLEARN_SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "ridge": _ridge_search_space,
    "random_forest": _random_forest_search_space,
    "extra_trees": _extra_trees_search_space,
}
_SKLEARN_BUILDERS = {
    "ridge": build_ridge,
    "random_forest": build_random_forest,
    "extra_trees": build_extra_trees,
}


def tune_sklearn_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int,
    n_trials: int = N_TRIALS,
) -> TuningResult:
    if model_name not in _SKLEARN_SEARCH_SPACES:
        raise ValueError(
            f"Unknown model_name {model_name!r}; expected one of {sorted(_SKLEARN_SEARCH_SPACES)}"
        )
    search_space = _SKLEARN_SEARCH_SPACES[model_name]
    builder = _SKLEARN_BUILDERS[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        model = builder(params, seed=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return regression_metrics(np.asarray(y_val), np.asarray(preds))["mae"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, study_name=f"{model_name}_n3"
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return TuningResult(
        model_name=model_name,
        best_params=study.best_params,
        best_val_mae=study.best_value,
        n_trials=n_trials,
    )


BuildTorchModel = Callable[[dict[str, Any]], nn.Module]
SearchSpace = Callable[[optuna.Trial], dict[str, Any]]


def tune_torch_model(
    model_name: str,
    build_model: BuildTorchModel,
    forward_fn: ForwardFn,
    search_space: SearchSpace,
    train_inputs: tuple[torch.Tensor, ...],
    y_train: torch.Tensor,
    val_inputs: tuple[torch.Tensor, ...],
    y_val: torch.Tensor,
    *,
    seed: int,
    n_trials: int = TORCH_N_TRIALS,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> TuningResult:
    """Optuna over a torch model's architecture/optimization hyperparameters;
    each trial trains with early stopping on `val_inputs`/`y_val` (never
    OBELiX test, never Hargreaves -- the caller controls what's passed)."""

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        model = build_model(params)
        result = train_with_early_stopping(
            model,
            forward_fn,
            train_inputs,
            y_train,
            val_inputs,
            y_val,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            max_epochs=max_epochs,
            patience=patience,
            loss="mae",
            seed=seed,
        )
        if result.best_epoch < 0:
            return float("inf")
        return result.val_mae_history[result.best_epoch]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, study_name=f"{model_name}_n3"
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return TuningResult(
        model_name=model_name,
        best_params=study.best_params,
        best_val_mae=study.best_value,
        n_trials=n_trials,
    )
