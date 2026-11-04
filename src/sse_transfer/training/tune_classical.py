"""Optuna hyperparameter tuning for the Phase 1 XGBoost/CatBoost baselines.

EXPERIMENT_PROTOCOL.md "Hyperparameter tuning": tune on train+validation
only (objective = validation MAE), never on the final test set. The search
budget (N_TRIALS) is fixed and documented here rather than tuned itself,
and is identical for both competing models so comparisons stay fair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd

from sse_transfer.evaluation.metrics import regression_metrics
from sse_transfer.models.classical_baselines import build_catboost, build_xgboost

#: Fixed, documented search budget (CLAUDE.md Phase 1 "Optuna": "Fixed
#: documented search budget"). Identical for XGBoost and CatBoost.
N_TRIALS = 40

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class TuningResult:
    model_name: str
    best_params: dict[str, Any]
    best_val_mae: float
    n_trials: int


def _xgboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _catboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 100, 800),
        "depth": trial.suggest_int("depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "bootstrap_type": "Bernoulli",
    }


_SEARCH_SPACES = {"xgboost": _xgboost_search_space, "catboost": _catboost_search_space}
_BUILDERS = {"xgboost": build_xgboost, "catboost": build_catboost}


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    seed: int,
    n_trials: int = N_TRIALS,
) -> TuningResult:
    if model_name not in _SEARCH_SPACES:
        raise ValueError(
            f"Unknown model_name {model_name!r}; expected one of {sorted(_SEARCH_SPACES)}"
        )

    search_space = _SEARCH_SPACES[model_name]
    builder = _BUILDERS[model_name]

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        model = builder(params, seed=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return regression_metrics(np.asarray(y_val), np.asarray(preds))["mae"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, study_name=f"{model_name}_phase1"
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return TuningResult(
        model_name=model_name,
        best_params=study.best_params,
        best_val_mae=study.best_value,
        n_trials=n_trials,
    )
