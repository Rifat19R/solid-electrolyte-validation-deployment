from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sse_transfer.models.ladder import MODEL_NAMES, fit_predict_model

_TRAIN_COMPOSITIONS = pd.Series(
    [
        "Li2O",
        "LiFePO4",
        "Na2O",
        "Li7La3Zr2O12",
        "LiCoO2",
        "Li3PS4",
        "LiNbO3",
        "Li2S",
        "LiAlO2",
        "Li4Ti5O12",
        "LiMn2O4",
        "Li1.3Al0.3Ti1.7(PO4)3",
    ]
)
_TRAIN_Y = np.array([-3.1, -4.2, -2.9, -3.5, -5.0, -3.8, -4.5, -3.0, -4.9, -3.3, -4.1, -3.6])
_EVAL_COMPOSITIONS = pd.Series(["Li2ZrO3", "LiTaO3", "Li3N", "LiBO2"])

_SKLEARN_PARAMS: dict[str, dict[str, object]] = {
    "ridge": {"alpha": 1.0},
    "random_forest": {
        "n_estimators": 10,
        "max_depth": 3,
        "min_samples_leaf": 1,
        "max_features": 0.5,
    },
    "extra_trees": {
        "n_estimators": 10,
        "max_depth": 3,
        "min_samples_leaf": 1,
        "max_features": 0.5,
    },
    "xgboost": {
        "n_estimators": 10,
        "max_depth": 2,
        "learning_rate": 0.3,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1,
        "reg_lambda": 1.0,
    },
    "catboost": {
        "iterations": 10,
        "depth": 2,
        "learning_rate": 0.3,
        "l2_leaf_reg": 1.0,
        "subsample": 1.0,
        "bootstrap_type": "Bernoulli",
    },
}

_TORCH_PARAMS: dict[str, dict[str, object]] = {
    "shallow_mlp": {"hidden_dims": [8], "dropout": 0.0, "lr": 1e-2, "weight_decay": 0.0},
    "pretrained_element_embedding": {
        "hidden_dims": [4],
        "dropout": 0.0,
        "lr": 1e-2,
        "weight_decay": 0.0,
    },
    "composition_attention": {
        "d_model": 8,
        "n_heads": 2,
        "n_layers": 1,
        "dropout": 0.0,
        "lr": 1e-2,
        "weight_decay": 0.0,
    },
}


def test_model_names_has_exactly_eight_models() -> None:
    assert len(MODEL_NAMES) == 8


@pytest.mark.parametrize("model_name", list(_SKLEARN_PARAMS))
def test_sklearn_models_fit_predict_finite_and_shaped(model_name: str) -> None:
    preds = fit_predict_model(
        model_name,
        _SKLEARN_PARAMS[model_name],
        _TRAIN_COMPOSITIONS,
        _TRAIN_Y,
        _EVAL_COMPOSITIONS,
        seed=0,
    )
    assert preds.shape == (len(_EVAL_COMPOSITIONS),)
    assert np.isfinite(preds).all()


@pytest.mark.parametrize("model_name", list(_TORCH_PARAMS))
def test_torch_models_fit_predict_finite_and_shaped(model_name: str) -> None:
    preds = fit_predict_model(
        model_name,
        _TORCH_PARAMS[model_name],
        _TRAIN_COMPOSITIONS,
        _TRAIN_Y,
        _EVAL_COMPOSITIONS,
        seed=0,
    )
    assert preds.shape == (len(_EVAL_COMPOSITIONS),)
    assert np.isfinite(preds).all()


def test_unknown_model_name_raises() -> None:
    with pytest.raises(ValueError):
        fit_predict_model(
            "not_a_model", {}, _TRAIN_COMPOSITIONS, _TRAIN_Y, _EVAL_COMPOSITIONS, seed=0
        )
