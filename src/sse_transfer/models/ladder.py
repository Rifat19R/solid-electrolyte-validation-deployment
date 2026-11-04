"""Fold-level fit/predict dispatcher for the N3-frozen 8-model ladder.

N4 (internal validation regimes) needs to refit each of the 8 frozen
models on many different (train-partition, eval-partition) splits of the
OBELiX development population -- always with the exact hyperparameters N3
already froze (`artifacts/frozen/model_ladder_frozen.json`), never re-tuning.
This module factors the per-model-family construction logic that
`scripts/transport_analysis/n3_build_model_ladder.py` wrote inline (one
code block per model) into a single reusable `fit_predict_model`
dispatcher, so N4's CV loop does not duplicate it.

Every feature encoder (`TrainFitCompositionScaler`, `TrainFitElementEmbedder`,
`ElementVocabulary`) is fit fresh inside this function from the given
`train_df` only, and applied unchanged to `eval_df` -- never fit on
`eval_df`, matching this project's train-only-preprocessing convention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sse_transfer.evaluation.internal_validation import inner_early_stop_split
from sse_transfer.features.composition_pipeline import TrainFitCompositionScaler
from sse_transfer.features.element_embeddings import ElementVocabulary, TrainFitElementEmbedder
from sse_transfer.models.classical_baselines import (
    build_catboost,
    build_extra_trees,
    build_random_forest,
    build_ridge,
    build_xgboost,
)
from sse_transfer.models.torch_regressors import (
    CompositionAttentionRegressor,
    ShallowMLPRegressor,
    attention_forward,
    mlp_forward,
    to_tensor,
    train_with_early_stopping,
)

ROOT = Path(__file__).resolve().parents[3]
FROZEN_LADDER_PATH = ROOT / "artifacts" / "frozen" / "model_ladder_frozen.json"

MAX_ELEMENTS = 10  # ElementVocabulary.encode's per-composition cap, same as N3.
MAX_EPOCHS = 150
PATIENCE = 15
INNER_STOP_FRACTION = 0.2

_SKLEARN_BUILDERS = {
    "ridge": build_ridge,
    "random_forest": build_random_forest,
    "extra_trees": build_extra_trees,
    "xgboost": build_xgboost,
    "catboost": build_catboost,
}

MODEL_NAMES: tuple[str, ...] = (
    "ridge",
    "random_forest",
    "extra_trees",
    "xgboost",
    "catboost",
    "shallow_mlp",
    "pretrained_element_embedding",
    "composition_attention",
)


def load_frozen_ladder() -> dict[str, Any]:
    if not FROZEN_LADDER_PATH.exists():
        raise FileNotFoundError(
            f"Missing frozen N3 model ladder at {FROZEN_LADDER_PATH}; run "
            "scripts/transport_analysis/n3_build_model_ladder.py first."
        )
    payload: dict[str, Any] = json.loads(FROZEN_LADDER_PATH.read_text(encoding="utf-8"))
    return payload


def fit_predict_model(
    model_name: str,
    hyperparameters: dict[str, Any],
    train_compositions: pd.Series,
    train_y: np.ndarray,
    eval_compositions: pd.Series,
    *,
    seed: int,
) -> np.ndarray:
    """Fit `model_name` (with its N3-frozen `hyperparameters`) on
    `(train_compositions, train_y)` and return predictions for
    `eval_compositions`. `eval_compositions` never influences any fitted
    statistic (feature scaler, element vocabulary, or early-stopping
    decision) -- for the torch candidates, the early-stopping monitor split
    is carved out of `train_compositions` alone via
    `evaluation.internal_validation.inner_early_stop_split`."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model_name {model_name!r}; expected one of {MODEL_NAMES}")

    if model_name in _SKLEARN_BUILDERS:
        scaler = TrainFitCompositionScaler().fit(train_compositions)
        X_train = scaler.transform(train_compositions)
        X_eval = scaler.transform(eval_compositions)
        model = _SKLEARN_BUILDERS[model_name](hyperparameters, seed=seed)
        model.fit(X_train, train_y)
        preds: np.ndarray = np.asarray(model.predict(X_eval), dtype=float)
        return preds

    train_compositions = train_compositions.reset_index(drop=True)
    n_train = len(train_compositions)
    inner_train_idx, inner_stop_idx = inner_early_stop_split(
        n_train, stop_fraction=INNER_STOP_FRACTION, seed=seed
    )
    inner_train_comp = train_compositions.iloc[inner_train_idx].reset_index(drop=True)
    inner_stop_comp = train_compositions.iloc[inner_stop_idx].reset_index(drop=True)
    y_inner_train = np.asarray(train_y)[inner_train_idx]
    y_inner_stop = np.asarray(train_y)[inner_stop_idx]

    if model_name == "shallow_mlp":
        scaler = TrainFitCompositionScaler().fit(inner_train_comp)
        Xtr = to_tensor(scaler.transform(inner_train_comp).to_numpy())
        Xstop = to_tensor(scaler.transform(inner_stop_comp).to_numpy())
        Xeval = to_tensor(scaler.transform(eval_compositions).to_numpy())
        torch.manual_seed(seed)
        hidden_dims = tuple(hyperparameters["hidden_dims"])
        model = ShallowMLPRegressor(
            in_dim=Xtr.shape[1], hidden_dims=hidden_dims, dropout=hyperparameters["dropout"]
        )
        fit_result = train_with_early_stopping(
            model,
            mlp_forward,
            (Xtr,),
            to_tensor(y_inner_train),
            (Xstop,),
            to_tensor(y_inner_stop),
            lr=hyperparameters["lr"],
            weight_decay=hyperparameters["weight_decay"],
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            loss="mae",
            seed=seed,
        )
        model = fit_result.model
        model.eval()
        with torch.no_grad():
            preds = model(Xeval).numpy()
        return np.asarray(preds, dtype=float)

    if model_name == "pretrained_element_embedding":
        embedder = TrainFitElementEmbedder().fit(inner_train_comp)
        Etr = to_tensor(embedder.transform(inner_train_comp))
        Estop = to_tensor(embedder.transform(inner_stop_comp))
        Eeval = to_tensor(embedder.transform(eval_compositions))
        torch.manual_seed(seed)
        model = ShallowMLPRegressor(
            in_dim=Etr.shape[1],
            hidden_dims=tuple(hyperparameters["hidden_dims"]),
            dropout=hyperparameters["dropout"],
        )
        fit_result = train_with_early_stopping(
            model,
            mlp_forward,
            (Etr,),
            to_tensor(y_inner_train),
            (Estop,),
            to_tensor(y_inner_stop),
            lr=hyperparameters["lr"],
            weight_decay=hyperparameters["weight_decay"],
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            loss="mae",
            seed=seed,
        )
        model = fit_result.model
        model.eval()
        with torch.no_grad():
            preds = model(Eeval).numpy()
        return np.asarray(preds, dtype=float)

    if model_name == "composition_attention":
        vocab = ElementVocabulary().fit(inner_train_comp)
        idx_tr, frac_tr, mask_tr = vocab.encode(inner_train_comp, max_elements=MAX_ELEMENTS)
        idx_stop, frac_stop, mask_stop = vocab.encode(inner_stop_comp, max_elements=MAX_ELEMENTS)
        idx_ev, frac_ev, mask_ev = vocab.encode(eval_compositions, max_elements=MAX_ELEMENTS)
        torch.manual_seed(seed)
        model = CompositionAttentionRegressor(
            vocab_size=vocab.vocab_size,
            d_model=hyperparameters["d_model"],
            n_heads=hyperparameters["n_heads"],
            n_layers=hyperparameters["n_layers"],
            dropout=hyperparameters["dropout"],
        )
        fit_result = train_with_early_stopping(
            model,
            attention_forward,
            (to_tensor(idx_tr, torch.long), to_tensor(frac_tr), to_tensor(mask_tr, torch.bool)),
            to_tensor(y_inner_train),
            (
                to_tensor(idx_stop, torch.long),
                to_tensor(frac_stop),
                to_tensor(mask_stop, torch.bool),
            ),
            to_tensor(y_inner_stop),
            lr=hyperparameters["lr"],
            weight_decay=hyperparameters["weight_decay"],
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            loss="mae",
            seed=seed,
        )
        model = fit_result.model
        model.eval()
        with torch.no_grad():
            preds = model(
                to_tensor(idx_ev, torch.long), to_tensor(frac_ev), to_tensor(mask_ev, torch.bool)
            ).numpy()
        return np.asarray(preds, dtype=float)

    raise AssertionError(f"unreachable: unhandled model_name {model_name!r}")
