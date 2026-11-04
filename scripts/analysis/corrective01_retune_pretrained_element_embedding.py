#!/usr/bin/env python3
"""Corrective 01, step 1: re-tune `pretrained_element_embedding`'s global
hyperparameters for the final external-transfer fit, using the now-corrected
`TrainFitElementEmbedder` representation (see
reports/remediation/specs/corrective01_pretrained_element_embedding_bugfix.md).

This is a minimal re-run of exactly the N3 code path for this ONE model
(`scripts/transport_analysis/n3_build_model_ladder.py` lines ~293-364):
same train(359)/validation(90) split, same search space, same
TORCH_N_TRIALS=20, same tuning seed 20260921. It does NOT touch
`artifacts/frozen/model_ladder_frozen.json` -- instead it writes a full copy of
that frozen ladder with only this one model's entry replaced, so every
downstream per-model-fragment script can be pointed at the corrected file
via a path override without any source-code change to those scripts.

Run with: python scripts/analysis/corrective01_retune_pretrained_element_embedding.py
"""

from __future__ import annotations

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
from sse_transfer.features.element_embeddings import TrainFitElementEmbedder  # noqa: E402
from sse_transfer.models.torch_regressors import (  # noqa: E402
    ShallowMLPRegressor,
    count_parameters,
    mlp_forward,
    to_tensor,
    train_with_early_stopping,
)
from sse_transfer.training.tune_ladder import TORCH_N_TRIALS, tune_torch_model  # noqa: E402
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

FROZEN_LADDER_PATH = ROOT / "artifacts" / "frozen" / "model_ladder_frozen.json"
OUT_DIR = ROOT / "artifacts" / "rerun"
OUT_PATH = OUT_DIR / "model_ladder_corrected.json"
TUNING_SEED = 20260921


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def _hash_params(params: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def load_obelix_dev_population() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    dataset = load_regression_dataset()
    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)
    raw_by_id = raw.set_index("ID")

    def attach(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[["ID", "y"]].copy()
        out["true_composition"] = raw_by_id.loc[out["ID"], "True Composition"].to_numpy()
        return out.reset_index(drop=True)

    return attach(dataset.train), attach(dataset.validation), len(dataset.test)


def main() -> None:
    train_df, val_df, n_test_untouched = load_obelix_dev_population()
    print(
        f"train={len(train_df)} validation={len(val_df)} "
        f"test(untouched,len-only)={n_test_untouched}"
    )

    y_train = train_df["y"].to_numpy()
    y_val = val_df["y"].to_numpy()
    ytr_t = to_tensor(y_train)
    yval_t = to_tensor(y_val)

    embedder = TrainFitElementEmbedder().fit(train_df["true_composition"])
    Etr = to_tensor(embedder.transform(train_df["true_composition"]))
    Eval = to_tensor(embedder.transform(val_df["true_composition"]))
    print(f"corrected feature matrix: train={tuple(Etr.shape)} val={tuple(Eval.shape)}")

    def embed_search_space(trial: optuna.Trial) -> dict[str, Any]:
        return {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 32, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }

    def build_embed_model(params: dict[str, Any]) -> torch.nn.Module:
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
    corrected_entry = {
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
        "corrective_note": (
            "Re-tuned under corrective01 after fixing the pymatgen Element "
            "bug in features/element_embeddings.py; search space, trial "
            "budget (TORCH_N_TRIALS=20), and tuning seed (20260921) are "
            "identical to the original N3 run. See "
            "reports/remediation/specs/corrective01_pretrained_element_embedding_bugfix.md."
        ),
    }

    frozen_ladder = json.loads(FROZEN_LADDER_PATH.read_text(encoding="utf-8"))
    old_entry = frozen_ladder["models"]["pretrained_element_embedding"]
    corrected_ladder = json.loads(json.dumps(frozen_ladder))  # deep copy
    corrected_ladder["models"]["pretrained_element_embedding"] = corrected_entry
    corrected_ladder["corrective01_note"] = (
        "This is a corrective, non-authoritative copy of "
        "artifacts/frozen/model_ladder_frozen.json with ONLY the "
        "pretrained_element_embedding entry replaced. The other 7 models' "
        "entries below are byte-identical to the historical frozen ladder "
        "and are reproduced here only so downstream per-model-fragment "
        "scripts can read a single consistent file when run with "
        "--model pretrained_element_embedding. Do not use this file for "
        "any other model."
    )
    corrected_ladder["git_commit_at_corrective_rerun"] = _git("rev-parse", "HEAD")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guard_output(OUT_PATH).write_text(
        json.dumps(corrected_ladder, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")

    print("\n=== OLD (broken-representation) vs CORRECTED hyperparameters ===")
    print("OLD:", json.dumps(old_entry, indent=2, default=str))
    print("CORRECTED:", json.dumps(corrected_entry, indent=2, default=str))


if __name__ == "__main__":
    main()
