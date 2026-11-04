"""Frozen, leakage-safe OBELiX regression dataset for Phase 1 classical
baselines (CLAUDE.md Phase 1; DATA_CONTRACT.md dataset role D).

Loads the Phase 0C frozen split (data/processed/v1/obelix_index.json,
hash-verified against its sidecar) and recomputes the exact/censored label
status directly from the raw CSV (never trusts a cached censoring index
blindly), then assembles the deterministic raw feature frame for
exact-labeled rows only. Reads only from data/raw/obelix/ and
data/processed/v1/obelix_index.json; never reads pretraining raw data
(data/raw/bvel_percolation, data/raw/bvse_hops, data/raw/dft_neb), so no
pretraining-leakage-excluded record can reappear here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from sse_transfer.data.obelix_loader import (
    LABEL_STATUS_LEFT_CENSORED,
    LABEL_STATUS_RIGHT_CENSORED,
    censoring_bound,
    label_status,
    load_obelix,
)
from sse_transfer.features.build_features import build_raw_feature_frame

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed" / "v1"
SPLIT_PATH = PROCESSED_DIR / "obelix_index.json"

VALID_SPLITS: tuple[str, ...] = ("train", "validation", "test")


class SplitIntegrityError(RuntimeError):
    """Raised when the frozen OBELiX split or target derivation fails an
    integrity check. Never caught silently: a Phase 1 run must stop here."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_split_hash(split_path: Path = SPLIT_PATH) -> str:
    """Raise SplitIntegrityError unless `split_path` matches its frozen
    `.sha256.json` sidecar exactly. Returns the verified hash."""
    if not split_path.exists():
        raise SplitIntegrityError(
            f"Missing frozen split file {split_path}. Run scripts/build_processed_v1.py first."
        )
    sidecar_path = split_path.with_suffix(split_path.suffix + ".sha256.json")
    if not sidecar_path.exists():
        raise SplitIntegrityError(f"Missing split hash sidecar {sidecar_path}.")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    actual = _sha256(split_path)
    if actual != sidecar["sha256"]:
        raise SplitIntegrityError(
            f"Frozen split hash mismatch for {split_path}: "
            f"expected {sidecar['sha256']}, got {actual}. The frozen split must "
            "never change; investigate before proceeding."
        )
    return actual


def load_frozen_split(split_path: Path = SPLIT_PATH) -> dict[str, Any]:
    verify_split_hash(split_path)
    return cast(dict[str, Any], json.loads(split_path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class RegressionDataset:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: tuple[str, ...]
    split_hash: str
    n_censored_excluded: int


def load_regression_dataset() -> RegressionDataset:
    """Build the Phase 1 primary-regression train/validation/test frames.

    Each returned frame has columns ["ID", "y", "obelix_split", *feature
    columns]. `y = log10(sigma_RT / S cm^-1)` (DATA_CONTRACT.md canonical
    label), computed only for exact-labeled rows; censored rows are
    excluded from the primary regression per DATA_CONTRACT.md "Censoring".
    """
    split = load_frozen_split()
    split_of: dict[str, str] = {}
    for name in VALID_SPLITS:
        for rid in split["splits"][name]:
            split_of[rid] = name

    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)

    ids_in_split = set(split_of)
    ids_in_raw = set(raw["ID"])
    if ids_in_split != ids_in_raw:
        raise SplitIntegrityError(
            "Frozen split ids do not match raw OBELiX ids "
            f"(split-only: {sorted(ids_in_split - ids_in_raw)[:5]}, "
            f"raw-only: {sorted(ids_in_raw - ids_in_split)[:5]})."
        )

    conductivity = raw["Ionic conductivity (S cm-1)"]
    status = label_status(conductivity)
    bound = censoring_bound(conductivity)
    exact_mask = (status == "EXACT").to_numpy()
    n_censored_excluded = int((~exact_mask).sum())

    exact_bounds = bound.to_numpy()[exact_mask]
    if not np.isfinite(exact_bounds).all():
        raise SplitIntegrityError("Non-finite exact conductivity value(s) found.")
    if (exact_bounds <= 0).any():
        raise SplitIntegrityError("Non-positive exact conductivity value(s); cannot take log10.")

    y = np.full(len(raw), np.nan, dtype=float)
    y[exact_mask] = np.log10(exact_bounds)
    if not np.isfinite(y[exact_mask]).all():
        raise SplitIntegrityError("Non-finite target value(s) after log10 transform.")

    features = build_raw_feature_frame(raw)
    frame = features.loc[exact_mask].copy()
    frame.insert(0, "ID", raw.loc[exact_mask, "ID"].to_numpy())
    frame.insert(1, "y", y[exact_mask])
    frame.insert(2, "obelix_split", [split_of[rid] for rid in frame["ID"]])
    frame = frame.reset_index(drop=True)

    train = frame[frame["obelix_split"] == "train"].reset_index(drop=True)
    validation = frame[frame["obelix_split"] == "validation"].reset_index(drop=True)
    test = frame[frame["obelix_split"] == "test"].reset_index(drop=True)

    ids_train, ids_val, ids_test = set(train["ID"]), set(validation["ID"]), set(test["ID"])
    if not (
        ids_train.isdisjoint(ids_val)
        and ids_train.isdisjoint(ids_test)
        and ids_val.isdisjoint(ids_test)
    ):
        raise SplitIntegrityError("Train/validation/test id overlap detected.")
    if len(train) + len(validation) + len(test) != len(frame):
        raise SplitIntegrityError("Some exact-labeled rows were not assigned to any split.")

    feature_columns = tuple(c for c in frame.columns if c not in {"ID", "y", "obelix_split"})

    return RegressionDataset(
        train=train,
        validation=validation,
        test=test,
        feature_columns=feature_columns,
        split_hash=verify_split_hash(),
        n_censored_excluded=n_censored_excluded,
    )


def load_censored_train_frame() -> pd.DataFrame:
    """Train-split rows with a LEFT_/RIGHT_CENSORED conductivity label
    (DATA_CONTRACT.md "Censoring") -- excluded from `load_regression_dataset`
    entirely, but usable as inequality-consistent training signal (Phase
    2.6A). Never includes a validation/test-split row (frozen validation/
    test targets are untouched by this function and by every caller of it)
    and never includes a MISSING-status row (undefined bound semantics).

    Returned frame has the same column shape as `RegressionDataset.train`
    (`["ID", "y", "obelix_split", *feature_columns]`) plus two sidecar
    columns consumed only by the censor-aware training loss, never treated
    as a model feature:
      - `censor_direction`: "LEFT_CENSORED" or "RIGHT_CENSORED".
      - `censor_bound_log10`: `log10(bound / S cm^-1)` -- the *inequality*
        bound, never a point label. `y` is NaN for every row here (DATA_
        CONTRACT.md "NEVER replace censored values with the bound ...").
    """
    split = load_frozen_split()
    split_of: dict[str, str] = {}
    for name in VALID_SPLITS:
        for rid in split["splits"][name]:
            split_of[rid] = name

    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)

    conductivity = raw["Ionic conductivity (S cm-1)"]
    status = label_status(conductivity)
    bound = censoring_bound(conductivity)

    is_train = raw["ID"].map(split_of).eq("train")
    is_censored_status = status.isin([LABEL_STATUS_LEFT_CENSORED, LABEL_STATUS_RIGHT_CENSORED])
    mask = (is_train & is_censored_status).to_numpy()

    censored_bounds = bound.to_numpy()[mask]
    if not np.isfinite(censored_bounds).all():
        raise SplitIntegrityError("Non-finite censored conductivity bound(s) found.")
    if (censored_bounds <= 0).any():
        raise SplitIntegrityError("Non-positive censored conductivity bound(s); cannot take log10.")

    features = build_raw_feature_frame(raw)
    frame = features.loc[mask].copy()
    frame.insert(0, "ID", raw.loc[mask, "ID"].to_numpy())
    frame.insert(1, "y", np.nan)
    frame.insert(2, "obelix_split", "train")
    frame["censor_direction"] = status.to_numpy()[mask]
    frame["censor_bound_log10"] = np.log10(censored_bounds)
    frame = frame.reset_index(drop=True)

    if (
        not frame["censor_direction"]
        .isin([LABEL_STATUS_LEFT_CENSORED, LABEL_STATUS_RIGHT_CENSORED])
        .all()
    ):
        raise SplitIntegrityError("load_censored_train_frame: non-censored row leaked into output.")
    if not np.isfinite(frame["censor_bound_log10"].to_numpy()).all():
        raise SplitIntegrityError("Non-finite log10 censor bound after transform.")

    return frame
