"""Model factories for Phase 1's classical conductivity baselines
(CLAUDE.md Phase 1 "Models"; EXPERIMENT_PROTOCOL.md "Required model
ladder" M0-M2). No neural networks or GNNs here.
"""

from __future__ import annotations

from typing import Any

from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def naive_mean_baseline() -> DummyRegressor:
    """M0: predict the train-set mean of y for every row."""
    return DummyRegressor(strategy="mean")


def naive_median_baseline() -> DummyRegressor:
    """M0: predict the train-set median of y for every row."""
    return DummyRegressor(strategy="median")


def build_xgboost(params: dict[str, Any], *, seed: int) -> XGBRegressor:
    """M1."""
    return XGBRegressor(**params, random_state=seed, n_jobs=1, objective="reg:squarederror")


def build_catboost(params: dict[str, Any], *, seed: int) -> CatBoostRegressor:
    """M2."""
    return CatBoostRegressor(**params, random_seed=seed, verbose=False, allow_writing_files=False)


def build_ridge(params: dict[str, Any], *, seed: int) -> Ridge:
    """N3 portable-benchmark model #1 (regularized linear)."""
    return Ridge(**params, random_state=seed)


def build_random_forest(params: dict[str, Any], *, seed: int) -> RandomForestRegressor:
    """N3 portable-benchmark model #2."""
    return RandomForestRegressor(**params, random_state=seed, n_jobs=1)


def build_extra_trees(params: dict[str, Any], *, seed: int) -> ExtraTreesRegressor:
    """N3 portable-benchmark model #3."""
    return ExtraTreesRegressor(**params, random_state=seed, n_jobs=1)
