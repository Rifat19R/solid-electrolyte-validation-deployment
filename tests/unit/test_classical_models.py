import joblib
import numpy as np
import pandas as pd

from sse_transfer.models.classical_baselines import (
    build_catboost,
    build_extra_trees,
    build_random_forest,
    build_ridge,
    build_xgboost,
    naive_mean_baseline,
    naive_median_baseline,
)


def test_naive_mean_baseline_predicts_train_mean() -> None:
    y_train = np.array([1.0, 2.0, 3.0, 10.0])
    model = naive_mean_baseline().fit(np.zeros((4, 1)), y_train)
    preds = model.predict(np.zeros((3, 1)))
    assert np.allclose(preds, y_train.mean())


def test_naive_median_baseline_predicts_train_median() -> None:
    y_train = np.array([1.0, 2.0, 3.0, 10.0])
    model = naive_median_baseline().fit(np.zeros((4, 1)), y_train)
    preds = model.predict(np.zeros((3, 1)))
    assert np.allclose(preds, np.median(y_train))


def _toy_frame(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
    y = X["f1"] * 2.0 - X["f2"] + rng.normal(scale=0.01, size=n)
    return X, y.to_numpy()


def test_xgboost_save_load_consistency(tmp_path) -> None:
    X, y = _toy_frame(40, seed=0)
    model = build_xgboost({"n_estimators": 20, "max_depth": 3}, seed=17)
    model.fit(X, y)
    preds_before = model.predict(X)

    path = tmp_path / "xgb_model.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    preds_after = reloaded.predict(X)

    assert np.allclose(preds_before, preds_after)


def test_catboost_save_load_consistency(tmp_path) -> None:
    X, y = _toy_frame(40, seed=1)
    model = build_catboost({"iterations": 20, "depth": 3}, seed=29)
    model.fit(X, y)
    preds_before = model.predict(X)

    path = tmp_path / "catboost_model.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    preds_after = reloaded.predict(X)

    assert np.allclose(preds_before, preds_after)


def test_xgboost_is_deterministic_for_fixed_seed() -> None:
    X, y = _toy_frame(30, seed=2)
    model_a = build_xgboost({"n_estimators": 15, "max_depth": 2}, seed=101)
    model_a.fit(X, y)
    model_b = build_xgboost({"n_estimators": 15, "max_depth": 2}, seed=101)
    model_b.fit(X, y)
    assert np.allclose(model_a.predict(X), model_b.predict(X))


def test_ridge_fits_and_predicts_finite_values() -> None:
    X, y = _toy_frame(40, seed=3)
    model = build_ridge({"alpha": 1.0}, seed=17)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    assert len(preds) == len(y)


def test_ridge_is_deterministic_for_fixed_seed() -> None:
    X, y = _toy_frame(30, seed=4)
    model_a = build_ridge({"alpha": 0.5}, seed=29).fit(X, y)
    model_b = build_ridge({"alpha": 0.5}, seed=29).fit(X, y)
    assert np.allclose(model_a.predict(X), model_b.predict(X))


def test_random_forest_fits_and_predicts_finite_values() -> None:
    X, y = _toy_frame(40, seed=5)
    model = build_random_forest({"n_estimators": 10, "max_depth": 3}, seed=43)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    assert len(preds) == len(y)


def test_random_forest_is_deterministic_for_fixed_seed() -> None:
    X, y = _toy_frame(30, seed=6)
    model_a = build_random_forest({"n_estimators": 10, "max_depth": 3}, seed=71).fit(X, y)
    model_b = build_random_forest({"n_estimators": 10, "max_depth": 3}, seed=71).fit(X, y)
    assert np.allclose(model_a.predict(X), model_b.predict(X))


def test_extra_trees_fits_and_predicts_finite_values() -> None:
    X, y = _toy_frame(40, seed=7)
    model = build_extra_trees({"n_estimators": 10, "max_depth": 3}, seed=101)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    assert len(preds) == len(y)


def test_extra_trees_is_deterministic_for_fixed_seed() -> None:
    X, y = _toy_frame(30, seed=8)
    model_a = build_extra_trees({"n_estimators": 10, "max_depth": 3}, seed=17).fit(X, y)
    model_b = build_extra_trees({"n_estimators": 10, "max_depth": 3}, seed=17).fit(X, y)
    assert np.allclose(model_a.predict(X), model_b.predict(X))
