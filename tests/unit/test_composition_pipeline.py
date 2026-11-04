import numpy as np
import pandas as pd

from sse_transfer.features.composition_features import COMPOSITION_FEATURE_NAMES
from sse_transfer.features.composition_pipeline import TrainFitCompositionScaler


def _sample_compositions() -> pd.Series:
    return pd.Series(["Li7La3Zr2O12", "LiFePO4", "Li3PO4", "Li2S", "Li4Ti5O12"])


def test_fit_transform_returns_expected_shape_and_columns() -> None:
    scaler = TrainFitCompositionScaler()
    out = scaler.fit_transform(_sample_compositions())
    assert out.shape == (5, len(COMPOSITION_FEATURE_NAMES))
    assert list(out.columns) == list(COMPOSITION_FEATURE_NAMES)


def test_transform_before_fit_raises() -> None:
    scaler = TrainFitCompositionScaler()
    try:
        scaler.transform(_sample_compositions())
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_transform_has_no_nans_after_train_only_imputation() -> None:
    scaler = TrainFitCompositionScaler().fit(_sample_compositions())
    out = scaler.transform(_sample_compositions())
    assert not out.isna().any().any()


def test_unseen_composition_uses_train_fit_statistics_not_refit() -> None:
    train = _sample_compositions()
    scaler = TrainFitCompositionScaler().fit(train)
    fingerprint_before = scaler.fingerprint()
    # Transform on a composition never seen at fit time must not mutate the
    # fitted statistics (train-only-fit invariant).
    scaler.transform(pd.Series(["Na3PS4"]))
    assert scaler.fingerprint() == fingerprint_before


def test_fingerprint_is_deterministic_for_same_training_data() -> None:
    train = _sample_compositions()
    fp_a = TrainFitCompositionScaler().fit(train).fingerprint()
    fp_b = TrainFitCompositionScaler().fit(train).fingerprint()
    assert fp_a == fp_b


def test_fingerprint_differs_for_different_training_data() -> None:
    fp_a = TrainFitCompositionScaler().fit(_sample_compositions()).fingerprint()
    fp_b = TrainFitCompositionScaler().fit(pd.Series(["NaCl", "KBr"])).fingerprint()
    assert fp_a != fp_b


def test_standardized_output_has_zero_mean_on_training_data() -> None:
    train = _sample_compositions()
    scaler = TrainFitCompositionScaler().fit(train)
    out = scaler.transform(train)
    means = out.mean(numeric_only=True).to_numpy()
    assert np.allclose(means, 0.0, atol=1e-8)
