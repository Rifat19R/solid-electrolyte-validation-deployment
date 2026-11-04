import warnings

import numpy as np
import pandas as pd
import pytest

from sse_transfer.features.element_embeddings import (
    ELEMENT_EMBEDDING_DIM,
    ElementVocabulary,
    TrainFitElementEmbedder,
    _raw_element_vector,
    composition_fractions,
)


def test_composition_fractions_sum_to_one() -> None:
    fractions = composition_fractions("Li7La3Zr2O12")
    assert fractions is not None
    assert fractions.keys() == {"Li", "La", "Zr", "O"}
    assert np.isclose(sum(fractions.values()), 1.0)


def test_composition_fractions_preserves_disorder_ratio() -> None:
    fractions = composition_fractions("Li0.5Al0.5O")
    assert fractions is not None
    assert np.isclose(fractions["Li"], fractions["Al"])


def test_composition_fractions_unparseable_returns_none() -> None:
    assert composition_fractions("not-a-formula!!!") is None


def test_element_embedder_transform_before_fit_raises() -> None:
    embedder = TrainFitElementEmbedder()
    with pytest.raises(RuntimeError):
        embedder.transform(pd.Series(["Li2O"]))


def test_element_embedder_output_shape() -> None:
    train = pd.Series(["Li2O", "LiFePO4", "Li7La3Zr2O12"])
    embedder = TrainFitElementEmbedder().fit(train)
    out = embedder.transform(train)
    assert out.shape == (3, ELEMENT_EMBEDDING_DIM)
    assert np.isfinite(out).all()


def test_element_embedder_is_standardized_on_train_vocabulary() -> None:
    train = pd.Series(["Li2O", "LiFePO4", "Li7La3Zr2O12", "Na3PS4"])
    embedder = TrainFitElementEmbedder().fit(train)
    # Per-element standardized vectors (pre-pooling) should average to ~0
    # across the fitted vocabulary.
    stacked = np.stack(list(embedder._standardized_table.values()))
    assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-8)


def test_element_embedder_unseen_element_falls_back_to_train_mean() -> None:
    train = pd.Series(["Li2O"])
    embedder = TrainFitElementEmbedder().fit(train)
    out_unseen = embedder.transform(pd.Series(["XeF2"]))  # Xe, F unseen in train
    assert out_unseen.shape == (1, ELEMENT_EMBEDDING_DIM)
    assert np.allclose(out_unseen[0], 0.0)


def test_element_embedder_unparseable_formula_yields_zero_row() -> None:
    train = pd.Series(["Li2O"])
    embedder = TrainFitElementEmbedder().fit(train)
    out = embedder.transform(pd.Series(["garbage!!!"]))
    assert np.allclose(out[0], 0.0)


def test_element_vocabulary_reserves_pad_and_unk() -> None:
    vocab = ElementVocabulary().fit(pd.Series(["Li2O", "LiFePO4"]))
    assert ElementVocabulary.PAD_INDEX == 0
    assert ElementVocabulary.UNK_INDEX == 1
    assert all(idx >= 2 for idx in vocab.symbol_to_index_.values())
    assert vocab.vocab_size == len(vocab.symbol_to_index_) + 2


def test_element_vocabulary_encode_before_fit_raises() -> None:
    vocab = ElementVocabulary()
    with pytest.raises(RuntimeError):
        vocab.encode(pd.Series(["Li2O"]), max_elements=3)


def test_element_vocabulary_encode_shapes_and_mask() -> None:
    vocab = ElementVocabulary().fit(pd.Series(["Li2O", "LiFePO4"]))
    idx, frac, mask = vocab.encode(pd.Series(["Li2O"]), max_elements=4)
    assert idx.shape == frac.shape == mask.shape == (1, 4)
    # Li2O has 2 distinct elements -> 2 valid slots, 2 padded.
    assert mask[0].sum() == 2
    assert np.isclose(frac[0][~mask[0]].sum(), 1.0)


def test_element_vocabulary_encode_unseen_element_maps_to_unk() -> None:
    vocab = ElementVocabulary().fit(pd.Series(["Li2O"]))
    idx, _, mask = vocab.encode(pd.Series(["NaCl"]), max_elements=4)
    valid_idx = idx[0][~mask[0]]
    assert all(i == ElementVocabulary.UNK_INDEX for i in valid_idx)


def test_raw_element_vector_is_not_all_nan() -> None:
    """Regression test: `_raw_element_vector` previously passed a raw
    element symbol (str) to `get_elemental_property`, which requires an
    object with a `.symbol` attribute. Every property/element pair raised
    `AttributeError`, swallowed by a blanket `except Exception`, so every
    raw Magpie property value was unconditionally NaN -- collapsing the
    entire pooled elemental-property embedding to a constant zero vector
    for every composition (see element_embeddings.py `_raw_element_vector`
    docstring)."""
    vector = _raw_element_vector("Li")
    assert not np.isnan(vector).all()
    assert np.isfinite(vector).any()


def test_element_embedder_fit_does_not_warn() -> None:
    train = pd.Series(["Li2O", "LiFePO4", "Li7La3Zr2O12", "Na3PS4"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TrainFitElementEmbedder().fit(train)


def test_element_embedder_standardized_vectors_differ_across_elements() -> None:
    """Regression test for the all-NaN bug: with real elemental data, the
    pre-pooling standardized vectors for distinct elements must not all
    collapse to the same (zero) vector."""
    train = pd.Series(["Li2O", "LiFePO4", "Li7La3Zr2O12", "Na3PS4"])
    embedder = TrainFitElementEmbedder().fit(train)
    vectors = list(embedder._standardized_table.values())
    assert not all(np.allclose(v, vectors[0]) for v in vectors[1:])


def test_element_embedder_transform_distinguishes_compositions() -> None:
    """Regression test for the all-NaN bug: chemically distinct
    compositions must not pool down to an identical embedding row."""
    train = pd.Series(["Li2O", "LiFePO4", "Li7La3Zr2O12", "Na3PS4"])
    embedder = TrainFitElementEmbedder().fit(train)
    out = embedder.transform(pd.Series(["Li2O", "Na3PS4"]))
    assert not np.allclose(out[0], out[1])


def test_element_embedder_handles_genuinely_all_nan_property_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A property that is genuinely undefined for every element in a
    (possibly small) training vocabulary must not raise `np.nanmedian`'s
    "All-NaN slice encountered" `RuntimeWarning`, and must fall back to a
    deterministic, finite imputed value."""
    import sse_transfer.features.element_embeddings as element_embeddings_module

    real_vector = element_embeddings_module._raw_element_vector

    def _vector_with_first_column_nan(symbol: str) -> np.ndarray:
        vector = real_vector(symbol).copy()
        vector[0] = float("nan")
        return vector

    monkeypatch.setattr(
        element_embeddings_module, "_raw_element_vector", _vector_with_first_column_nan
    )
    train = pd.Series(["Li2O", "Na3PS4"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        embedder = element_embeddings_module.TrainFitElementEmbedder().fit(train)
    assert np.isfinite(embedder.mean_).all()
    assert np.isfinite(embedder.std_).all()
    assert embedder.mean_[0] == 0.0  # deterministic all-NaN-column fallback
    out = embedder.transform(train)
    assert np.isfinite(out).all()


def test_element_vocabulary_encode_truncates_to_highest_fraction_elements() -> None:
    vocab = ElementVocabulary().fit(pd.Series(["LiFePO4"]))
    idx, frac, mask = vocab.encode(pd.Series(["LiFePO4"]), max_elements=2)
    # LiFePO4 has 4 distinct elements; only the top 2 by fraction are kept,
    # filling both slots (0 padded).
    assert mask[0].sum() == 0
    assert np.isclose(frac[0][~mask[0]].sum(), 1.0)
