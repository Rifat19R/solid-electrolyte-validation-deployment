"""Stoichiometrically-pooled, train-fit-only elemental property embeddings.

Phase 1.9 candidate D ("stoichiometric pretrained element embeddings +
shallow regressor") needs a per-element embedding that is distinct from the
composition-level Magpie statistics `composition_features.py` already
computes (min/max/range/mean/avg_dev/mode across elements). This module
reuses the same deterministic Magpie elemental property table (Ward et al.
2016, via the matminer dependency already pinned in pyproject.toml -- no new
external download or provenance entry needed) but pools the *raw per-element*
22-property vectors by fractional stoichiometric weight instead, matching
DATA_CONTRACT.md's preferred site representation
`h_i = sum_j occupancy_ij * embedding(element_j)` and preserving disorder
(OBELiX "True Composition" already reports formula-unit-scaled, i.e.
occupancy-scaled, element amounts).

Standardization/imputation statistics are fit from the training-split
vocabulary of elements only (EXPERIMENT_PROTOCOL.md "Never tune on the final
test set" / no leakage of validation/test-only elements into fit statistics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition, Element

_FEATURIZER = ElementProperty.from_preset("magpie")
ELEMENT_PROPERTY_NAMES: tuple[str, ...] = tuple(_FEATURIZER.features)
ELEMENT_EMBEDDING_DIM: int = len(ELEMENT_PROPERTY_NAMES)


def composition_fractions(formula: object) -> dict[str, float] | None:
    """Element symbol -> stoichiometric fraction of `formula`, normalized to
    sum to 1. Uses the raw (possibly non-integer, disorder-scaled) amounts
    from pymatgen `Composition`, so fractional occupancy is preserved rather
    than rounded. Returns None for an unparseable or empty formula (fail-soft,
    matching `composition_features.composition_features`)."""
    try:
        comp = Composition(str(formula))
        total = float(sum(comp.values()))
        if not np.isfinite(total) or total <= 0:
            return None
        return {el.symbol: float(amt) / total for el, amt in comp.items()}
    except Exception:
        return None


def _raw_element_vector(symbol: str) -> np.ndarray:
    """22-dim raw Magpie property vector for one element symbol; NaN where
    the property is undefined for that element (e.g. GSmagmom for some
    elements), left for the caller to impute.

    `get_elemental_property` requires a pymatgen `Element` (it reads
    `elem.symbol` internally), not a raw string -- passing `symbol` directly
    raised `AttributeError` for every property/element pair, silently
    swallowed by the `except Exception` below, which made every raw
    property value NaN unconditionally (see regression test
    `test_raw_element_vector_is_not_all_nan`)."""
    element = Element(symbol)
    values: list[float] = []
    for prop in ELEMENT_PROPERTY_NAMES:
        try:
            values.append(float(_FEATURIZER.data_source.get_elemental_property(element, prop)))
        except Exception:
            values.append(float("nan"))
    return np.array(values, dtype=float)


class TrainFitElementEmbedder:
    """Fits standardization (mean/std) and NaN-imputation (median) statistics
    for the raw per-element Magpie property table from a training set's
    element vocabulary, then pools per-composition embeddings by
    stoichiometric fraction. Unseen (train-vocabulary) elements at transform
    time fall back to the train-standardized zero vector (i.e. the train
    mean) rather than raising."""

    def __init__(self) -> None:
        self.vocab_: tuple[str, ...] = ()
        self.mean_: np.ndarray = np.zeros(ELEMENT_EMBEDDING_DIM)
        self.std_: np.ndarray = np.ones(ELEMENT_EMBEDDING_DIM)
        self._standardized_table: dict[str, np.ndarray] = {}
        self.fitted_: bool = False

    def fit(self, train_compositions: pd.Series) -> TrainFitElementEmbedder:
        elements: set[str] = set()
        for formula in train_compositions:
            fractions = composition_fractions(formula)
            if fractions:
                elements.update(fractions.keys())
        vocab = sorted(elements)

        if vocab:
            raw_table = np.stack([_raw_element_vector(el) for el in vocab])
            # A property can still be legitimately undefined for every
            # element actually present in a (possibly small) training
            # vocabulary (e.g. GSmagmom for a vocabulary of only noble
            # gases). `np.nanmedian` warns ("All-NaN slice encountered") if
            # asked to reduce such a column, so only reduce columns with at
            # least one finite training observation; an all-NaN column
            # falls back to 0.0 directly. The fallback value is immaterial:
            # a column constant across the whole vocabulary gets std<1e-8
            # below, so every element's standardized value for that column
            # is 0 regardless of which constant was imputed.
            has_finite = np.isfinite(raw_table).any(axis=0)
            col_median = np.zeros(raw_table.shape[1])
            if has_finite.any():
                col_median[has_finite] = np.nanmedian(raw_table[:, has_finite], axis=0)
            imputed = np.where(np.isnan(raw_table), col_median, raw_table)
            mean = imputed.mean(axis=0)
            std = imputed.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
        else:
            imputed = np.zeros((0, ELEMENT_EMBEDDING_DIM))
            mean = np.zeros(ELEMENT_EMBEDDING_DIM)
            std = np.ones(ELEMENT_EMBEDDING_DIM)

        self.vocab_ = tuple(vocab)
        self.mean_ = mean
        self.std_ = std
        self._standardized_table = {el: (imputed[i] - mean) / std for i, el in enumerate(vocab)}
        self.fitted_ = True
        return self

    def transform(self, compositions: pd.Series) -> np.ndarray:
        """Fraction-weighted mean of standardized per-element vectors, one
        row per composition. Shape (len(compositions), ELEMENT_EMBEDDING_DIM)."""
        if not self.fitted_:
            raise RuntimeError("TrainFitElementEmbedder.transform called before fit()")
        out = np.zeros((len(compositions), ELEMENT_EMBEDDING_DIM), dtype=float)
        for i, formula in enumerate(compositions):
            fractions = composition_fractions(formula)
            if not fractions:
                continue  # all-zero (= train-mean) fallback row
            pooled = np.zeros(ELEMENT_EMBEDDING_DIM)
            for el, frac in fractions.items():
                pooled += frac * self._standardized_table.get(el, np.zeros(ELEMENT_EMBEDDING_DIM))
            out[i] = pooled
        return out


class ElementVocabulary:
    """Train-fit element-symbol -> integer-index mapping for the composition-
    attention model (candidate E). Index 0 is reserved for padding, index 1
    for an explicit UNKNOWN element (any element absent from the training
    vocabulary, e.g. a validation-only element)."""

    PAD_INDEX = 0
    UNK_INDEX = 1

    def __init__(self) -> None:
        self.symbol_to_index_: dict[str, int] = {}
        self.fitted_: bool = False

    def fit(self, train_compositions: pd.Series) -> ElementVocabulary:
        elements: set[str] = set()
        for formula in train_compositions:
            fractions = composition_fractions(formula)
            if fractions:
                elements.update(fractions.keys())
        self.symbol_to_index_ = {el: i + 2 for i, el in enumerate(sorted(elements))}
        self.fitted_ = True
        return self

    @property
    def vocab_size(self) -> int:
        """Total embedding-table size, including the PAD and UNK slots."""
        if not self.fitted_:
            raise RuntimeError("ElementVocabulary.vocab_size accessed before fit()")
        return len(self.symbol_to_index_) + 2

    def encode(
        self, compositions: pd.Series, *, max_elements: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (element_index, fraction, pad_mask), each shape
        (len(compositions), max_elements). `pad_mask` is True at padded
        (invalid) positions, matching `torch.nn.TransformerEncoder`'s
        `src_key_padding_mask` convention. A composition with more than
        `max_elements` distinct elements keeps only its `max_elements`
        highest-fraction elements (renormalized) rather than raising."""
        if not self.fitted_:
            raise RuntimeError("ElementVocabulary.encode called before fit()")
        n = len(compositions)
        idx = np.zeros((n, max_elements), dtype=np.int64)
        frac = np.zeros((n, max_elements), dtype=np.float32)
        mask = np.ones((n, max_elements), dtype=bool)
        for i, formula in enumerate(compositions):
            fractions = composition_fractions(formula)
            if not fractions:
                continue
            items = sorted(fractions.items(), key=lambda kv: -kv[1])[:max_elements]
            total = sum(f for _, f in items)
            for j, (el, f) in enumerate(items):
                idx[i, j] = self.symbol_to_index_.get(el, self.UNK_INDEX)
                frac[i, j] = f / total if total > 0 else 0.0
                mask[i, j] = False
        return idx, frac, mask
