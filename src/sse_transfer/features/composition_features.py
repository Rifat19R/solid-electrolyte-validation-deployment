"""Deterministic composition-based features for the OBELiX conductivity task.

Row-wise Magpie elemental-property statistics (Ward et al. 2016) computed
from parsed pymatgen `Composition` objects. Every value is a pure function
of a single row's composition string: there is no fitting step, so these
features cannot leak across train/validation/test splits and are identical
regardless of which other rows are present (CLAUDE.md Phase 1 "Features":
deterministic, leakage-safe, no test-fitted preprocessing).
"""

from __future__ import annotations

import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

_FEATURIZER = ElementProperty.from_preset("magpie")
COMPOSITION_FEATURE_NAMES: tuple[str, ...] = tuple(_FEATURIZER.feature_labels())


def is_disordered_composition(formula: object, tol: float = 1e-3) -> bool:
    """True if any element amount in `formula` is non-integer beyond `tol`.

    A fractional-occupancy signature (DATA_CONTRACT.md "Disorder"): OBELiX's
    "True Composition" reports formula-unit-scaled site occupancies, so a
    partially occupied site shows up as a non-integer element count (e.g.
    "Li5.46Hf12.13P18.0O72.0"). An unparseable formula is reported as not
    disordered rather than raising, matching `composition_features`'s
    fail-soft convention.
    """
    try:
        comp = Composition(str(formula))
    except Exception:
        return False
    return any(abs(amt - round(amt)) > tol for amt in comp.values())


def composition_features(compositions: pd.Series) -> pd.DataFrame:
    """Magpie composition features for a series of raw formula strings.

    `compositions` should be the OBELiX "True Composition" column. A value
    that fails to parse yields an all-NaN row rather than raising, so one
    malformed formula does not abort the whole batch; callers must impute
    NaNs explicitly using train-only statistics.
    """
    rows: list[list[float]] = []
    for value in compositions:
        try:
            comp = Composition(str(value))
            rows.append(list(_FEATURIZER.featurize(comp)))
        except Exception:
            rows.append([float("nan")] * len(COMPOSITION_FEATURE_NAMES))
    return pd.DataFrame(rows, columns=list(COMPOSITION_FEATURE_NAMES), index=compositions.index)
