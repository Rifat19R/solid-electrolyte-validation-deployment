"""Assembles the full deterministic raw feature frame for Phase 1's
classical conductivity baselines, before train-only preprocessing
(imputation / one-hot encoding, see `preprocessing.py`) is applied.
"""

from __future__ import annotations

import pandas as pd

from sse_transfer.features.composition_features import composition_features
from sse_transfer.features.metadata_features import (
    FAMILY_COLUMN,
    NUMERIC_METADATA_COLUMNS,
    metadata_features,
)

COMPOSITION_COLUMN = "True Composition"

#: The only OBELiX columns Phase 1 features are ever computed from. No IDs
#: (ID, ICSD ID, Laskowski ID, Liion ID, DOI, Cif ID, ...) and no
#: target-derived columns (Ionic conductivity, IC (Total), IC (Bulk)).
RAW_INPUT_COLUMNS: tuple[str, ...] = (COMPOSITION_COLUMN, *NUMERIC_METADATA_COLUMNS, FAMILY_COLUMN)


def build_raw_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """`raw` is the OBELiX all/train/test frame; must contain RAW_INPUT_COLUMNS.

    Returned frame shares `raw`'s index (row-wise, deterministic, no fitting).
    """
    comp = composition_features(raw[COMPOSITION_COLUMN])
    meta = metadata_features(raw)
    return pd.concat([comp, meta], axis=1)
