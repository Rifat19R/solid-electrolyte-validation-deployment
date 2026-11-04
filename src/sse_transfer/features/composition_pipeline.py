"""Composition-only tabular feature pipeline shared identically between
OBELiX and Hargreaves for the npj cross-dataset extension (N3).

Wraps `composition_features.composition_features` (Magpie/matminer, already
leakage-safe/deterministic per that module's docstring) with a train-only
fitted median-impute + standardize step, following the same fit/transform
convention as `features.preprocessing.TrainFitPreprocessor` and
`features.element_embeddings.TrainFitElementEmbedder`: every statistic is
computed once from a training composition series and reused unchanged on
any other composition series (OBELiX validation, or the entirely separate
Hargreaves external cohort). Never re-fit on Hargreaves or on OBELiX
validation/test (EXPERIMENT_PROTOCOL.md "Never tune on the final test set";
DATA_CONTRACT.md "Train-only preprocessing").
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import pandas as pd

from sse_transfer.features.composition_features import (
    COMPOSITION_FEATURE_NAMES,
    composition_features,
)

FEATURE_SCHEMA_NAME = "magpie_composition_only_v1"


@dataclass
class TrainFitCompositionScaler:
    medians_: dict[str, float] = field(default_factory=dict)
    means_: dict[str, float] = field(default_factory=dict)
    stds_: dict[str, float] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, train_compositions: pd.Series) -> TrainFitCompositionScaler:
        raw = composition_features(train_compositions)
        medians = raw.median(numeric_only=True)
        imputed = raw.fillna(medians)
        means = imputed.mean()
        stds = imputed.std().fillna(1.0)
        stds = stds.where(stds.abs() > 1e-8, 1.0)
        self.medians_ = {str(k): float(v) for k, v in medians.items()}
        self.means_ = {str(k): float(v) for k, v in means.items()}
        self.stds_ = {str(k): float(v) for k, v in stds.items()}
        self.fitted_ = True
        return self

    def transform(self, compositions: pd.Series) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("TrainFitCompositionScaler.transform called before fit()")
        raw = composition_features(compositions)
        raw = raw[list(COMPOSITION_FEATURE_NAMES)]
        raw = raw.fillna(pd.Series(self.medians_))
        scaled = (raw - pd.Series(self.means_)) / pd.Series(self.stds_)
        return scaled

    def fit_transform(self, train_compositions: pd.Series) -> pd.DataFrame:
        self.fit(train_compositions)
        return self.transform(train_compositions)

    def fingerprint(self) -> str:
        if not self.fitted_:
            raise RuntimeError("TrainFitCompositionScaler.fingerprint called before fit()")
        payload = json.dumps(
            {
                "schema_name": FEATURE_SCHEMA_NAME,
                "feature_names": list(COMPOSITION_FEATURE_NAMES),
                "medians": self.medians_,
                "means": self.means_,
                "stds": self.stds_,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
