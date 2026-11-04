"""N15 analysis 7 -- fixed [20, 30] degC temperature-window sensitivity
(post-review sensitivity analysis, NOT part of the N5 pretest lock).

Composes `sse_transfer.data.hargreaves_cohort`'s existing pure functions
UNCHANGED (imported, never edited) with a narrower explicit RT window,
rather than touching that module's own frozen `RT_WINDOW_LOW_C`/
`RT_WINDOW_HIGH_C` constants or its `build_cohorts()` entry point (both are
referenced by `artifacts/frozen/external_transfer_lock.json`'s
`cohort_build_code_sha256` and must not change).
"""

from __future__ import annotations

import pandas as pd

from sse_transfer.data.hargreaves_cohort import (
    MATCH_NONE,
    aggregate_by_formula,
    build_obelix_formula_doi_index,
    classify_overlap,
    rt_window_mask,
    safe_reduced_formula,
    valid_target_mask,
)

N15_TEMPERATURE_WINDOW_VERSION = "1.0"


def build_strict_cohort_custom_window(
    hargreaves_raw: pd.DataFrame,
    obelix_raw: pd.DataFrame,
    *,
    low_c: float,
    high_c: float,
) -> pd.DataFrame:
    """Rebuild N2's STRICT (overlap_classification == NO_MATCH) aggregated
    cohort under an arbitrary [low_c, high_c] window, using
    `hargreaves_cohort`'s exact classification/aggregation rules. Returns the
    same shape `aggregate_by_formula` returns (one row per surviving
    reduced_formula, with `log_target_rt_median`, `n_source_rows`,
    `n_unique_sources`).
    """
    obelix_index = build_obelix_formula_doi_index(obelix_raw)

    ledger = hargreaves_raw.copy()
    ledger["reduced_formula"] = ledger["composition"].apply(safe_reduced_formula)
    ledger["in_window"] = rt_window_mask(ledger["temperature"], low=low_c, high=high_c)
    ledger["valid_target"] = valid_target_mask(ledger["target"]) & ledger["log_target"].notna()
    ledger["overlap_classification"] = [
        classify_overlap(rf, doi, obelix_index)
        for rf, doi in zip(ledger["reduced_formula"], ledger["source"], strict=False)
    ]

    base_valid = ledger["in_window"] & ledger["valid_target"] & ledger["reduced_formula"].notna()
    strict_valid = base_valid & (ledger["overlap_classification"] == MATCH_NONE)

    return aggregate_by_formula(ledger.loc[strict_valid])
