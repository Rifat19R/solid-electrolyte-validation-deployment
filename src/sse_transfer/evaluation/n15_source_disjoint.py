"""N15 analysis 1 -- fully source-disjoint external cohort (post-review
sensitivity analysis, NOT part of the N5 pretest lock).

Turns N2b's audit (reports/cross_dataset_study/n2b_source_overlap_audit.md: 51/204 strict-
cohort compositions share a literature source DOI with OBELiX) into an
actual composition-level exclusion, recomputed from raw metadata only.

Reuses N6's already-frozen strict-cohort predictions (no re-inference) and
never reads data/raw/obelix/all.csv's DOI column through
`hargreaves_cohort.build_obelix_formula_doi_index` (which does not split
OBELiX's pipe-delimited multi-DOI cells) -- this module implements its own,
stricter, '|'-splitting DOI normalization, documented as a methodological
refinement over N2/N2b's per-composition check, not a retroactive change to
any frozen N2/N2b number.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sse_transfer.evaluation.metrics import regression_metrics

N15_SOURCE_DISJOINT_VERSION = "1.0"

RETAINED_EXCLUSION_REASON = "RETAINED"


def normalize_doi(doi: object) -> str | None:
    """Lowercase/stripped DOI, or None for a missing value.

    Intentionally identical single-DOI normalization to
    `sse_transfer.data.hargreaves_cohort._normalize_doi` (reimplemented here
    rather than importing a private symbol across modules); the '|'-splitting
    step is applied separately by `split_doi_field`.
    """
    if doi is None or (isinstance(doi, float) and pd.isna(doi)):
        return None
    text = str(doi).strip().lower()
    if not text or text == "nan":
        return None
    return text


def split_doi_field(doi: object) -> set[str]:
    """Every individual normalized DOI in a (possibly '|'-delimited) cell.

    OBELiX stores multiple co-citing DOIs in a single pipe-delimited string
    for a minority of rows (verified: 6/599 rows); a naive whole-string
    normalization (as `hargreaves_cohort._normalize_doi` does) would treat
    such a cell as one opaque token and under-detect overlap against a
    Hargreaves row citing just one of the pipe-joined DOIs.
    """
    normalized = normalize_doi(doi)
    if normalized is None:
        return set()
    return {part.strip() for part in normalized.split("|") if part.strip()}


def build_obelix_doi_set(obelix_df: pd.DataFrame, doi_column: str = "DOI") -> set[str]:
    """Flat set of every individual normalized DOI appearing anywhere in
    OBELiX's raw DOI column, split on '|'."""
    dois: set[str] = set()
    for cell in obelix_df[doi_column]:
        dois |= split_doi_field(cell)
    return dois


def classify_source_disjoint(
    strict_retained_ledger: pd.DataFrame,
    obelix_doi_set: set[str],
    *,
    formula_column: str = "reduced_formula",
    source_column: str = "source",
    exclusion_reason_column: str = "exclusion_reason",
) -> dict[str, dict[str, Any]]:
    """`{reduced_formula: {"is_source_disjoint": bool, "contributing_dois": [...],
    "overlapping_dois": [...]}}` for every reduced_formula present in
    `strict_retained_ledger` rows whose
    `exclusion_reason_column` equals 'RETAINED' (N2's strict pre-aggregation
    204-formula population). A formula is source-disjoint iff NONE of its
    contributing raw rows' (possibly multi-DOI) source cell overlaps
    `obelix_doi_set`.
    """
    retained = strict_retained_ledger.loc[
        strict_retained_ledger[exclusion_reason_column] == RETAINED_EXCLUSION_REASON
    ]
    out: dict[str, dict[str, Any]] = {}
    for formula, group in retained.groupby(formula_column):
        contributing: set[str] = set()
        for cell in group[source_column]:
            contributing |= split_doi_field(cell)
        overlapping = contributing & obelix_doi_set
        out[str(formula)] = {
            "is_source_disjoint": len(overlapping) == 0,
            "n_contributing_rows": int(len(group)),
            "contributing_dois": sorted(contributing),
            "overlapping_dois": sorted(overlapping),
        }
    return out


def source_disjoint_formulas(classification: dict[str, dict[str, Any]]) -> list[str]:
    """Sorted list of formulas classified source-disjoint -- deterministic order."""
    return sorted(f for f, v in classification.items() if v["is_source_disjoint"])


def score_subset_from_frozen_predictions(
    strict_predictions: dict[str, Any],
    keep_formulas: set[str],
) -> dict[str, Any]:
    """Recompute MAE/RMSE/Spearman on the subset of an N6
    `results.<model>.strict.predictions` payload (keys `reduced_formula`,
    `y_true`, `y_pred`) whose `reduced_formula` is in `keep_formulas`. No
    prediction is recomputed -- this only re-scores already-frozen values on
    a smaller row set."""
    formulas: list[str] = strict_predictions["reduced_formula"]
    y_true = np.asarray(strict_predictions["y_true"], dtype=float)
    y_pred = np.asarray(strict_predictions["y_pred"], dtype=float)
    mask = np.array([f in keep_formulas for f in formulas])
    if mask.sum() == 0:
        raise ValueError("score_subset_from_frozen_predictions: empty subset after filtering")
    metrics = regression_metrics(y_true[mask], y_pred[mask])
    metrics["n"] = int(mask.sum())
    return metrics
