"""Cross-dataset study, reviewer-remediation pass (N20): pure helper
functions for the four reviewer-remediation audits (A external family audit,
B random-CV duplicate/leakage audit, C cross-database label compatibility).

Kept independent of training/model code (CLAUDE.md "Keep evaluation
independent from training"): nothing here fits a model, tunes
hyperparameters, or reads a raw file -- callers pass already-loaded
frames/series. Reuses `sse_transfer.evaluation.metrics.regression_metrics`/
`bootstrap_metric_ci` for every statistic rather than reimplementing MAE/
RMSE/Spearman/bootstrap-CI machinery a third time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from sse_transfer.evaluation.metrics import bootstrap_metric_ci, regression_metrics

REVIEWER_REMEDIATION_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Part A: external family audit
# ---------------------------------------------------------------------------

#: Deterministic crosswalk from a raw Hargreaves `family` string to the
#: matching value in OBELiX's harmonized `family_filled` vocabulary (the
#: exact strings produced by
#: `sse_transfer.evaluation.internal_validation.canonicalize_and_fill_family`
#: over the 449-row OBELiX dev population). Built by exact match after
#: case/hyphen/space normalization and singular/plural folding ONLY --
#: no fuzzy/edit-distance matching is used here (unlike
#: `family_harmonization.detect_alias_clusters`, which is detection-only and
#: explicitly rejects short fuzzy matches such as LISICON/NASICON as two
#: chemically distinct families). A raw Hargreaves family with no exact
#: normalized match to any OBELiX family is left unmapped (`None`) rather
#: than guessed -- this includes real, well-known SSE chemistries with no
#: OBELiX-vocabulary counterpart (Glass, Glass-Ceramic, Lysonite, Olivine,
#: Phenakite, Rocksalt) and one explicitly rejected merge:
#: "Zircon" (Hargreaves) is NOT mapped to OBELiX's "zirconates" -- a
#: zircon-structure mineral (ZrSiO4-type) and a zirconate oxide family are
#: chemically distinct despite the shared root, mirroring the project's
#: existing LISICON/NASICON precedent of never merging on string similarity
#: alone.
HARGREAVES_TO_OBELIX_FAMILY_CROSSWALK: dict[str, str | None] = {
    "Anti-Perovskite": "antiperovskite",
    "Argyrodite": "argyrodites",
    "Garnet": "garnet",
    "LISICON": "LISICON",
    "NASICON": "NASICON",
    "Perovskite": "perovskites",
    "Thio-LISICON": "thio-LISICON",
    "Glass": None,
    "Glass-Ceramic": None,
    "Lysonite": None,
    "Olivine": None,
    "Other": None,
    "Phenakite": None,
    "Rocksalt": None,
    "Zircon": None,
}

#: Raw Hargreaves family labels for which no crosswalk decision is possible
#: because the label itself is a non-informative catch-all, not a real
#: chemical family -- kept distinct from an unambiguous "real family with no
#: OBELiX analog" (Glass, Olivine, ...), per the task's requirement to record
#: "unmatched/ambiguous cases" separately from a genuine seen/unseen split.
AMBIGUOUS_HARGREAVES_FAMILY_LABELS: frozenset[str] = frozenset({"Other"})

SEEN = "SEEN_IN_OBELIX"
UNSEEN = "UNSEEN_IN_OBELIX"
AMBIGUOUS = "AMBIGUOUS_NON_INFORMATIVE_LABEL"
MISSING = "MISSING_FAMILY_LABEL"


def classify_family_seen_status(
    raw_hargreaves_family: object, obelix_family_vocabulary: set[str]
) -> str:
    """One row's family-seen classification against the OBELiX dev-pool
    family vocabulary (`{SEEN, UNSEEN, AMBIGUOUS, MISSING}`).

    `obelix_family_vocabulary` is the set of harmonized family values
    actually present (n>=1) in the 449-row OBELiX development population --
    a crosswalk match to a family that happens not to occur in the dev pool
    would still be classified UNSEEN, since "seen" means seen in the
    development set specifically, not merely a member of OBELiX's Family
    vocabulary in the abstract.
    """
    if raw_hargreaves_family is None or (
        isinstance(raw_hargreaves_family, float) and pd.isna(raw_hargreaves_family)
    ):
        return MISSING
    raw = str(raw_hargreaves_family)
    if raw in AMBIGUOUS_HARGREAVES_FAMILY_LABELS:
        return AMBIGUOUS
    if raw not in HARGREAVES_TO_OBELIX_FAMILY_CROSSWALK:
        return AMBIGUOUS
    matched = HARGREAVES_TO_OBELIX_FAMILY_CROSSWALK[raw]
    if matched is None:
        return UNSEEN
    return SEEN if matched in obelix_family_vocabulary else UNSEEN


def assign_formula_family(
    retained_ledger: pd.DataFrame,
    *,
    formula_column: str = "reduced_formula",
    family_column: str = "family",
) -> dict[str, dict[str, Any]]:
    """Per aggregated reduced_formula (one strict-cohort row after N2's
    median aggregation may pool >=1 raw Hargreaves rows): the raw family
    label(s) contributed by its source rows, and whether they agree.

    A formula whose contributing raw rows disagree on `family` is flagged
    `family_agreement=False` and `assigned_family=None` -- never silently
    resolved by majority vote, since a disagreement here means the same
    aggregated conductivity value would be assigned a different family
    depending on which source row's label was trusted.
    """
    out: dict[str, dict[str, Any]] = {}
    for formula, group in retained_ledger.groupby(formula_column):
        raw_values = group[family_column].tolist()
        unique_values = sorted({v for v in raw_values if pd.notna(v)})
        agreement = len(unique_values) <= 1
        out[str(formula)] = {
            "n_contributing_rows": int(len(group)),
            "raw_family_values": unique_values,
            "family_agreement": agreement,
            "assigned_family": unique_values[0] if agreement and unique_values else None,
        }
    return out


# ---------------------------------------------------------------------------
# Part B: random-CV duplicate / leakage audit
# ---------------------------------------------------------------------------


def find_duplicate_formula_groups(canonical_formulas: pd.Series) -> dict[str, list[int]]:
    """`{canonical_formula: [row positions]}` for every canonical formula
    (pymatgen-reduced string) that appears more than once in
    `canonical_formulas` (a 0..n-1 positionally indexed Series). Rows with a
    null/unparseable formula are excluded (never grouped as "the same
    formula" by a missing-value coincidence)."""
    positions_by_formula: dict[str, list[int]] = {}
    for pos, formula in enumerate(canonical_formulas.to_numpy()):
        if formula is None or (isinstance(formula, float) and pd.isna(formula)):
            continue
        positions_by_formula.setdefault(str(formula), []).append(pos)
    return {f: p for f, p in positions_by_formula.items() if len(p) > 1}


def fold_id_array(n_rows: int, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Row position -> outer-fold index (the fold in which that row is the
    TEST partition), from the `(train_idx, test_idx)` tuples a splitter
    function returns. Raises if any row is never a test row or is a test row
    in more than one fold (a malformed/non-partitioning split)."""
    fold_id = np.full(n_rows, -1, dtype=int)
    for fold_i, (_train_idx, test_idx) in enumerate(folds):
        if (fold_id[test_idx] != -1).any():
            raise RuntimeError("fold_id_array: a row is the test row of more than one fold")
        fold_id[test_idx] = fold_i
    if (fold_id < 0).any():
        raise RuntimeError("fold_id_array: a row was never a test row in any fold")
    return fold_id


def duplicate_groups_crossing_folds(
    duplicate_groups: dict[str, list[int]], fold_id: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Per duplicate-formula group: which fold(s) its member rows fall into,
    and whether the group crosses >1 fold. A group crossing folds means at
    least one row of that formula is the TEST row of a fold while another
    row of the SAME formula is necessarily in that fold's TRAIN partition
    (K-fold: every row not selected as that fold's test row is in its
    train) -- i.e. the model sees one measurement of a formula in training
    while being scored on another measurement of the identical formula in
    that fold's test partition."""
    out: dict[str, dict[str, Any]] = {}
    for formula, positions in duplicate_groups.items():
        fold_ids = sorted({int(fold_id[p]) for p in positions})
        out[formula] = {
            "n_rows": len(positions),
            "fold_ids": fold_ids,
            "crosses_folds": len(fold_ids) > 1,
        }
    return out


# ---------------------------------------------------------------------------
# Part C: cross-database label compatibility
# ---------------------------------------------------------------------------


def paired_label_compatibility(
    obelix_y_by_formula: dict[str, float],
    hargreaves_log_target_by_formula: dict[str, float],
    *,
    seed: int,
    n_boot: int = 2000,
    ci: float = 0.95,
    min_n_for_ci: int = 10,
) -> dict[str, Any]:
    """Compare OBELiX's canonical `y` against Hargreaves'
    `log_target_rt_median` for every reduced_formula present in both maps
    (an explicit calibration/compatibility audit -- these rows are never
    used as external-test data; see DATA_CONTRACT.md "Canonical label").

    Treats OBELiX as the reference (`y_true`) and Hargreaves as the
    comparison (`y_pred`) purely to reuse `regression_metrics`'s existing
    MAE/RMSE/Spearman machinery -- there is no true/predicted relationship
    between two independently measured databases; `mae`/`rmse` here measure
    cross-database DISCREPANCY, not model error, and the report must say so.
    """
    shared = sorted(set(obelix_y_by_formula) & set(hargreaves_log_target_by_formula))
    if not shared:
        return {
            "n_paired_formulas": 0,
            "note": "No formula is present in both the OBELiX and Hargreaves per-formula maps.",
        }
    obelix_vals = np.array([obelix_y_by_formula[f] for f in shared], dtype=float)
    hargreaves_vals = np.array([hargreaves_log_target_by_formula[f] for f in shared], dtype=float)
    signed_diff = hargreaves_vals - obelix_vals

    base = regression_metrics(obelix_vals, hargreaves_vals)
    # Pearson r is undefined for a constant input (zero variance); guard the
    # same way every other correlation call site in this project does
    # (e.g. evaluation.regime_utility.rank_agreement,
    # uncertainty.applicability_domain.ad_score_error_correlation) rather
    # than letting scipy raise ConstantInputWarning and return NaN anyway.
    pearson_defined = (
        len(shared) > 1
        and not np.allclose(obelix_vals, obelix_vals[0])
        and not np.allclose(hargreaves_vals, hargreaves_vals[0])
    )
    pearson_r = (
        float(scipy_stats.pearsonr(obelix_vals, hargreaves_vals).statistic)
        if pearson_defined
        else float("nan")
    )

    out: dict[str, Any] = {
        "n_paired_formulas": len(shared),
        "reduced_formulas": shared,
        "mean_absolute_discrepancy": base["mae"],
        "median_absolute_discrepancy": float(np.median(np.abs(signed_diff))),
        "rmse": base["rmse"],
        "spearman_rho": base["spearman_rho"],
        "pearson_r": float(pearson_r),
        "median_signed_difference_hargreaves_minus_obelix": float(np.median(signed_diff)),
        "mean_signed_difference_hargreaves_minus_obelix": float(np.mean(signed_diff)),
        "n_pairs_absolute_discrepancy_over_1_log10": int(np.sum(np.abs(signed_diff) > 1.0)),
        "n_pairs_absolute_discrepancy_over_2_log10": int(np.sum(np.abs(signed_diff) > 2.0)),
    }
    if len(shared) >= min_n_for_ci:
        out["mae_bootstrap_ci"] = bootstrap_metric_ci(
            obelix_vals, hargreaves_vals, "mae", seed=seed, n_boot=n_boot, ci=ci
        )
        out["spearman_bootstrap_ci"] = bootstrap_metric_ci(
            obelix_vals, hargreaves_vals, "spearman_rho", seed=seed, n_boot=n_boot, ci=ci
        )
    else:
        out["bootstrap_ci_note"] = (
            f"n_paired_formulas={len(shared)} < min_n_for_ci={min_n_for_ci}; "
            "bootstrap CI not computed (not statistically meaningful at this n)."
        )
    return out
