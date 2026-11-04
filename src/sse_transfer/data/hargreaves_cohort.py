"""Strict/relaxed external Hargreaves conductivity cohort construction (N2).

See DATA_CONTRACT.md "Canonical identity and overlap audit" and reports/npj/
n1_external_data_audit.md / n2_external_cohort.md. This module never writes to
data/raw/ (immutable per DATA_CONTRACT.md); it only reads the raw Hargreaves
CSV and OBELiX raw CSV and derives an in-memory / data/processed/ view.

Composition-only overlap classification (no structure data exists for
Hargreaves), reusing the classification-name convention from
sse_transfer.audit.leakage:
  EXACT_COMPOSITION_AND_DOI_MATCH -- same reduced formula AND same source DOI
      as an OBELiX row: near-certain the same underlying measurement/material.
  COMPOSITION_ONLY_MATCH -- same reduced formula, different/unverifiable DOI:
      plausibly a different sample/polymorph of the same nominal material, or
      the same material reported independently by two papers.
  NO_MATCH -- reduced formula does not appear anywhere in OBELiX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

HARGREAVES_COHORT_SCHEMA_VERSION = "1.0"

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data"
RAW_HARGREAVES = _DATA_ROOT / "raw" / "hargreaves_conductivity" / "LiIonDatabase.csv"
RAW_OBELIX = _DATA_ROOT / "raw" / "obelix" / "all.csv"

#: DATA_CONTRACT.md canonical target y = log10(sigma_RT / S cm^-1). Room-temperature
#: window chosen on physical/measurement-convention grounds (ambient lab temperature,
#: matching common "room temperature" reporting practice in the SSE literature and the
#: window this project independently reproduced against the Hargreaves abstract in N1;
#: see reports/npj/n1_external_data_audit.md section 4) -- fixed BEFORE inspecting how
#: many rows it yields, per EXPERIMENT_PROTOCOL.md "Never select a split because it
#: produces a better score."
RT_WINDOW_LOW_C = 15.0
RT_WINDOW_HIGH_C = 35.0

EXCLUSION_RT_WINDOW = "INCOMPATIBLE_TEMPERATURE_CONVENTION"
EXCLUSION_INVALID_TARGET = "INVALID_OR_MISSING_TARGET"
EXCLUSION_UNPARSEABLE_COMPOSITION = "UNPARSEABLE_COMPOSITION"
EXCLUSION_EXACT_DOI_DUP = "EXACT_COMPOSITION_AND_DOI_MATCH_WITH_OBELIX"
EXCLUSION_COMPOSITION_ONLY = "COMPOSITION_ONLY_MATCH_WITH_OBELIX"

MATCH_EXACT_DOI = "EXACT_COMPOSITION_AND_DOI_MATCH"
MATCH_COMPOSITION_ONLY = "COMPOSITION_ONLY_MATCH"
MATCH_NONE = "NO_MATCH"


def load_hargreaves_raw() -> pd.DataFrame:
    """Read the raw Hargreaves CSV, skipping its embedded license/citation preamble.

    The file's first 4 physical rows (2 text rows + 2 blank rows) precede the real
    header on row index 4; pandas' default blank-line skipping shifts the header
    position, so skip_blank_lines=False is required (see reports/npj/
    n1_external_data_audit.md section 1).
    """
    df = pd.read_csv(RAW_HARGREAVES, skiprows=4, skip_blank_lines=False)
    df = df.dropna(how="all").reset_index(drop=True)
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["log_target"] = pd.to_numeric(df["log_target"], errors="coerce")
    return df


def load_obelix_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_OBELIX)


def safe_reduced_formula(composition_str: object) -> str | None:
    """pymatgen reduced formula for a composition string, or None if unparseable.

    Never raises: an unparseable composition is a MISSING-target-equivalent
    condition for the overlap audit, not a crash.
    """
    if composition_str is None or (isinstance(composition_str, float) and pd.isna(composition_str)):
        return None
    text = str(composition_str).strip()
    # Some raw Hargreaves pseudo-binary/glass compositions (e.g. "(Li2S)0.7 (P2S5)0.3")
    # are wrapped in literal double quotes in the source CSV -- a CSV-quoting
    # artifact around commas inside the formula, not part of the chemistry.
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    if not text:
        return None
    try:
        return str(Composition(text).reduced_formula)
    except Exception:
        return None


def rt_window_mask(
    temperature_c: pd.Series, low: float = RT_WINDOW_LOW_C, high: float = RT_WINDOW_HIGH_C
) -> pd.Series:
    """True where temperature falls in the predeclared room-temperature window."""
    return (temperature_c >= low) & (temperature_c <= high)


def valid_target_mask(target: pd.Series) -> pd.Series:
    """True where the linear conductivity target is a finite positive number.

    A non-positive or non-finite conductivity is physically invalid and cannot
    be log-transformed to the canonical y = log10(sigma_RT / S cm^-1) target.
    """
    numeric = pd.to_numeric(target, errors="coerce")
    return numeric.apply(lambda v: bool(np.isfinite(v) and v > 0))


def classify_overlap(
    reduced_formula: str | None,
    source_doi: object,
    obelix_formula_to_dois: dict[str, set[str]],
) -> str:
    """Classify one Hargreaves row against the set of OBELiX (formula -> DOIs)."""
    if reduced_formula is None or reduced_formula not in obelix_formula_to_dois:
        return MATCH_NONE
    obelix_dois = obelix_formula_to_dois[reduced_formula]
    doi = _normalize_doi(source_doi)
    if doi is not None and doi in obelix_dois:
        return MATCH_EXACT_DOI
    return MATCH_COMPOSITION_ONLY


def _normalize_doi(doi: object) -> str | None:
    if doi is None or (isinstance(doi, float) and pd.isna(doi)):
        return None
    text = str(doi).strip().lower()
    if not text or text == "nan":
        return None
    return text


def build_obelix_formula_doi_index(obelix_df: pd.DataFrame) -> dict[str, set[str]]:
    """Map OBELiX reduced formula -> set of normalized DOIs reporting it.

    Uses both `Reduced Composition` and `True Composition` (re-reduced via
    pymatgen) so a Hargreaves composition string matches OBELiX regardless of
    which of the two OBELiX composition columns is closer to it in form.
    """
    index: dict[str, set[str]] = {}
    for _, row in obelix_df.iterrows():
        doi = _normalize_doi(row.get("DOI"))
        for col in ("Reduced Composition", "True Composition"):
            rf = safe_reduced_formula(row.get(col))
            if rf is None:
                continue
            index.setdefault(rf, set())
            if doi is not None:
                index[rf].add(doi)
    return index


@dataclass(frozen=True)
class AggregatedMeasurement:
    reduced_formula: str
    n_source_rows: int
    n_unique_sources: int
    log_target_rt_median: float
    aggregation_rule: str


#: One deterministic rule, fixed before any external evaluation. Repeated
#: RT-window measurements of the same reduced formula are collapsed to their
#: MEDIAN log10(conductivity) -- chosen over the mean because conductivity
#: measurements are heavy-tailed/log-scale and a median is robust to a single
#: outlying sample/method without silently discarding disagreement (the
#: dispersion is retained separately, not thrown away -- see
#: reports/npj/n2_external_cohort.md). This mirrors DATA_CONTRACT.md's
#: instruction to "define one deterministic aggregation rule" rather than an
#: ad hoc per-composition choice.
AGGREGATION_RULE = "median_log10_conductivity_across_RT_window_rows_per_reduced_formula"


def aggregate_by_formula(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated RT-window rows of the same reduced formula.

    Input `df` must already be filtered to valid, RT-window, non-excluded rows
    and have a `reduced_formula` column. Output has one row per unique
    reduced_formula with the median log10(target) and provenance counts.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "reduced_formula",
                "n_source_rows",
                "n_unique_sources",
                "log_target_rt_median",
                "aggregation_rule",
            ]
        )
    records = []
    for formula, group in df.groupby("reduced_formula"):
        records.append(
            AggregatedMeasurement(
                reduced_formula=str(formula),
                n_source_rows=len(group),
                n_unique_sources=int(group["source"].nunique()),
                log_target_rt_median=float(group["log_target"].median()),
                aggregation_rule=AGGREGATION_RULE,
            )
        )
    return pd.DataFrame([r.__dict__ for r in records])


@dataclass(frozen=True)
class CohortBuildResult:
    ledger: pd.DataFrame
    strict_cohort: pd.DataFrame
    relaxed_cohort: pd.DataFrame
    stage_counts: dict[str, int]


def build_cohorts() -> CohortBuildResult:
    """Full N2 pipeline: raw Hargreaves -> RT filter -> validity filter ->
    overlap classification -> strict/relaxed exclusion -> deterministic
    aggregation. Returns the row-level ledger and both aggregated cohorts.
    """
    hargreaves = load_hargreaves_raw()
    obelix = load_obelix_raw()
    obelix_index = build_obelix_formula_doi_index(obelix)

    stage_counts: dict[str, int] = {"raw_rows": len(hargreaves)}

    ledger = hargreaves.copy()
    ledger["reduced_formula"] = ledger["composition"].apply(safe_reduced_formula)
    ledger["in_rt_window"] = rt_window_mask(ledger["temperature"])
    ledger["valid_target"] = valid_target_mask(ledger["target"]) & ledger["log_target"].notna()
    ledger["overlap_classification"] = [
        classify_overlap(rf, doi, obelix_index)
        for rf, doi in zip(ledger["reduced_formula"], ledger["source"], strict=False)
    ]

    rt_mask = ledger["in_rt_window"]
    stage_counts["rt_window_rows"] = int(rt_mask.sum())
    stage_counts["rt_window_and_valid_target_rows"] = int((rt_mask & ledger["valid_target"]).sum())
    stage_counts["rt_window_and_parseable_composition_rows"] = int(
        (rt_mask & ledger["reduced_formula"].notna()).sum()
    )

    base_valid = ledger["in_rt_window"] & ledger["valid_target"] & ledger["reduced_formula"].notna()

    # STRICT: exclude any composition-only or DOI overlap with OBELiX (conservative).
    strict_valid = base_valid & (ledger["overlap_classification"] == MATCH_NONE)
    # RELAXED: exclude only exact composition+DOI overlap (near-certain duplicate
    # measurement); a shared composition alone is not excluded, since the same
    # nominal material can be legitimately re-measured by an independent source.
    relaxed_valid = base_valid & (ledger["overlap_classification"] != MATCH_EXACT_DOI)

    stage_counts["strict_pre_aggregation_rows"] = int(strict_valid.sum())
    stage_counts["relaxed_pre_aggregation_rows"] = int(relaxed_valid.sum())

    strict_cohort = aggregate_by_formula(ledger.loc[strict_valid])
    relaxed_cohort = aggregate_by_formula(ledger.loc[relaxed_valid])

    stage_counts["strict_independent_rows"] = len(strict_cohort)
    stage_counts["relaxed_independent_rows"] = len(relaxed_cohort)

    def _exclusion_reason(row: pd.Series) -> str:
        if not row["in_rt_window"]:
            return EXCLUSION_RT_WINDOW
        if pd.isna(row["reduced_formula"]):
            return EXCLUSION_UNPARSEABLE_COMPOSITION
        if not row["valid_target"]:
            return EXCLUSION_INVALID_TARGET
        if row["overlap_classification"] == MATCH_EXACT_DOI:
            return EXCLUSION_EXACT_DOI_DUP
        if row["overlap_classification"] == MATCH_COMPOSITION_ONLY:
            return EXCLUSION_COMPOSITION_ONLY
        return "RETAINED"

    ledger["exclusion_reason"] = ledger.apply(_exclusion_reason, axis=1)

    return CohortBuildResult(
        ledger=ledger,
        strict_cohort=strict_cohort,
        relaxed_cohort=relaxed_cohort,
        stage_counts=stage_counts,
    )
