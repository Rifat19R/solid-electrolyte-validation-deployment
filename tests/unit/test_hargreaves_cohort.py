import pandas as pd
import pytest

from sse_transfer.data import hargreaves_cohort
from sse_transfer.data.hargreaves_cohort import (
    EXCLUSION_COMPOSITION_ONLY,
    EXCLUSION_EXACT_DOI_DUP,
    EXCLUSION_INVALID_TARGET,
    EXCLUSION_RT_WINDOW,
    EXCLUSION_UNPARSEABLE_COMPOSITION,
    MATCH_COMPOSITION_ONLY,
    MATCH_EXACT_DOI,
    MATCH_NONE,
    RAW_OBELIX,
    aggregate_by_formula,
    build_cohorts,
    build_obelix_formula_doi_index,
    classify_overlap,
    rt_window_mask,
    safe_reduced_formula,
    valid_target_mask,
)


def test_rt_window_mask_boundaries_inclusive() -> None:
    temps = pd.Series([14.9, 15.0, 25.0, 35.0, 35.1, None])
    result = rt_window_mask(temps)
    assert result.tolist() == [False, True, True, True, False, False]


def test_safe_reduced_formula_parses_simple_composition() -> None:
    assert safe_reduced_formula("Li7La3Zr2O12") == "Li7La3Zr2O12"


def test_safe_reduced_formula_strips_csv_quote_artifact() -> None:
    quoted = '"(Li2S)0.7 (P2S5)0.3"'
    unquoted = "(Li2S)0.7 (P2S5)0.3"
    assert safe_reduced_formula(quoted) == safe_reduced_formula(unquoted)
    assert safe_reduced_formula(quoted) is not None


def test_safe_reduced_formula_returns_none_for_garbage() -> None:
    assert safe_reduced_formula("not a formula!!") is None
    assert safe_reduced_formula(None) is None
    assert safe_reduced_formula(float("nan")) is None
    assert safe_reduced_formula("") is None


def test_valid_target_mask_rejects_nonpositive_and_nonfinite() -> None:
    targets = pd.Series([1e-5, 0.0, -1.0, float("inf"), float("nan"), "5e-3"])
    result = valid_target_mask(targets)
    assert result.tolist() == [True, False, False, False, False, True]


def test_build_obelix_formula_doi_index_uses_both_composition_columns() -> None:
    obelix = pd.DataFrame(
        {
            "Reduced Composition": ["Li7La3Zr2O12"],
            "True Composition": ["Li7.0La3.0Zr2.0O12.0"],
            "DOI": ["10.1234/example"],
        }
    )
    index = build_obelix_formula_doi_index(obelix)
    assert index["Li7La3Zr2O12"] == {"10.1234/example"}


def test_classify_overlap_exact_doi_match() -> None:
    index = {"Li7La3Zr2O12": {"10.1234/example"}}
    assert classify_overlap("Li7La3Zr2O12", "10.1234/example", index) == MATCH_EXACT_DOI


def test_classify_overlap_composition_only_when_doi_differs() -> None:
    index = {"Li7La3Zr2O12": {"10.1234/example"}}
    assert classify_overlap("Li7La3Zr2O12", "10.9999/other", index) == MATCH_COMPOSITION_ONLY


def test_classify_overlap_no_match_when_formula_absent() -> None:
    index = {"Li7La3Zr2O12": {"10.1234/example"}}
    assert classify_overlap("LiFePO4", "10.1234/example", index) == MATCH_NONE


def test_classify_overlap_no_match_for_unparseable_formula() -> None:
    index = {"Li7La3Zr2O12": {"10.1234/example"}}
    assert classify_overlap(None, "10.1234/example", index) == MATCH_NONE


def test_aggregate_by_formula_uses_median_and_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "reduced_formula": ["LiFePO4", "LiFePO4", "LiFePO4", "Li7La3Zr2O12"],
            "source": ["10.1/a", "10.1/a", "10.2/b", "10.3/c"],
            "log_target": [-4.0, -6.0, -5.0, -3.0],
        }
    )
    result = aggregate_by_formula(df).set_index("reduced_formula")
    assert result.loc["LiFePO4", "log_target_rt_median"] == -5.0
    assert result.loc["LiFePO4", "n_source_rows"] == 3
    assert result.loc["LiFePO4", "n_unique_sources"] == 2
    assert result.loc["Li7La3Zr2O12", "log_target_rt_median"] == -3.0
    # Determinism: rerunning on a differently-ordered but identical input gives the same result.
    shuffled = df.iloc[::-1].reset_index(drop=True)
    result2 = aggregate_by_formula(shuffled).set_index("reduced_formula")
    assert result.sort_index().equals(result2.sort_index())


def test_aggregate_by_formula_empty_input() -> None:
    empty = pd.DataFrame(columns=["reduced_formula", "source", "log_target"])
    result = aggregate_by_formula(empty)
    assert len(result) == 0
    assert "reduced_formula" in result.columns


def test_build_cohorts_gate_n2_passes_and_relaxed_is_superset_size() -> None:
    result = build_cohorts()
    # Gate N2 (DATA_CONTRACT.md / EXPERIMENT_PROTOCOL.md v2.0): strict
    # external cohort must have >=100 independent rows to be statistically useful.
    assert result.stage_counts["strict_independent_rows"] >= 100
    # Relaxed excludes strictly less than strict (only exact-DOI duplicates, not
    # composition-only overlap), so it can never be smaller than strict.
    strict_n = result.stage_counts["strict_independent_rows"]
    relaxed_n = result.stage_counts["relaxed_independent_rows"]
    assert relaxed_n >= strict_n
    # Every raw row must be accounted for by exactly one ledger disposition.
    assert len(result.ledger) == result.stage_counts["raw_rows"]
    assert result.ledger["exclusion_reason"].notna().all()


def test_build_cohorts_strict_cohort_has_no_overlap_with_obelix() -> None:
    result = build_cohorts()
    strict_formulas = set(result.strict_cohort["reduced_formula"])
    obelix_index = build_obelix_formula_doi_index(pd.read_csv(RAW_OBELIX))
    assert strict_formulas.isdisjoint(obelix_index.keys())


# Synthetic stand-ins for the two raw files, with one row per ledger
# disposition. They exercise the same build_cohorts() code path as the two
# real-data tests above, which need the non-redistributable Hargreaves file;
# the real-data population size gate (>=100 strict rows) stays with those.
_SYNTHETIC_HARGREAVES = pd.DataFrame(
    {
        "composition": [
            "LiFePO4",
            "LiFePO4",
            "Li7La3Zr2O12",
            "Li3PS4",
            "Li2S",
            "not a formula!!",
            "LiCl",
        ],
        "temperature": [25.0, 30.0, 25.0, 25.0, 100.0, 25.0, 25.0],
        "target": [1e-4, 1e-5, 1e-3, 1e-3, 1e-2, 1e-4, 0.0],
        "log_target": [-4.0, -5.0, -3.0, -3.0, -2.0, -4.0, float("nan")],
        "source": ["10.1/a", "10.1/b", "10.9/OBX", "10.1/c", "10.1/d", "10.1/e", "10.1/f"],
    }
)
_SYNTHETIC_OBELIX = pd.DataFrame(
    {
        "Reduced Composition": ["Li7La3Zr2O12", "Li3PS4"],
        "True Composition": ["Li7.0La3.0Zr2.0O12.0", "Li3.0P1.0S4.0"],
        "DOI": ["10.9/obx", "10.9/obx2"],
    }
)


@pytest.fixture
def synthetic_cohorts(monkeypatch: pytest.MonkeyPatch) -> hargreaves_cohort.CohortBuildResult:
    monkeypatch.setattr(hargreaves_cohort, "load_hargreaves_raw", _SYNTHETIC_HARGREAVES.copy)
    monkeypatch.setattr(hargreaves_cohort, "load_obelix_raw", _SYNTHETIC_OBELIX.copy)
    return build_cohorts()


def test_build_cohorts_synthetic_ledger_and_strict_relaxed_accounting(
    synthetic_cohorts: hargreaves_cohort.CohortBuildResult,
) -> None:
    result = synthetic_cohorts
    assert len(result.ledger) == result.stage_counts["raw_rows"] == 7
    assert result.ledger["exclusion_reason"].tolist() == [
        "RETAINED",
        "RETAINED",
        EXCLUSION_EXACT_DOI_DUP,
        EXCLUSION_COMPOSITION_ONLY,
        EXCLUSION_RT_WINDOW,
        EXCLUSION_UNPARSEABLE_COMPOSITION,
        EXCLUSION_INVALID_TARGET,
    ]
    assert result.stage_counts["strict_independent_rows"] == 1
    assert result.stage_counts["relaxed_independent_rows"] == 2
    assert set(result.strict_cohort["reduced_formula"]) == {"LiFePO4"}
    # Relaxed keeps the composition-only overlap, never the exact-DOI duplicate.
    assert set(result.relaxed_cohort["reduced_formula"]) == {"LiFePO4", "Li3PS4"}
    strict = result.strict_cohort.set_index("reduced_formula")
    assert strict.loc["LiFePO4", "log_target_rt_median"] == -4.5
    assert strict.loc["LiFePO4", "n_unique_sources"] == 2


def test_build_cohorts_synthetic_strict_cohort_has_no_overlap_with_obelix(
    synthetic_cohorts: hargreaves_cohort.CohortBuildResult,
) -> None:
    strict_formulas = set(synthetic_cohorts.strict_cohort["reduced_formula"])
    obelix_index = build_obelix_formula_doi_index(_SYNTHETIC_OBELIX)
    assert obelix_index.keys() == {"Li7La3Zr2O12", "Li3PS4"}
    assert strict_formulas and strict_formulas.isdisjoint(obelix_index.keys())
