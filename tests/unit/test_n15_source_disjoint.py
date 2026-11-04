import pandas as pd
import pytest

from sse_transfer.evaluation.n15_source_disjoint import (
    build_obelix_doi_set,
    classify_source_disjoint,
    normalize_doi,
    score_subset_from_frozen_predictions,
    source_disjoint_formulas,
    split_doi_field,
)


def test_normalize_doi_lowercases_and_strips() -> None:
    assert normalize_doi("  10.1000/ABC  ") == "10.1000/abc"
    assert normalize_doi(None) is None
    assert normalize_doi(float("nan")) is None


def test_split_doi_field_splits_pipe_delimited_cell() -> None:
    assert split_doi_field("10.1/a|10.2/b|10.3/c") == {"10.1/a", "10.2/b", "10.3/c"}
    assert split_doi_field("10.1/a") == {"10.1/a"}
    assert split_doi_field(None) == set()


def test_build_obelix_doi_set_splits_multi_doi_rows() -> None:
    obelix = pd.DataFrame({"DOI": ["10.1/a", "10.2/b|10.3/c", "10.1/a"]})
    doi_set = build_obelix_doi_set(obelix)
    assert doi_set == {"10.1/a", "10.2/b", "10.3/c"}


def test_classify_source_disjoint_excludes_any_overlapping_row() -> None:
    ledger = pd.DataFrame(
        {
            "reduced_formula": ["LiF", "LiF", "Li2O", "LiCl"],
            "source": ["10.1/x", "10.9/overlap", "10.2/y", "10.3/z"],
            "exclusion_reason": ["RETAINED", "RETAINED", "RETAINED", "EXCLUDED_OTHER"],
        }
    )
    obelix_doi_set = {"10.9/overlap"}
    out = classify_source_disjoint(ledger, obelix_doi_set)
    # LiF has two contributing rows, one of which overlaps -> excluded.
    assert out["LiF"]["is_source_disjoint"] is False
    assert out["LiF"]["overlapping_dois"] == ["10.9/overlap"]
    # Li2O has no overlap -> retained.
    assert out["Li2O"]["is_source_disjoint"] is True
    # LiCl was not RETAINED in the ledger -> not part of the strict population at all.
    assert "LiCl" not in out


def test_source_disjoint_formulas_sorted_and_filtered() -> None:
    classification = {
        "LiF": {"is_source_disjoint": False},
        "Li2O": {"is_source_disjoint": True},
        "LiCl": {"is_source_disjoint": True},
    }
    assert source_disjoint_formulas(classification) == ["Li2O", "LiCl"]


def test_score_subset_from_frozen_predictions_filters_by_formula() -> None:
    predictions = {
        "reduced_formula": ["LiF", "Li2O", "LiCl"],
        "y_true": [1.0, 2.0, 3.0],
        "y_pred": [1.5, 2.0, 10.0],
    }
    out = score_subset_from_frozen_predictions(predictions, keep_formulas={"LiF", "Li2O"})
    assert out["n"] == 2
    assert out["mae"] == pytest.approx((0.5 + 0.0) / 2)


def test_score_subset_from_frozen_predictions_rejects_empty_subset() -> None:
    predictions = {"reduced_formula": ["LiF"], "y_true": [1.0], "y_pred": [1.0]}
    with pytest.raises(ValueError):
        score_subset_from_frozen_predictions(predictions, keep_formulas={"NotPresent"})
