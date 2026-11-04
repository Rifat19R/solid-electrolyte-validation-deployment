import numpy as np
import pandas as pd
import pytest

from sse_transfer.evaluation.reviewer_remediation import (
    AMBIGUOUS,
    HARGREAVES_TO_OBELIX_FAMILY_CROSSWALK,
    MISSING,
    SEEN,
    UNSEEN,
    assign_formula_family,
    classify_family_seen_status,
    duplicate_groups_crossing_folds,
    find_duplicate_formula_groups,
    fold_id_array,
    paired_label_compatibility,
)


def test_crosswalk_rejects_zircon_zirconate_merge() -> None:
    # Explicit documented decision: Zircon (Hargreaves) is NOT the same
    # family as OBELiX's "zirconates" despite the shared root.
    assert HARGREAVES_TO_OBELIX_FAMILY_CROSSWALK["Zircon"] is None


def test_classify_family_seen_status_seen() -> None:
    vocab = {"garnet", "NASICON"}
    assert classify_family_seen_status("Garnet", vocab) == SEEN
    assert classify_family_seen_status("NASICON", vocab) == SEEN


def test_classify_family_seen_status_unseen_real_family() -> None:
    vocab = {"garnet"}
    assert classify_family_seen_status("Olivine", vocab) == UNSEEN
    # Crosswalk matches "zirconates" conceptually rejected -> Zircon is UNSEEN.
    assert classify_family_seen_status("Zircon", vocab) == UNSEEN


def test_classify_family_seen_status_matched_family_absent_from_dev_pool_is_unseen() -> None:
    # "Garnet" crosswalks to "garnet", but if "garnet" has zero rows in this
    # particular dev-pool vocabulary, it must not be called SEEN.
    vocab = {"NASICON"}
    assert classify_family_seen_status("Garnet", vocab) == UNSEEN


def test_classify_family_seen_status_ambiguous_and_missing() -> None:
    vocab = {"garnet"}
    assert classify_family_seen_status("Other", vocab) == AMBIGUOUS
    assert classify_family_seen_status(None, vocab) == MISSING
    assert classify_family_seen_status(float("nan"), vocab) == MISSING
    assert classify_family_seen_status("SomeUnknownLabel", vocab) == AMBIGUOUS


def test_assign_formula_family_agreement() -> None:
    ledger = pd.DataFrame(
        {
            "reduced_formula": ["LiF", "LiF", "Li2O"],
            "family": ["halides", "halides", "oxides"],
        }
    )
    out = assign_formula_family(ledger)
    assert out["LiF"]["family_agreement"] is True
    assert out["LiF"]["assigned_family"] == "halides"
    assert out["Li2O"]["n_contributing_rows"] == 1


def test_assign_formula_family_disagreement_not_resolved() -> None:
    ledger = pd.DataFrame(
        {
            "reduced_formula": ["LiF", "LiF"],
            "family": ["halides", "oxides"],
        }
    )
    out = assign_formula_family(ledger)
    assert out["LiF"]["family_agreement"] is False
    assert out["LiF"]["assigned_family"] is None
    assert out["LiF"]["raw_family_values"] == ["halides", "oxides"]


def test_find_duplicate_formula_groups() -> None:
    formulas = pd.Series(["LiF", "Li2O", "LiF", "LiF", None, "LiCl"])
    out = find_duplicate_formula_groups(formulas)
    assert out == {"LiF": [0, 2, 3]}


def test_fold_id_array_basic() -> None:
    folds = [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
    ]
    fold_id = fold_id_array(4, folds)
    assert list(fold_id) == [0, 0, 1, 1]


def test_fold_id_array_rejects_row_never_tested() -> None:
    folds = [(np.array([1, 2]), np.array([0]))]
    with pytest.raises(RuntimeError):
        fold_id_array(3, folds)


def test_duplicate_groups_crossing_folds() -> None:
    fold_id = np.array([0, 0, 1, 1, 2])
    groups = {"LiF": [0, 1], "Li2O": [1, 2], "LiCl": [3, 4]}
    out = duplicate_groups_crossing_folds(groups, fold_id)
    assert out["LiF"]["crosses_folds"] is False
    assert out["LiF"]["fold_ids"] == [0]
    assert out["Li2O"]["crosses_folds"] is True
    assert out["Li2O"]["fold_ids"] == [0, 1]
    assert out["LiCl"]["crosses_folds"] is True


def test_paired_label_compatibility_basic_stats() -> None:
    obelix = {"LiF": -3.0, "Li2O": -5.0, "LiCl": -4.0}
    hargreaves = {"LiF": -3.2, "Li2O": -4.8, "LiCl": -4.0, "Extra": -1.0}
    out = paired_label_compatibility(obelix, hargreaves, seed=0, min_n_for_ci=10)
    assert out["n_paired_formulas"] == 3
    assert out["mean_absolute_discrepancy"] == pytest.approx((0.2 + 0.2 + 0.0) / 3)
    assert "bootstrap_ci_note" in out  # n=3 < min_n_for_ci=10


def test_paired_label_compatibility_empty_intersection() -> None:
    out = paired_label_compatibility({"LiF": -3.0}, {"Li2O": -5.0}, seed=0)
    assert out["n_paired_formulas"] == 0


def test_paired_label_compatibility_computes_ci_above_threshold() -> None:
    rng = np.random.default_rng(0)
    obelix = {f"F{i}": float(rng.normal()) for i in range(15)}
    hargreaves = {k: v + 0.1 for k, v in obelix.items()}
    out = paired_label_compatibility(obelix, hargreaves, seed=0, min_n_for_ci=10, n_boot=200)
    assert "mae_bootstrap_ci" in out
    assert out["mae_bootstrap_ci"]["point"] == pytest.approx(0.1, abs=1e-9)


def test_paired_label_compatibility_constant_obelix_values_pearson_is_nan_without_warning() -> None:
    """Regression test: Pearson r is undefined for a constant input; must be
    NaN without scipy raising ConstantInputWarning (matches the guard
    already used by every other correlation call site in this project)."""
    import math
    import warnings

    obelix = {"LiF": -3.0, "Li2O": -3.0, "LiCl": -3.0}
    hargreaves = {"LiF": -3.2, "Li2O": -4.8, "LiCl": -4.0}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = paired_label_compatibility(obelix, hargreaves, seed=0, min_n_for_ci=10)
    assert math.isnan(out["pearson_r"])
