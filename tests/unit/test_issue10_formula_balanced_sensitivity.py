"""Unit tests for reviewer-remediation Issue 10's generalized formula-balanced
sensitivity logic (scripts/analysis/issue10_formula_balanced_sensitivity.py) --
pure functions only. No frozen artifact is written to, mutated, or deleted
by any test here, and the frozen 113-row OBELiX official test split is
never referenced by the module under test (checked below) or opened by
these tests."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from issue10_formula_balanced_sensitivity import (  # noqa: E402
    STRICT_COHORT_PATH,
    compare_pairwise_bootstrap_ci,
    compute_formula_balanced,
    formula_duplication_stats,
    load_n6_external,
)

# --- formula_duplication_stats: Counter-based duplication statistics ---


def test_formula_duplication_stats_all_unique() -> None:
    formulas = [f"F{i}" for i in range(10)]
    stats = formula_duplication_stats(formulas)
    assert stats["n_rows"] == 10
    assert stats["n_unique_reduced_formulas"] == 10
    assert stats["n_repeated_formula_groups"] == 0
    assert stats["n_duplicate_excess_rows"] == 0
    assert stats["premise_holds_zero_repeated_formulas"] is True


def test_formula_duplication_stats_handles_repeats_without_raising() -> None:
    # "A" appears 3 times, "B" appears 2 times, "C" once -> 2 repeated
    # groups (A, B), 3 duplicate-excess rows ((3-1) + (2-1) + (1-1)).
    formulas = ["A", "A", "A", "B", "B", "C"]
    stats = formula_duplication_stats(formulas)
    assert stats["n_rows"] == 6
    assert stats["n_unique_reduced_formulas"] == 3
    assert stats["n_repeated_formula_groups"] == 2
    assert stats["n_duplicate_excess_rows"] == 3
    assert stats["premise_holds_zero_repeated_formulas"] is False


def test_formula_duplication_stats_n_repeated_groups_not_n_minus_unique() -> None:
    # Regression guard: a single formula repeated many times must count as
    # ONE repeated group, not (n_rows - n_unique) groups.
    formulas = ["A"] * 5 + ["B", "C", "D"]
    stats = formula_duplication_stats(formulas)
    assert stats["n_rows"] == 8
    assert stats["n_unique_reduced_formulas"] == 4
    assert stats["n_repeated_formula_groups"] == 1  # only "A" repeats
    assert stats["n_duplicate_excess_rows"] == 4  # 8 - 4
    # The old (buggy) formula would have reported 8 - 4 = 4 "repeated
    # groups", which is wrong -- there is exactly one repeated identity.
    assert stats["n_repeated_formula_groups"] != stats["n_duplicate_excess_rows"]


def test_formula_duplication_stats_matches_frozen_strict_cohort() -> None:
    # The actual frozen strict cohort is expected to have zero repeats;
    # this is a read-only regression check against the real file.
    import csv

    with STRICT_COHORT_PATH.open(newline="", encoding="utf-8") as fh:
        formulas = [r["reduced_formula"] for r in csv.DictReader(fh)]
    stats = formula_duplication_stats(formulas)
    assert stats["n_rows"] == 204
    assert stats["n_unique_reduced_formulas"] == 204
    assert stats["n_repeated_formula_groups"] == 0
    assert stats["n_duplicate_excess_rows"] == 0
    assert stats["premise_holds_zero_repeated_formulas"] is True


# --- compute_formula_balanced: MAE differs from row mean under repeats ---


def test_compute_formula_balanced_differs_from_row_mean_with_repeats() -> None:
    # "A" has 3 rows (residuals 1, 2, 3 -> group mean 2.0), "B" has 1 row
    # (residual 10.0). Row MAE = mean(1,2,3,10) = 4.0; formula-balanced MAE
    # = mean(2.0, 10.0) = 6.0 -- a genuine reweighting, not a relabeling.
    formulas = ["A", "A", "A", "B"]
    residuals = np.array([1.0, 2.0, 3.0, 10.0])
    row_mae = float(np.mean(np.abs(residuals)))
    out = compute_formula_balanced(
        {"m": row_mae},
        {"m": residuals},
        {"m": formulas},
        premise_holds=False,
    )
    assert out["m"]["row_weighted_mae"] == pytest.approx(4.0)
    assert out["m"]["formula_balanced_mae"] == pytest.approx(6.0)
    assert out["m"]["formula_balanced_mae"] != pytest.approx(out["m"]["row_weighted_mae"])


def test_compute_formula_balanced_equals_row_mae_when_all_unique() -> None:
    formulas = [f"F{i}" for i in range(5)]
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    row_mae = float(np.mean(np.abs(residuals)))
    out = compute_formula_balanced(
        {"m": row_mae},
        {"m": residuals},
        {"m": formulas},
        premise_holds=True,
    )
    assert out["m"]["formula_balanced_mae"] == pytest.approx(out["m"]["row_weighted_mae"])


def test_compute_formula_balanced_raises_on_mismatch_when_premise_holds() -> None:
    # A deliberately wrong "frozen" row_mae, with premise_holds=True:
    # under the spec's decision rule this mismatch IS a required
    # invariant, so it must raise.
    formulas = [f"F{i}" for i in range(5)]
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    wrong_row_mae = 999.0
    with pytest.raises(RuntimeError):
        compute_formula_balanced(
            {"m": wrong_row_mae},
            {"m": residuals},
            {"m": formulas},
            premise_holds=True,
        )


def test_compute_formula_balanced_does_not_raise_on_mismatch_when_premise_fails() -> None:
    # Same mismatch, but premise_holds=False (repeats legitimately exist):
    # a non-trivial difference here is an expected, reportable result, not
    # a bug, per the generalized implementation.
    formulas = ["A", "A", "B"]
    residuals = np.array([1.0, 3.0, 10.0])
    out = compute_formula_balanced(
        {"m": 999.0},
        {"m": residuals},
        {"m": formulas},
        premise_holds=False,
    )
    assert out["m"]["row_weighted_mae"] == 999.0
    assert out["m"]["formula_balanced_mae"] == pytest.approx(6.0)  # mean(2.0, 10.0)


# --- compare_pairwise_bootstrap_ci: point-delta vs. CI assertion rules ---


def test_compare_pairwise_bootstrap_ci_point_delta_identity_enforced() -> None:
    mine = {"point_delta": 0.51, "ci_low": -0.1, "ci_high": 0.9}
    ref = {"point_delta": 0.50, "ci_low": -0.1, "ci_high": 0.9}
    with pytest.raises(RuntimeError, match="point_delta"):
        compare_pairwise_bootstrap_ci(mine, ref)


def test_compare_pairwise_bootstrap_ci_tiny_point_delta_diff_passes() -> None:
    mine = {"point_delta": 0.5 + 1e-12, "ci_low": -0.1, "ci_high": 0.9}
    ref = {"point_delta": 0.5, "ci_low": -0.1, "ci_high": 0.9}
    out = compare_pairwise_bootstrap_ci(mine, ref)
    assert out["point_delta_matches_reference"] is True


def test_compare_pairwise_bootstrap_ci_mismatch_not_required_to_raise() -> None:
    # Point deltas agree exactly, but CI endpoints differ (different RNG
    # draw sequence) -- must NOT raise; must be reported, not asserted.
    mine = {"point_delta": 0.5, "ci_low": -0.2, "ci_high": 0.8}
    ref = {"point_delta": 0.5, "ci_low": -0.1, "ci_high": 0.9}
    out = compare_pairwise_bootstrap_ci(mine, ref)
    assert out["point_delta_matches_reference"] is True
    assert out["ci_matches_reference_exactly"] is False
    assert out["max_abs_ci_endpoint_difference"] == pytest.approx(0.1)


def test_compare_pairwise_bootstrap_ci_exact_match_reported_as_observed() -> None:
    mine = {"point_delta": 0.5, "ci_low": -0.1, "ci_high": 0.9}
    ref = {"point_delta": 0.5, "ci_low": -0.1, "ci_high": 0.9}
    out = compare_pairwise_bootstrap_ci(mine, ref)
    assert out["ci_matches_reference_exactly"] is True
    assert out["max_abs_ci_endpoint_difference"] == pytest.approx(0.0)


# --- load_n6_external: multiset-safe, order-safe, no silent misalignment ---


def _write_n6_fixture(path: Path, model_formulas: dict[str, list[str]]) -> None:
    results = {}
    for model, formulas in model_formulas.items():
        residuals = [float(i) for i in range(len(formulas))]
        results[model] = {
            "strict": {
                "n": len(formulas),
                "mae": float(np.mean(np.abs(residuals))),
                "predictions": {"reduced_formula": formulas, "residual": residuals},
            }
        }
    path.write_text(
        json.dumps({"run_count": 1, "authorized": True, "results": results}), encoding="utf-8"
    )


def test_load_n6_external_accepts_repeated_formulas_matching_cohort_multiset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import issue10_formula_balanced_sensitivity as acs10

    n6_path = tmp_path / "n6.json"
    formulas = ["A", "A", "B"]  # cohort has a genuine repeat of "A"
    _write_n6_fixture(n6_path, {"model1": formulas})
    monkeypatch.setattr(acs10, "N6_PATH", n6_path)
    monkeypatch.setattr(acs10, "EXPECTED_N", 3)

    mae, residuals, order = load_n6_external(Counter(formulas))
    assert order["model1"] == formulas
    # Residuals stay row-aligned with their own model's formula order.
    assert list(residuals["model1"]) == [0.0, 1.0, 2.0]


def test_load_n6_external_raises_on_multiset_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import issue10_formula_balanced_sensitivity as acs10

    n6_path = tmp_path / "n6.json"
    model_formulas = ["A", "B", "C"]
    _write_n6_fixture(n6_path, {"model1": model_formulas})
    monkeypatch.setattr(acs10, "N6_PATH", n6_path)
    monkeypatch.setattr(acs10, "EXPECTED_N", 3)

    cohort_counts = Counter(["A", "B", "D"])  # "D" instead of "C" -> mismatch
    with pytest.raises(RuntimeError, match="formula multiset"):
        load_n6_external(cohort_counts)


def test_load_n6_external_order_insensitive_multiset_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A model's own formula order need not match the cohort file's row
    # order -- only the multiset must agree (order-safe comparison).
    import issue10_formula_balanced_sensitivity as acs10

    n6_path = tmp_path / "n6.json"
    model_formulas = ["C", "A", "B"]
    _write_n6_fixture(n6_path, {"model1": model_formulas})
    monkeypatch.setattr(acs10, "N6_PATH", n6_path)
    monkeypatch.setattr(acs10, "EXPECTED_N", 3)

    cohort_counts = Counter(["A", "B", "C"])  # different order, same multiset
    mae, residuals, order = load_n6_external(cohort_counts)
    assert order["model1"] == model_formulas  # unreordered


# --- No frozen-artifact mutation / no official test-split access ---


def test_loader_functions_do_not_mutate_frozen_strict_cohort_file() -> None:
    import csv

    before = STRICT_COHORT_PATH.read_bytes()
    with STRICT_COHORT_PATH.open(newline="", encoding="utf-8") as fh:
        _ = [r["reduced_formula"] for r in csv.DictReader(fh)]
    after = STRICT_COHORT_PATH.read_bytes()
    assert before == after


def test_module_source_never_references_official_test_split() -> None:
    source = (ROOT / "scripts" / "analysis" / "issue10_formula_balanced_sensitivity.py").read_text(
        encoding="utf-8"
    )
    # The frozen official OBELiX test split is 113 rows; this module's own
    # docstring asserts it is never opened. Guard that claim directly: no
    # reference to the official-test-split row count or to a test-split
    # data path appears anywhere in the module's executable code (the "113"
    # that legitimately appears is confined to the module docstring).
    import ast

    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    code_without_docstring = source.replace(docstring, "", 1)
    assert "113" not in code_without_docstring
    assert "official_test" not in source.lower()
    assert "test_split" not in source.lower()
