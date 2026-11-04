import pandas as pd

from sse_transfer.evaluation.n15_temperature_window import build_strict_cohort_custom_window


def _hargreaves_row(composition: str, source: str, temperature: float, target: float) -> dict:
    import math

    return {
        "composition": composition,
        "source": source,
        "temperature": temperature,
        "target": target,
        "log_target": math.log10(target),
    }


def test_narrower_window_is_subset_of_wider_window() -> None:
    hargreaves = pd.DataFrame(
        [
            _hargreaves_row("LiF", "10.1/a", 22.0, 1e-4),  # inside [20,30] and [15,35]
            _hargreaves_row("Li2O", "10.2/b", 32.0, 1e-5),  # inside [15,35] only
            _hargreaves_row("LiCl", "10.3/c", 5.0, 1e-6),  # outside both
        ]
    )
    obelix = pd.DataFrame({"Reduced Composition": [], "True Composition": [], "DOI": []})

    narrow = build_strict_cohort_custom_window(hargreaves, obelix, low_c=20.0, high_c=30.0)
    wide = build_strict_cohort_custom_window(hargreaves, obelix, low_c=15.0, high_c=35.0)

    narrow_formulas = set(narrow["reduced_formula"])
    wide_formulas = set(wide["reduced_formula"])
    assert narrow_formulas.issubset(wide_formulas)
    assert "LiF" in narrow_formulas
    assert "Li2O" not in narrow_formulas  # excluded by the narrower window
    assert "Li2O" in wide_formulas
    assert "LiCl" not in wide_formulas  # excluded by RT window entirely


def test_overlap_exclusion_still_applied_under_custom_window() -> None:
    hargreaves = pd.DataFrame([_hargreaves_row("LiFePO4", "10.1/shared", 25.0, 1e-4)])
    obelix = pd.DataFrame(
        {
            "Reduced Composition": ["LiFePO4"],
            "True Composition": ["LiFePO4"],
            "DOI": ["10.1/shared"],
        }
    )
    out = build_strict_cohort_custom_window(hargreaves, obelix, low_c=20.0, high_c=30.0)
    assert out.empty


def test_deterministic_repeat_call_identical() -> None:
    hargreaves = pd.DataFrame(
        [
            _hargreaves_row("LiF", "10.1/a", 22.0, 1e-4),
            _hargreaves_row("LiF", "10.1/b", 24.0, 2e-4),
        ]
    )
    obelix = pd.DataFrame({"Reduced Composition": [], "True Composition": [], "DOI": []})
    out1 = build_strict_cohort_custom_window(hargreaves, obelix, low_c=20.0, high_c=30.0)
    out2 = build_strict_cohort_custom_window(hargreaves, obelix, low_c=20.0, high_c=30.0)
    pd.testing.assert_frame_equal(out1, out2)
