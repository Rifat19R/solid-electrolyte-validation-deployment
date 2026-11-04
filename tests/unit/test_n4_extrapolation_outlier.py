from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "transport_analysis"))

from n4_internal_validation import _extrapolation_outlier_section  # noqa: E402


def _regime(pooled_mae: float, macro: float, median: float, per_family: dict) -> dict:
    return {
        "pooled_mae": pooled_mae,
        "pooled_mae_over_major_family_rows": pooled_mae,
        "macro_family_mae": macro,
        "median_family_mae": median,
        "per_family_mae": per_family,
    }


def test_no_outlier_reports_clean_section() -> None:
    manifest = {
        "results": {
            "modelA": {
                "A_random_5fold_cv": _regime(1.0, 1.0, 1.0, {"fam1": {"mae": 1.0, "n": 10}}),
                "B_family_group_kfold": _regime(1.1, 1.1, 1.1, {"fam1": {"mae": 1.1, "n": 10}}),
                "C_lomfo": _regime(1.2, 1.2, 1.2, {"fam1": {"mae": 1.2, "n": 10}}),
            }
        }
    }
    section = _extrapolation_outlier_section(manifest)
    assert "No (model, regime) pooled MAE exceeded" in section


def test_detects_and_attributes_outlier_to_driving_family() -> None:
    manifest = {
        "results": {
            "ridge": {
                "A_random_5fold_cv": _regime(1.2, 2.0, 1.5, {"fam1": {"mae": 1.5, "n": 100}}),
                "B_family_group_kfold": _regime(
                    500.0,
                    80.0,
                    1.8,
                    {
                        "fam1": {"mae": 1.8, "n": 90},
                        "NASICON": {"mae": 1300.0, "n": 130},
                    },
                ),
                "C_lomfo": _regime(2.0, 2.0, 2.0, {"fam1": {"mae": 2.0, "n": 100}}),
            }
        }
    }
    section = _extrapolation_outlier_section(manifest)
    assert "ridge / B_family_group_kfold" in section
    assert "NASICON" in section
    assert "fam1" not in section.split("Driven by:")[1].split(".\n")[0]


def test_does_not_flag_a_mild_regime_difference() -> None:
    manifest = {
        "results": {
            "catboost": {
                "A_random_5fold_cv": _regime(0.7, 1.7, 1.5, {"fam1": {"mae": 0.7, "n": 100}}),
                "B_family_group_kfold": _regime(1.6, 2.2, 1.9, {"fam1": {"mae": 1.6, "n": 100}}),
                "C_lomfo": _regime(1.6, 1.6, 1.6, {"fam1": {"mae": 1.6, "n": 100}}),
            }
        }
    }
    section = _extrapolation_outlier_section(manifest)
    assert "No (model, regime) pooled MAE exceeded" in section
