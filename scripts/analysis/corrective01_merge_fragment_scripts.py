#!/usr/bin/env python3
"""Corrective 01: re-run each fragment-producing script's OWN `merge_fragments()`
function (byte-identical merge logic to the historical pipeline -- not a
reimplementation) against the current fragment directories (7 historical +
1 corrected `pretrained_element_embedding` fragment each), but monkeypatched
to write to a NEW corrective output path/report instead of the historical
locked artifact. Historical `ARTIFACT_JSON`/`REPORT_MD` files for n4b, n20b,
acs08b, n21, and acs04 are never touched by this script (verified by the
caller via SHA-256 diff before/after).

(N6 is handled separately by `corrective01_merge_n6.py`, since N6's own
`merge_fragments` has one-shot-lock semantics that don't apply cleanly to a
fresh output path -- see that script's docstring.)

Run with: python scripts/analysis/corrective01_merge_fragment_scripts.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "transport_analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

OUT_DIR = ROOT / "artifacts" / "rerun"
OUT_REPORTS_DIR = ROOT / "reports" / "rerun" / "corrective01_reports"

TARGETS = [
    # (module_name, new_artifact_filename, new_report_filename)
    (
        "n4b_nested_internal_validation",
        "n4b_nested_internal_validation_corrected.json",
        "n4b_nested_internal_validation_corrected.md",
    ),
    (
        "n20b_formula_group_kfold",
        "n20b_formula_group_kfold_corrected.json",
        "n20b_formula_group_kfold_corrected.md",
    ),
    (
        "issue08b_chemical_system_groupkfold",
        "issue08b_chemical_system_groupkfold_corrected.json",
        "issue08b_chemical_system_groupkfold_corrected.md",
    ),
    (
        "n21_repeated_cv_sensitivity",
        "n21_repeated_cv_sensitivity_corrected.json",
        "n21_repeated_cv_sensitivity_corrected.md",
    ),
    (
        "issue04_size_matched_control",
        "issue04_size_matched_control_corrected.json",
        "issue04_size_matched_control_corrected.md",
    ),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for mod_name, art_name, rep_name in TARGETS:
        print(f"\n=== merging {mod_name} ===")
        mod = importlib.import_module(mod_name)
        mod.ARTIFACT_JSON = OUT_DIR / art_name
        mod.REPORT_MD = OUT_REPORTS_DIR / rep_name
        mod.merge_fragments()
        print(f"[corrective01] wrote {mod.ARTIFACT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
