#!/usr/bin/env python3
"""Corrective 01: re-run the pure-aggregation scripts (no model fitting) that
depend only on N4B/N6/N20B/regime-E against the corrected artifacts, writing
to new corrective output paths. Each target's own `main()` logic is reused
unmodified; only its module-level input/output path constants are
monkeypatched. See reports/remediation/corrective01_impact_map.md section 3.

Run with: python scripts/analysis/corrective01_merge_aggregators_batch1.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "transport_analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

CORR = ROOT / "artifacts" / "authoritative"
RERUN = ROOT / "artifacts" / "rerun"  # outputs; the authoritative record is never rewritten
N4B_CORRECTED = CORR / "n4b_nested_internal_validation_corrected.json"
N6_CORRECTED = CORR / "n6_external_transfer_corrected.json"
N20B_CORRECTED = CORR / "n20b_formula_group_kfold_corrected.json"
E_CORRECTED = CORR / "issue08b_chemical_system_groupkfold_corrected.json"
OUT_REPORTS = ROOT / "reports" / "rerun" / "corrective01_reports"
OUT_REPORTS.mkdir(parents=True, exist_ok=True)

for p in [N4B_CORRECTED, N6_CORRECTED, N20B_CORRECTED, E_CORRECTED]:
    assert p.exists(), f"missing {p}"


def run(
    mod_name: str,
    patches: dict[str, Path],
    out_json_attr: str,
    out_report_attr: str | None,
    out_name: str,
    out_report_name: str | None,
) -> None:
    print(f"\n=== {mod_name} ===")
    mod = importlib.import_module(mod_name)
    for attr, val in patches.items():
        setattr(mod, attr, val)
    setattr(mod, out_json_attr, RERUN / out_name)
    if out_report_attr and out_report_name:
        setattr(mod, out_report_attr, OUT_REPORTS / out_report_name)
    mod.main()
    print(f"[corrective01] wrote {getattr(mod, out_json_attr).relative_to(ROOT)}")


# n7b: primary regime-utility A/B/C, 8-model + 6-model
run(
    "n7b_regime_utility_nested",
    {"N4B_PATH": N4B_CORRECTED, "N6_PATH": N6_CORRECTED},
    "OUTPUT_PATH",
    "REPORT_PATH",
    "n7b_regime_utility_nested_corrected.json",
    "n7b_regime_utility_nested_corrected.md",
)

# n8: applicability domain / AURC
run(
    "n8_applicability_domain",
    {"N6_PATH": N6_CORRECTED},
    "OUTPUT_PATH",
    "REPORT_PATH",
    "n8_applicability_domain_corrected.json",
    "n8_applicability_domain_corrected.md",
)

# n9b: relaxed-cohort sensitivity
run(
    "n9b_relaxed_sensitivity_nested",
    {"N4B_PATH": N4B_CORRECTED, "N6_PATH": N6_CORRECTED},
    "OUTPUT_PATH",
    "REPORT_PATH",
    "n9b_relaxed_sensitivity_nested_corrected.json",
    "n9b_relaxed_sensitivity_nested_corrected.md",
)

# n21c: primary external context
run(
    "n21c_primary_external_context",
    {"N6_ARTIFACT": N6_CORRECTED},
    "OUTPUT_JSON",
    "REPORT_MD",
    "n21c_primary_external_context_corrected.json",
    "n21c_primary_external_context_corrected.md",
)

# n21d: AD benchmark closure (also reads reviewer_family_audit.json, unaffected, unchanged)
run(
    "n21d_ad_benchmark_closure",
    {"N6_PATH": N6_CORRECTED},
    "OUTPUT_JSON",
    "REPORT_MD",
    "n21d_ad_benchmark_closure_corrected.json",
    "n21d_ad_benchmark_closure_corrected.md",
)

# acs08b_regime_utility: regime-E utility stats
run(
    "issue08b_regime_utility",
    {"E_PATH": E_CORRECTED, "N4B_PATH": N4B_CORRECTED, "N6_PATH": N6_CORRECTED},
    "OUTPUT_PATH",
    "REPORT_PATH",
    "issue08b_regime_utility_corrected.json",
    "issue08b_regime_utility_corrected.md",
)

# acs03: common-support LOMFO
run(
    "issue03_common_support_lomfo",
    {"N4B_PATH": N4B_CORRECTED, "N6_PATH": N6_CORRECTED},
    "OUTPUT_PATH",
    "REPORT_PATH",
    "issue03_common_support_lomfo_corrected.json",
    "issue03_common_support_lomfo_corrected.md",
)

# acs09a: missing-family sensitivity Variant A
run(
    "issue09a_missing_family_excluded",
    {"N4B_PATH": N4B_CORRECTED, "N6_PATH": N6_CORRECTED},
    "OUTPUT_PATH",
    "REPORT_PATH",
    "issue09a_missing_family_excluded_corrected.json",
    "issue09a_missing_family_excluded_corrected.md",
)

print("\n=== batch1 done ===")
