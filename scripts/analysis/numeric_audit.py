#!/usr/bin/env python3
"""Reviewer remediation numeric audit: verify every quantitative claim
added to manuscript/source/ by this remediation pass (Issues 01, 03,
04, 05, 06, 08a, 08b, 09) against its source artifact, by recomputing the
expected formatted string directly from the frozen artifact and checking
it appears literally in the manuscript text. Same methodology as
the removed figures_v16 generation's n16_numeric_audit.py through
the removed figures_v22 generation's n22_numeric_audit.py, which remain
authoritative for pre-existing (pre-remediation) claims; this script
only covers claims introduced by this pass.

Run with: python scripts/analysis/numeric_audit.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "src"))
from sse_transfer.utils.manuscript_text import read_plain  # noqa: E402

MAIN = read_plain(ROOT / "manuscript" / "source" / "00_main.md")
SI = read_plain(ROOT / "manuscript" / "source" / "01_si.md")
FULL_TEXT = MAIN + "\n" + SI

missing: list[str] = []
n_checks = 0


def load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text())


def check(label: str, expected_substring: str, *, where: str = "either") -> None:
    global n_checks
    n_checks += 1
    text = {"main": MAIN, "si": SI, "either": FULL_TEXT}[where]
    if expected_substring not in text:
        missing.append(f"{label}: {expected_substring!r} (searched: {where})")


# --- Issue 08a: wording fix -- assert the banned phrase is fully gone ---
n_checks += 1
if "leak-free" in FULL_TEXT:
    missing.append("issue08a: banned phrase 'leak-free' still present in manuscript text")

# --- Issue 03: common-support LOMFO (corrected 8-model view is authoritative
# post Supplementary Note 27; historical artifact kept only for the n=396 constant) ---
issue03 = load("authoritative/issue03_common_support_lomfo_corrected.json")
issue03c = load("authoritative/issue03_common_support_lomfo_corrected.json")
cs = issue03c["common_support_regime_summary"]
check("issue03 common-n", "396-row")
check(
    "issue03 A common gap (corrected)",
    f"{cs['A_random_5fold_cv']['median_abs_mae_gap']:.2f}",
    where="main",
)
check(
    "issue03 B common gap (corrected)",
    f"{cs['B_family_group_kfold']['median_abs_mae_gap']:.2f}",
    where="main",
)
check("issue03 common CI text (corrected)", "[-0.2952, 0.3655]", where="si")

# --- Issue 04: training-size-matched control ---
issue04 = load("authoritative/issue04_size_matched_control_corrected.json")
ridge = issue04["results"]["ridge"]
shallow_mlp = issue04["results"]["shallow_mlp"]
check("issue04 ridge B blowup", f"{ridge['regime_B_family_group_kfold_pooled_mae']:.1f}")
check(
    "issue04 shallow_mlp B blowup",
    f"{shallow_mlp['regime_B_family_group_kfold_pooled_mae']:.1f}",
)
check("issue04 ridge control seed values", "1.41, 1.30, 1.28", where="main")
check("issue04 mlp control seed values", "1.15, 1.15, 1.14", where="main")

# --- Issue 05: fold-local shift distance (drop counts identical) ---
issue05 = load("authoritative/issue05_fold_local_shift_distance.json")
b_dropped_acs05 = [f["n_dropped"] for f in issue05["per_fold_dims_dropped"]["B_family_group_kfold"]]
if b_dropped_acs05 != [26, 22, 23, 22, 22]:
    missing.append(f"issue05: expected B dropped [26,22,23,22,22], got {b_dropped_acs05}")
n_checks += 1
check("issue05 B dropped counts in SI", "26, 22, 23, 22, 22", where="si")
check("issue05 closest regime unchanged", "LOMFO", where="si")

# --- Issue 06: DOI-cluster bootstrap ---
issue06 = load("authoritative/issue06_doi_cluster_bootstrap_corrected.json")
n_clusters = issue06["n_clusters"]
max_cluster = issue06["max_cluster_size"]
check("issue06 n_clusters", f"{n_clusters} source clusters")
check("issue06 max cluster size", f"largest = {max_cluster} rows")
if n_clusters != 85 or max_cluster != 10:
    missing.append(f"issue06: expected 85 clusters / max 10, got {n_clusters}/{max_cluster}")
n_checks += 1

# --- Issue 09 Variant A ---
issue09a = load("authoritative/issue09a_missing_family_excluded_corrected.json")
excl_n = issue09a["n_excluding_missing_A_B"]
if excl_n != 431:
    missing.append(f"issue09a: expected 431 rows excluding __MISSING__, got {excl_n}")
n_checks += 1
check("issue09a n rows", "18 of 449", where="si")

# --- Issue 08b: chemical-system-GroupKFold (regime E) ---
issue08b = load("authoritative/issue08b_chemical_system_groupkfold_corrected.json")
if issue08b["n_unique_chemical_systems"] != 153:
    missing.append(
        f"issue08b: expected 153 chemical systems, got {issue08b['n_unique_chemical_systems']}"
    )
n_checks += 1
check("issue08b n chemical systems", "153 unique chemical systems", where="si")
issue08b_ru = load("authoritative/issue08b_regime_utility_corrected.json")
e_gap = issue08b_ru["e_regime_summary"]["median_abs_mae_gap"]
a_gap = issue08b_ru["a_regime_summary"]["median_abs_mae_gap"]
check("issue08b E median gap (corrected)", f"{e_gap:.4f}", where="si")
check("issue08b A median gap (E context)", f"{a_gap:.4f}", where="si")
delta = issue08b_ru["pairwise_bootstrap_ci_E_minus_A"]["median_abs_mae_gap"]
check(
    "issue08b E-A delta CI (corrected)",
    f"[{delta['ci_low']:.4f}, {delta['ci_high']:.4f}]",
    where="si",
)

# --- Issue 09 Variant B: __MISSING__ as singleton groups (regime B'),
# corrected pretrained_element_embedding fragment; other 7 models unaffected ---
issue09b = load("authoritative/issue09b_missing_singleton_groupkfold_corrected.json")
ridge_bp = issue09b["results"]["ridge"]["pooled_mae"]
mlp_bp = issue09b["results"]["shallow_mlp"]["pooled_mae"]
check("issue09b ridge B' MAE", f"{ridge_bp:.3f}", where="si")
check("issue09b shallow_mlp B' MAE", f"{mlp_bp:.3f}", where="si")
issue09b_ru = load("authoritative/issue09b_regime_utility_corrected.json")
bp_rho = issue09b_ru["bprime_regime_summary"]["spearman_rho"]
check("issue09b B' spearman rho (corrected)", f"{bp_rho:.4f}", where="si")

# --- Issue 10: formula-balanced external-estimand sensitivity ---
issue10 = load("authoritative/issue10_formula_balanced_sensitivity_corrected.json")
if not issue10["premise_holds_zero_repeated_formulas"]:
    missing.append("issue10: expected zero repeated-formula groups in the strict cohort")
n_checks += 1
if issue10["n_unique_reduced_formulas"] != 204 or issue10["n_repeated_formula_groups"] != 0:
    missing.append(
        "issue10: expected 204 unique formulas / 0 repeated groups, got "
        f"{issue10['n_unique_reduced_formulas']}/{issue10['n_repeated_formula_groups']}"
    )
n_checks += 1
if (
    not issue10["ranking_identical"]
    or not issue10["regime_utility_summaries_identical_to_row_weighted"]
):
    missing.append(
        "issue10: expected formula-balanced ranking/regime-utility identical to row-weighted"
    )
n_checks += 1
check("issue10 zero repeated groups", "zero repeated-formula groups", where="either")
check("issue10 note reference", "Supplementary Note 36", where="main")
check("issue10 SI premise sentence", "zero repeated-formula groups", where="si")

# --- Corrective 01: pretrained_element_embedding feature-lookup bugfix --
# hard-checks of the CORRECTED numbers directly against the manuscript,
# so no stale historical broken-model number can pass this audit merely
# because the old artifact still exists on disk.
CORR = "authoritative/"
n4b_c = load(CORR + "n4b_nested_internal_validation_corrected.json")["results"][
    "pretrained_element_embedding"
]
check(
    "corrective01 A internal MAE", f"{n4b_c['A_random_5fold_cv']['pooled_mae']:.2f}", where="main"
)
check(
    "corrective01 B internal MAE", f"{n4b_c['B_family_group_kfold']['pooled_mae']:.4f}", where="si"
)
check(
    "corrective01 C internal MAE",
    f"{n4b_c['C_lomfo']['pooled_mae_over_major_family_rows']:.4f}",
    where="si",
)
n20b_c = load(CORR + "n20b_formula_group_kfold_corrected.json")
check(
    "corrective01 D internal MAE",
    f"{n20b_c['results']['pretrained_element_embedding']['pooled_mae']:.4f}",
    where="si",
)
e8b_c = load(CORR + "issue08b_chemical_system_groupkfold_corrected.json")
e_pem_mae = e8b_c["results"]["pretrained_element_embedding"]["pooled_mae"]
if not (1.52 < e_pem_mae < 1.53):
    missing.append(f"corrective01 E internal MAE: expected ~1.5213, got {e_pem_mae:.4f}")
n_checks += 1
n6_c = load(CORR + "n6_external_transfer_corrected.json")
ext_strict = n6_c["results"]["pretrained_element_embedding"]["strict"]
check("corrective01 external strict MAE", f"{ext_strict['mae']:.2f}", where="main")
check(
    "corrective01 external strict Spearman",
    f"{ext_strict['spearman_rho']:.2f}",
    where="main",
)
n7b_c = load(CORR + "n7b_regime_utility_nested_corrected.json")
b8 = n7b_c["per_regime_summary"]["B_family_group_kfold"]
c8 = n7b_c["per_regime_summary"]["C_lomfo"]
check("corrective01 8-model B median gap", f"{b8['median_abs_mae_gap']:.4f}", where="main")
check("corrective01 8-model C spearman rho", f"{c8['spearman_rho']:.4f}", where="main")
n15_stable_c = load(CORR + "n15_artifacts/n15_stable_model_sensitivity.json")
b6 = n15_stable_c["per_regime_summary"]["B_family_group_kfold"]
check("corrective01 6-model B median gap", f"{b6['median_abs_mae_gap']:.4f}", where="main")
glass_split_c = load(CORR + "phase1_glass_split_corrected.json")
pem_subsets = glass_split_c["per_model_subset_metrics"]
check(
    "corrective01 SEEN MAE",
    f"{pem_subsets['SEEN']['pretrained_element_embedding']['mae']:.2f}",
    where="main",
)
check(
    "corrective01 UNSEEN_GLASS MAE",
    f"{pem_subsets['UNSEEN_GLASS']['pretrained_element_embedding']['mae']:.2f}",
    where="main",
)
check(
    "corrective01 UNSEEN_CRYSTALLINE MAE",
    f"{pem_subsets['UNSEEN_CRYSTALLINE']['pretrained_element_embedding']['mae']:.2f}",
    where="main",
)
n8_c = load(CORR + "n8_applicability_domain_corrected.json")
aurc_pem = n8_c["per_model"]["pretrained_element_embedding"]["risk_coverage_mae"]
check(
    "corrective01 AURC coarse-grid MAE@25%",
    f"{[r for r in aurc_pem if r['retained_fraction'] == 0.25][0]['mae']:.4f}",
    where="si",
)
n21d_c = load(CORR + "n21d_ad_benchmark_closure_corrected.json")
check(
    "corrective01 cross_model_dispersion mean AURC",
    f"{n21d_c['summary']['cross_model_dispersion']['mean_aurc_across_8_models']:.4f}",
    where="si",
)
issue10_c = load(CORR + "issue10_formula_balanced_sensitivity_corrected.json")
if not (
    issue10_c["ranking_identical"]
    and issue10_c["regime_utility_summaries_identical_to_row_weighted"]
):
    missing.append("corrective01 issue10: expected corrected formula-balanced == row-weighted")
n_checks += 1
issue09b_c = load(CORR + "issue09b_missing_singleton_groupkfold_corrected.json")
bp_pem = issue09b_c["results"]["pretrained_element_embedding"]["pooled_mae"]
check("corrective01 issue09b B' pretrained_element_embedding MAE", f"{bp_pem:.3f}", where="si")

print(f"{n_checks} checks run, {len(missing)} missing")
if missing:
    for m in missing:
        print("MISSING:", m)
    sys.exit(1)
print("ALL PASS")
