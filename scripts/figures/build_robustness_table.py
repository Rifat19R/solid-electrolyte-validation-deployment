"""Main-text tables, read programmatically from authoritative artifacts.

All inputs are under artifacts/authoritative/ (A = that directory):
table1_cohort_definitions.csv  <- data/manifests/external_cohort.json,
    A/n4b_nested_internal_validation_corrected.json (dev/test n),
    A/n15_artifacts/n15_source_disjoint.json
table2_external_performance_ci.csv  <- A/n15_artifacts/n15_external_uncertainty.json
table3_regime_comparison_primary_and_stable.csv
    <- A/n7b_regime_utility_nested_corrected.json,
       A/n15_artifacts/n15_stable_model_sensitivity.json
si_table1_robustness_summary.csv <- A/n15_artifacts/n15_source_disjoint.json,
    A/n15_artifacts/n15_temperature_sensitivity.json
"""

from __future__ import annotations

import csv
import json

from palette_import import AUTHORITATIVE_DIR, MODEL_ORDER, REPO_ROOT, TABLES_DIR


def _load(name: str) -> dict:
    return json.loads((AUTHORITATIVE_DIR / name).read_text())


def build_table1() -> None:
    manifest = json.loads((REPO_ROOT / "data" / "manifests" / "external_cohort.json").read_text())
    n4b = _load("n4b_nested_internal_validation_corrected.json")
    n15sd = _load("n15_artifacts/n15_source_disjoint.json")
    sc = manifest["stage_counts"]

    rows = [
        {
            "cohort": "OBELiX development population",
            "n": n4b["obelix_dev_population_n"],
            "role": "internal train+validation, frozen",
            "independence": "n/a (source dataset)",
        },
        {
            "cohort": "OBELiX official test split",
            "n": n4b["obelix_test_untouched_row_count"],
            "role": "never opened at any point (old project)",
            "independence": "n/a (not used in this study)",
        },
        {
            "cohort": "Hargreaves raw literature database",
            "n": sc["raw_rows"],
            "role": "external source, 214 literature sources",
            "independence": "not restricted",
        },
        {
            "cohort": "Hargreaves room-temperature window (15-35C)",
            "n": sc["rt_window_rows"],
            "role": "predeclared RT window, pre-exclusion",
            "independence": "not restricted",
        },
        {
            "cohort": "Strict external cohort (primary)",
            "n": sc["strict_independent_rows"],
            "role": "primary external transfer target (Table 2, Fig. 3)",
            "independence": "composition-disjoint from OBELiX; not source-independent",
        },
        {
            "cohort": "Source-disjoint subset of strict cohort",
            "n": n15sd["n_source_disjoint"],
            "role": "post-review sensitivity cohort (N15 analysis 1, Table 4)",
            "independence": (
                f"composition-disjoint AND source-disjoint "
                f"({n15sd['n_excluded_for_source_overlap']} of "
                f"{n15sd['n_strict_cohort']} strict rows excluded, "
                f"{n15sd['fraction_excluded'] * 100:.1f}%)"
            ),
        },
        {
            "cohort": "Relaxed external cohort (companion sensitivity)",
            "n": sc["relaxed_independent_rows"],
            "role": "companion robustness cohort (SI-9); strict subset of relaxed",
            "independence": "composition-disjoint on literature-source-DOI-exact matches only",
        },
    ]

    out_path = TABLES_DIR / "table1_cohort_definitions.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


def build_table2() -> None:
    n15 = _load("n15_artifacts/n15_external_uncertainty.json")
    strict = n15["hargreaves_strict"]
    per_model = strict["per_model"]

    rows = []
    for model in MODEL_ORDER:
        d = per_model[model]
        rows.append(
            {
                "model": model,
                "point_mae": round(d["point_mae"], 4),
                "ci_low_95": round(d["ci_low"], 4),
                "ci_high_95": round(d["ci_high"], 4),
                "probability_rank_1": round(d["probability_rank_1"], 4),
                "n": strict["n_rows"],
            }
        )
    rows.sort(key=lambda r: r["point_mae"])

    out_path = TABLES_DIR / "table2_external_performance_ci.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows, n_boot={strict['n_boot']}, seed={strict['seed']})")


def build_table3() -> None:
    n7b = _load("n7b_regime_utility_nested_corrected.json")
    n15 = _load("n15_artifacts/n15_stable_model_sensitivity.json")
    primary = n7b["per_regime_summary"]
    stable = n15["per_regime_summary"]

    rows = []
    for regime in ["A_random_5fold_cv", "B_family_group_kfold", "C_lomfo"]:
        p = primary[regime]
        s = stable[regime]
        rows.append(
            {
                "regime": regime,
                "n_models_primary": p["n_models"],
                "median_abs_mae_gap_primary": round(p["median_abs_mae_gap"], 4),
                "spearman_rho_primary": round(p["spearman_rho"], 4),
                "kendall_tau_primary": round(p["kendall_tau"], 4),
                "top_model_selection_regret_primary": round(p["top_model_selection_regret"], 4),
                "n_models_stable": s["n_models"],
                "median_abs_mae_gap_stable": round(s["median_abs_mae_gap"], 4),
                "spearman_rho_stable": round(s["spearman_rho"], 4),
                "kendall_tau_stable": round(s["kendall_tau"], 4),
                "top_model_selection_regret_stable": round(s["top_model_selection_regret"], 4),
            }
        )

    out_path = TABLES_DIR / "table3_regime_comparison_primary_and_stable.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


def build_table4() -> None:
    n15sd = _load("n15_artifacts/n15_source_disjoint.json")
    n15temp = _load("n15_artifacts/n15_temperature_sensitivity.json")

    rows = []
    for model in MODEL_ORDER:
        sd = n15sd["per_model"][model]
        temp = n15temp["per_model"][model]
        rows.append(
            {
                "model": model,
                "mae_strict_n204": round(sd["full_strict"]["mae"], 4),
                "mae_source_disjoint_n153": round(sd["source_disjoint"]["mae"], 4),
                "mae_delta_source_disjoint": round(
                    sd["mae_delta_source_disjoint_minus_full_strict"], 4
                ),
                "mae_window_20_30_n197": round(temp["window_20_30"]["mae"], 4),
                "mae_delta_window_only_same_formula_subset": round(
                    temp["mae_delta_window_change_only"], 4
                ),
                "mae_delta_window_20_30_vs_full_strict": round(
                    temp["mae_delta_vs_full_strict_204"], 4
                ),
            }
        )
    rows.sort(key=lambda r: r["mae_strict_n204"])

    out_path = TABLES_DIR / "si_table1_robustness_summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Wrote {out_path} ({len(rows)} rows); n_source_disjoint={n15sd['n_source_disjoint']}, "
        f"n_window_20_30={n15temp['n_window_20_30']}"
    )


if __name__ == "__main__":
    build_table1()
    build_table2()
    build_table3()
    build_table4()
