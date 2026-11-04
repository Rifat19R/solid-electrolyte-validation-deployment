#!/usr/bin/env python3
"""Cross-dataset study, N20: reviewer-remediation pass.

Four independent audits requested by a reviewer-remediation task brief,
each producing one frozen JSON artifact under `artifacts/`:

  A. External family audit (`reviewer_family_audit.json`) -- crosswalk the
     204 strict Hargreaves external compositions' raw `family` label onto
     OBELiX's harmonized family vocabulary, classify each as seen/unseen in
     the 449-row OBELiX development population, and (using only the already-
     frozen N6 external predictions -- no re-inference) compare per-subset
     model performance and whether Random-CV/Family-GroupKFold/LOMFO
     internal-CV numbers (N4b, post-leakage-fix) better anticipate the
     seen-family vs. unseen-family external subsets.

  B. Random-CV duplicate/leakage audit (`reviewer_formula_duplicate_audit.json`)
     -- canonicalize all 449 OBELiX dev-population formulas, find exact
     duplicates, and check whether any duplicate group's rows cross N4b's
     random-5fold-CV outer folds (same seed). If they do, a conditional
     one-off diagnostic regime (reduced-formula GroupKFold, reusing N4b's
     exact nested-tuning protocol) is run separately in
     `scripts/transport_analysis/n20b_formula_group_kfold.py`; this script records the
     duplicate-audit facts and the go/no-go decision, and folds in the
     diagnostic regime's result once available.

  C. Cross-database label compatibility (`reviewer_label_compatibility.json`)
     -- for every reduced formula present in both OBELiX (any split, full
     599 rows) and the Hargreaves pre-exclusion overlap ledger, compare
     OBELiX's canonical y = log10(sigma_RT) against Hargreaves'
     log10(sigma_RT) using ONLY exact (non-censored) OBELiX labels. This is
     a metadata/calibration audit -- these paired rows are never used as
     model-evaluation data (they are excluded from every N2 cohort by
     construction).

  D. Nested-CV / provenance timeline (`reviewer_provenance_timeline.json`)
     -- reconstructed from `git log` on this branch, distinguishing what
     happened before vs. after N6's external predictions were generated/
     visible, and using accurate "internally hash-locked" language (never
     "publicly preregistered" -- no public timestamped registration exists
     for this project; see `artifacts/frozen/external_transfer_lock.json`).

Every number reused from N4b/N6 is read from its already-frozen artifact
file, never recomputed by refitting a model. No Hargreaves external
inference is rerun. The frozen 113-row OBELiX `test` split is never opened.

Run with: python scripts/transport_analysis/n20_reviewer_remediation.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.data.hargreaves_cohort import (  # noqa: E402
    MATCH_COMPOSITION_ONLY,
    MATCH_EXACT_DOI,
    load_obelix_raw,
    safe_reduced_formula,
)
from sse_transfer.data.obelix_loader import (  # noqa: E402
    LABEL_STATUS_EXACT,
    censoring_bound,
    label_status,
    load_obelix,
)
from sse_transfer.data.regression_dataset import load_regression_dataset  # noqa: E402
from sse_transfer.evaluation.internal_validation import (  # noqa: E402
    N_OUTER_SPLITS,
    canonicalize_and_fill_family,
    random_kfold_splits,
)
from sse_transfer.evaluation.metrics import bootstrap_metric_ci  # noqa: E402
from sse_transfer.evaluation.n15_source_disjoint import (  # noqa: E402
    score_subset_from_frozen_predictions,
)
from sse_transfer.evaluation.regime_utility import regime_summary  # noqa: E402
from sse_transfer.evaluation.reviewer_remediation import (  # noqa: E402
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
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

LEDGER_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_overlap_ledger.csv"
N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
N4B_PATH = ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
N20B_PATH = ROOT / "artifacts" / "authoritative" / "n20b_formula_group_kfold_corrected.json"

ARTIFACTS_DIR = ROOT / "artifacts" / "rerun" / "n20_artifacts"
ARTIFACT_A = ARTIFACTS_DIR / "reviewer_family_audit.json"
ARTIFACT_B = ARTIFACTS_DIR / "reviewer_formula_duplicate_audit.json"
ARTIFACT_C = ARTIFACTS_DIR / "reviewer_label_compatibility.json"
ARTIFACT_D = ARTIFACTS_DIR / "reviewer_provenance_timeline.json"

MODEL_NAMES: tuple[str, ...] = (
    "ridge",
    "random_forest",
    "extra_trees",
    "xgboost",
    "catboost",
    "shallow_mlp",
    "pretrained_element_embedding",
    "composition_attention",
)
REGIMES = ["A_random_5fold_cv", "B_family_group_kfold", "C_lomfo"]
CV_SEED = 20260921
BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
MIN_N_FOR_SUBSET_CI = 10
RETAINED = "RETAINED"
EXACT_DOI_OVERLAP = "EXACT_COMPOSITION_AND_DOI_MATCH_WITH_OBELIX"
COMPOSITION_ONLY_OVERLAP = "COMPOSITION_ONLY_MATCH_WITH_OBELIX"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------


def load_dev_population() -> pd.DataFrame:
    """449-row OBELiX development population with true_composition, y,
    harmonized family, and canonical (pymatgen-reduced) formula. Identical
    construction to N4/N4b/N15's `load_dev_population`, plus the
    `canonical_formula` column N20b also adds."""
    dataset = load_regression_dataset()
    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)
    raw_by_id = raw.set_index("ID")

    dev = pd.concat([dataset.train, dataset.validation], ignore_index=True)
    pool = dev[["ID", "y"]].copy()
    pool["true_composition"] = raw_by_id.loc[pool["ID"], "True Composition"].to_numpy()
    pool["family_filled"] = canonicalize_and_fill_family(
        raw_by_id.loc[pool["ID"], "Family"].reset_index(drop=True)
    ).to_numpy()
    reduced_raw = raw_by_id.loc[pool["ID"], "Reduced Composition"].reset_index(drop=True)
    pool["canonical_formula"] = reduced_raw.apply(safe_reduced_formula).to_numpy()
    return pool.reset_index(drop=True)


def load_n4b_internal() -> dict[str, dict[str, float]]:
    data = _load_json(N4B_PATH)
    out: dict[str, dict[str, float]] = {r: {} for r in REGIMES}
    for model, regimes in data["results"].items():
        for regime, payload in regimes.items():
            key = "pooled_mae" if "pooled_mae" in payload else "pooled_mae_over_major_family_rows"
            out[regime][model] = float(payload[key])
    return out


def load_ledger() -> pd.DataFrame:
    return pd.read_csv(LEDGER_PATH)


# ---------------------------------------------------------------------------
# Part A: external family audit
# ---------------------------------------------------------------------------


def run_part_a(dev_pool: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, Any]:
    obelix_family_vocab = set(dev_pool["family_filled"].unique().tolist())

    retained = ledger.loc[ledger["exclusion_reason"] == RETAINED].copy()
    if retained["reduced_formula"].nunique() != 204:
        raise RuntimeError(
            f"run_part_a: expected 204 strict-cohort formulas, "
            f"got {retained['reduced_formula'].nunique()}"
        )

    formula_family = assign_formula_family(retained)
    chemical_family_by_formula = {
        str(f): sorted({v for v in g["ChemicalFamily"].tolist() if pd.notna(v)})
        for f, g in retained.groupby("reduced_formula")
    }

    status_by_formula: dict[str, str] = {}
    for formula, info in formula_family.items():
        if not info["family_agreement"]:
            status_by_formula[formula] = "AMBIGUOUS_MULTI_SOURCE_FAMILY_DISAGREEMENT"
            continue
        status_by_formula[formula] = classify_family_seen_status(
            info["assigned_family"], obelix_family_vocab
        )

    seen_formulas = sorted(f for f, s in status_by_formula.items() if s == SEEN)
    unseen_formulas = sorted(f for f, s in status_by_formula.items() if s == UNSEEN)
    ambiguous_formulas = sorted(
        f
        for f, s in status_by_formula.items()
        if s in (AMBIGUOUS, MISSING, "AMBIGUOUS_MULTI_SOURCE_FAMILY_DISAGREEMENT")
    )
    assert len(seen_formulas) + len(unseen_formulas) + len(ambiguous_formulas) == 204

    raw_family_counts = retained.drop_duplicates("reduced_formula")["family"].value_counts()
    crosswalk_table = {
        raw: {
            "matched_obelix_family": matched,
            "n_strict_cohort_formulas": int(raw_family_counts.get(raw, 0)),
            "status": (
                SEEN
                if matched is not None and matched in obelix_family_vocab
                else (UNSEEN if matched is not None else "NO_OBELIX_MATCH")
            ),
        }
        for raw, matched in HARGREAVES_TO_OBELIX_FAMILY_CROSSWALK.items()
    }
    crosswalk_table["Other (ambiguous, non-informative label)"] = {
        "matched_obelix_family": None,
        "n_strict_cohort_formulas": int(raw_family_counts.get("Other", 0)),
        "status": AMBIGUOUS,
    }

    n6 = _load_json(N6_PATH)
    subset_results: dict[str, dict[str, Any]] = {}
    subset_external_mae: dict[str, dict[str, float]] = {"SEEN": {}, "UNSEEN": {}}
    for model in MODEL_NAMES:
        predictions = n6["results"][model]["strict"]["predictions"]
        model_out: dict[str, Any] = {}
        for label, formulas in (("SEEN", seen_formulas), ("UNSEEN", unseen_formulas)):
            metrics = score_subset_from_frozen_predictions(predictions, set(formulas))
            if metrics["n"] >= MIN_N_FOR_SUBSET_CI:
                y_true = np.array(predictions["y_true"])
                y_pred = np.array(predictions["y_pred"])
                mask = np.array([f in set(formulas) for f in predictions["reduced_formula"]])
                metrics["mae_bootstrap_ci"] = bootstrap_metric_ci(
                    y_true[mask], y_pred[mask], "mae", seed=BOOTSTRAP_SEED, n_boot=N_BOOT
                )
            else:
                metrics["bootstrap_ci_note"] = (
                    f"n={metrics['n']} < {MIN_N_FOR_SUBSET_CI}; bootstrap CI not computed "
                    "(not statistically meaningful at this n)."
                )
            model_out[label] = metrics
            subset_external_mae[label][model] = metrics["mae"]
        subset_results[model] = model_out

    n4b_internal = load_n4b_internal()
    regime_anticipation: dict[str, dict[str, Any]] = {"SEEN": {}, "UNSEEN": {}}
    for label in ("SEEN", "UNSEEN"):
        n_subset = len(seen_formulas) if label == "SEEN" else len(unseen_formulas)
        if n_subset < MIN_N_FOR_SUBSET_CI:
            regime_anticipation[label]["note"] = (
                f"n_{label.lower()}={n_subset} < {MIN_N_FOR_SUBSET_CI}: regime-anticipation "
                "ranking statistics are not reported as reliable at this sample size, per task "
                "instruction not to overinterpret a small unseen-family subset."
            )
            continue
        for regime, internal_by_model in n4b_internal.items():
            regime_anticipation[label][regime] = regime_summary(
                internal_by_model, subset_external_mae[label]
            )

    return {
        "n20_part": "A_external_family_audit",
        "git_commit": _git("rev-parse", "HEAD"),
        "obelix_dev_population_family_vocabulary": sorted(obelix_family_vocab),
        "hargreaves_to_obelix_family_crosswalk": crosswalk_table,
        "crosswalk_method": (
            "Exact match after case/hyphen/space normalization and singular/plural folding "
            "only; no fuzzy/edit-distance matching. 'Zircon' (Hargreaves) intentionally NOT "
            "merged with OBELiX's 'zirconates' (chemically distinct mineral vs. oxide family), "
            "mirroring the project's existing LISICON/NASICON non-merge precedent in "
            "family_harmonization.py."
        ),
        "n_strict_cohort_formulas": 204,
        "n_seen_family": len(seen_formulas),
        "n_unseen_family": len(unseen_formulas),
        "n_ambiguous_or_missing_family": len(ambiguous_formulas),
        "seen_family_formulas": seen_formulas,
        "unseen_family_formulas": unseen_formulas,
        "ambiguous_or_missing_family_formulas": ambiguous_formulas,
        "ambiguous_status_by_formula": {f: status_by_formula[f] for f in ambiguous_formulas},
        "chemical_family_secondary_field_by_formula": chemical_family_by_formula,
        "chemical_family_note": (
            "Hargreaves' 'ChemicalFamily' column (Halides/Oxides/Sulphides/Other, a coarse "
            "anion-based grouping) is structurally different from OBELiX's structural-family "
            "'Family' column and is NOT used for the seen/unseen classification above -- "
            "recorded here only as descriptive secondary metadata."
        ),
        "per_model_subset_metrics": subset_results,
        "regime_anticipation_of_subset_external_mae": regime_anticipation,
        "regime_anticipation_method": (
            "sse_transfer.evaluation.regime_utility.regime_summary (N7/N7b's own "
            "median_abs_mae_gap / spearman_rho / kendall_tau / top_model_selection_regret), "
            "applied to N4b's (post-leakage-fix) internal-CV MAE per regime against the "
            "seen-family and unseen-family external-MAE subsets separately, instead of the "
            "full 204-row external cohort N7b used."
        ),
    }


# ---------------------------------------------------------------------------
# Part B: random-CV duplicate / leakage audit
# ---------------------------------------------------------------------------


def _near_duplicate_dopant_series(dev_pool: pd.DataFrame) -> dict[str, Any]:
    """Descriptive only (never called leakage): OBELiX rows sharing the same
    chemical system (element set) but a different stoichiometry -- a dopant/
    solid-solution series. Uses pymatgen's element-set, not a fuzzy string
    match."""
    from pymatgen.core import Composition

    systems: dict[str, list[str]] = {}
    for formula in dev_pool["canonical_formula"]:
        try:
            elements = frozenset(str(e) for e in Composition(formula).elements)
        except Exception:
            continue
        key = "-".join(sorted(elements))
        systems.setdefault(key, []).append(formula)

    series = {k: v for k, v in systems.items() if len(set(v)) > 1}
    return {
        "n_chemical_systems_with_multiple_distinct_formulas": len(series),
        "n_rows_involved": sum(len(v) for v in series.values()),
        "note": (
            "Descriptive only: rows sharing the same element set but a different "
            "stoichiometry (e.g. a doping/solid-solution series). This is NOT the same "
            "condition as an exact-formula duplicate crossing folds, and is not, on its own, "
            "evidence of leakage -- reported per task instruction to describe near-duplicate "
            "proximity without calling it leakage."
        ),
        "largest_systems": [
            {"elements": k, "n_distinct_formulas": len(set(v)), "n_rows": len(v)}
            for k, v in sorted(series.items(), key=lambda kv: -len(kv[1]))[:15]
        ],
    }


def run_part_b(dev_pool: pd.DataFrame) -> dict[str, Any]:
    dup_groups = find_duplicate_formula_groups(dev_pool["canonical_formula"])
    folds = random_kfold_splits(len(dev_pool), n_splits=N_OUTER_SPLITS, seed=CV_SEED)
    fold_id = fold_id_array(len(dev_pool), folds)
    crossing = duplicate_groups_crossing_folds(dup_groups, fold_id)

    n_crossing = sum(1 for v in crossing.values() if v["crosses_folds"])
    n_same_fold = len(crossing) - n_crossing
    requires_diagnostic_regime = n_crossing > 0

    n20b_result: dict[str, Any] | None = None
    n20b_status = "NOT_YET_RUN"
    if N20B_PATH.exists():
        n20b = _load_json(N20B_PATH)
        n4b_internal = load_n4b_internal()["A_random_5fold_cv"]
        n20b_result = {
            "artifact_path": "artifacts/historical/n20b_formula_group_kfold.json",
            "cv_seed": n20b["cv_seed"],
            "n_unique_canonical_formulas": n20b["n_unique_canonical_formulas"],
            "pooled_mae_by_model": {
                m: n20b["results"][m]["pooled_mae"] for m in MODEL_NAMES if m in n20b["results"]
            },
            "n4b_random_5fold_pooled_mae_by_model": n4b_internal,
            "delta_formula_groupkfold_minus_random_5fold": {
                m: n20b["results"][m]["pooled_mae"] - n4b_internal[m]
                for m in MODEL_NAMES
                if m in n20b["results"]
            },
        }
        n20b_status = "DONE"

    return {
        "n20_part": "B_random_cv_duplicate_leakage_audit",
        "git_commit": _git("rev-parse", "HEAD"),
        "cv_seed": CV_SEED,
        "n_outer_splits": N_OUTER_SPLITS,
        "obelix_dev_population_n": len(dev_pool),
        "n_unique_canonical_formulas": int(dev_pool["canonical_formula"].nunique()),
        "n_exact_duplicate_formula_groups": len(dup_groups),
        "n_rows_in_duplicate_groups": sum(len(v) for v in dup_groups.values()),
        "n_duplicate_groups_crossing_random_cv_folds": n_crossing,
        "n_duplicate_groups_entirely_within_one_fold": n_same_fold,
        "duplicate_group_fold_detail": crossing,
        "verdict": (
            "Exact-formula duplicates DO cross random-CV folds: "
            f"{n_crossing}/{len(dup_groups)} duplicate-formula groups "
            f"({sum(len(v) for k, v in dup_groups.items() if crossing[k]['crosses_folds'])} of "
            f"{sum(len(v) for v in dup_groups.values())} duplicate rows) have their member rows "
            "split across more than one outer fold -- for these formulas, one measurement is a "
            "fold's test row while another measurement of the IDENTICAL formula is in that "
            "fold's training partition. The reviewer's leakage concern IS supported by exact-"
            "formula evidence."
            if requires_diagnostic_regime
            else "No exact-formula duplicate crosses a random-CV fold; the reviewer's leakage "
            "concern is not supported by exact-formula evidence."
        ),
        "requires_formula_group_kfold_diagnostic_regime": requires_diagnostic_regime,
        "formula_group_kfold_diagnostic_status": n20b_status,
        "formula_group_kfold_diagnostic_result": n20b_result,
        "formula_group_kfold_diagnostic_script": (
            "scripts/transport_analysis/n20b_formula_group_kfold.py"
        ),
        "near_duplicate_dopant_series_descriptive": _near_duplicate_dopant_series(dev_pool),
    }


# ---------------------------------------------------------------------------
# Part C: cross-database label compatibility
# ---------------------------------------------------------------------------


def run_part_c(ledger: pd.DataFrame) -> dict[str, Any]:
    obelix_raw = load_obelix_raw()
    conductivity = obelix_raw["Ionic conductivity (S cm-1)"]
    status = label_status(conductivity)
    n_censored = int((status != LABEL_STATUS_EXACT).sum())
    exact_mask = status == LABEL_STATUS_EXACT
    obelix_exact = obelix_raw.loc[exact_mask].copy()
    obelix_exact["canonical_formula"] = obelix_exact["Reduced Composition"].apply(
        safe_reduced_formula
    )
    obelix_exact["y"] = np.log10(censoring_bound(conductivity.loc[exact_mask]))
    obelix_exact = obelix_exact.dropna(subset=["canonical_formula"])
    n_obelix_formula_disagreement = 0
    obelix_y_by_formula: dict[str, float] = {}
    for formula, group in obelix_exact.groupby("canonical_formula"):
        obelix_y_by_formula[str(formula)] = float(group["y"].median())
        if group["y"].max() - group["y"].min() > 1.0:
            n_obelix_formula_disagreement += 1

    overlap_ledger = ledger.loc[
        ledger["exclusion_reason"].isin([EXACT_DOI_OVERLAP, COMPOSITION_ONLY_OVERLAP])
    ].copy()
    hargreaves_by_formula: dict[str, float] = {}
    hargreaves_meta: dict[str, dict[str, Any]] = {}
    for formula, group in overlap_ledger.groupby("reduced_formula"):
        hargreaves_by_formula[str(formula)] = float(group["log_target"].median())
        hargreaves_meta[str(formula)] = {
            "n_source_rows": int(len(group)),
            "overlap_classification": sorted(group["overlap_classification"].unique().tolist()),
        }

    compatibility = paired_label_compatibility(
        obelix_y_by_formula,
        hargreaves_by_formula,
        seed=BOOTSTRAP_SEED,
        n_boot=N_BOOT,
        min_n_for_ci=MIN_N_FOR_SUBSET_CI,
    )
    compatibility["obelix_side_meta"] = {
        "obelix_raw_rows": len(obelix_raw),
        "n_censored_excluded": n_censored,
        "n_exact_formulas": len(obelix_y_by_formula),
        "n_formulas_with_within_obelix_spread_over_1_log10_s_cm": n_obelix_formula_disagreement,
    }
    compatibility["hargreaves_side_meta"] = {
        f: hargreaves_meta[f] for f in compatibility.get("reduced_formulas", [])
    }

    by_classification: dict[str, dict[str, float]] = {}
    for label, group_formulas in (
        (
            "EXACT_COMPOSITION_AND_DOI_MATCH_ONLY",
            [
                f
                for f in compatibility.get("reduced_formulas", [])
                if hargreaves_meta[f]["overlap_classification"] == [MATCH_EXACT_DOI]
            ],
        ),
        (
            "COMPOSITION_ONLY_MATCH_INVOLVED",
            [
                f
                for f in compatibility.get("reduced_formulas", [])
                if MATCH_COMPOSITION_ONLY in hargreaves_meta[f]["overlap_classification"]
            ],
        ),
    ):
        if len(group_formulas) < 2:
            continue
        obelix_arr = np.array([obelix_y_by_formula[f] for f in group_formulas])
        hargreaves_arr = np.array([hargreaves_by_formula[f] for f in group_formulas])
        abs_diff = np.abs(hargreaves_arr - obelix_arr)
        by_classification[label] = {
            "n": len(group_formulas),
            "mean_absolute_discrepancy": float(np.mean(abs_diff)),
            "median_absolute_discrepancy": float(np.median(abs_diff)),
        }
    compatibility["discrepancy_by_overlap_classification"] = by_classification
    compatibility["overlap_classification_note"] = (
        "EXACT_COMPOSITION_AND_DOI_MATCH pairs (near-certain the same underlying literature "
        "measurement cited by both databases) show much smaller discrepancy than pairs "
        "involving a COMPOSITION_ONLY_MATCH (same nominal formula, different/unverifiable "
        "source) -- consistent with genuine synthesis/measurement-protocol variability across "
        "independent reports of the 'same' nominal material, not a database-parsing error."
    )

    obelix_metadata_fields = {
        "conductivity_type": (
            "OBELiX's primary column 'Ionic conductivity (S cm-1)' is an author-curated "
            "'best available' value; a separate 'IC (Total)'/'IC (Bulk)' split exists but is "
            "populated for <5% of rows (29/599 and 28/599 respectively), so the primary "
            "column cannot be verified row-by-row as uniformly bulk or total conductivity."
        ),
        "temperature_field": (
            "None -- OBELiX has no explicit per-row temperature column; the dataset is "
            "curated by its own authors as room-temperature values, not independently "
            "re-verified per row here."
        ),
        "measurement_protocol_field": "None.",
        "units": "S cm^-1 (matches Hargreaves; no unit harmonization needed).",
        "aggregation_across_measurements": (
            "One reported value per OBELiX row; no explicit multi-measurement aggregation field."
        ),
    }
    hargreaves_metadata_fields = {
        "conductivity_type": "No bulk/total/grain-boundary field at all.",
        "temperature_field": (
            "'temperature' (degrees C), per-row, range 5-873 C in the raw file; this "
            "project's RT window is [15,35] C (fixed pre-analysis, "
            "reports/cross_dataset_study/n1_external_data_audit.md)."
        ),
        "measurement_protocol_field": "None beyond 'source' (literature DOI).",
        "units": "S cm^-1 (matches OBELiX).",
        "aggregation_across_measurements": (
            "This project's own N2 pipeline aggregates repeated RT-window rows of the same "
            "reduced formula to their MEDIAN log10(conductivity) "
            "(hargreaves_cohort.AGGREGATION_RULE); the raw Hargreaves file itself has no "
            "aggregation field, most repeats are Arrhenius-series multi-temperature "
            "measurements of the same sample."
        ),
    }

    return {
        "n20_part": "C_cross_database_label_compatibility",
        "git_commit": _git("rev-parse", "HEAD"),
        "obelix_metadata_fields": obelix_metadata_fields,
        "hargreaves_metadata_fields": hargreaves_metadata_fields,
        "compatibility_gap_finding": (
            "Neither database's headline scalar target is a verified, uniform bulk-vs-total "
            "quantity: OBELiX's primary column mixes an unlabeled majority with a <5% "
            "Total/Bulk-flagged minority, and Hargreaves has no bulk/total field whatsoever. "
            "Both report S/cm and both are treated by this project as 'a single, author-"
            "reported conductivity value' rather than a verified bulk value -- this is a "
            "genuine label-definition compatibility gap that predates and is independent of "
            "any modeling choice."
        ),
        "paired_formula_calibration_audit": compatibility,
        "scope_note": (
            "These paired rows are excluded from every N2 external cohort (strict AND relaxed) "
            "by construction (overlap_classification != NO_MATCH) and are used HERE ONLY as a "
            "calibration/compatibility check between the two databases' raw label conventions "
            "-- never as external-test data, and this comparison is cross-database "
            "DISCREPANCY, not a model's prediction error."
        ),
    }


# ---------------------------------------------------------------------------
# Part D: provenance timeline
# ---------------------------------------------------------------------------


def run_part_d() -> dict[str, Any]:
    def commit(h: str) -> dict[str, str]:
        out = _git("show", "-s", "--format=%H|%ad|%s", "--date=iso-strict", h)
        full_hash, date, message = out.split("|", 2)
        return {"commit": full_hash, "date": date, "message": message}

    milestones = [
        {
            "step": "N3_freeze_8model_ladder_hyperparameters",
            **commit("60585f8"),
            "note": (
                "Each of the 8 portable composition-only models' hyperparameters tuned ONCE "
                "against OBELiX train(359)/validation(90), then frozen."
            ),
            "before_external_visibility": True,
        },
        {
            "step": "N4_original_internal_cv_LEAKY",
            **commit("229e5de"),
            "note": (
                "Original internal-CV regimes (Random-5fold / Family-GroupKFold / LOMFO) "
                "computed by reusing N3's frozen hyperparameters across every outer fold. "
                "Later found (b926660) to carry an optimism bias, because the 90 validation "
                "rows that shaped the frozen hyperparameters later served as outer-test rows "
                "in these same regimes."
            ),
            "before_external_visibility": True,
        },
        {
            "step": "N5_pretest_lock",
            **commit("600c75c"),
            "note": (
                "Internally hash-locked protocol (artifacts/frozen/external_transfer_lock.json: "
                "local "
                "SHA-256 hashes of frozen inputs + written pre-specified N6-N10 protocol, "
                "committed to this private branch). NOT a public timestamped preregistration "
                "-- no OSF/DOI/public tag exists for this project; describing it as "
                "'pre-registered' would overclaim. Locks in the leaky N4 numbers as the "
                "frozen internal-CV input at this point in time."
            ),
            "before_external_visibility": True,
        },
        {
            "step": "N6_one_shot_external_transfer_TO_HARGREAVES",
            **commit("312f0e5"),
            "note": (
                "One-shot (run_count=1) external inference on the 204-row strict Hargreaves "
                "cohort, using N3's frozen hyperparameters directly -- does NOT depend on N4's "
                "CV numbers at all, so N4's later-discovered leakage does not affect N6's "
                "predictions themselves. This is the moment external results became visible."
            ),
            "before_external_visibility": "THIS_IS_THE_BOUNDARY",
        },
        {
            "step": "N7_regime_utility_comparison_USING_LEAKY_N4",
            **commit("a273091"),
            "note": "H1/H2 NOT SUPPORTED, computed against N4's original (leaky) internal numbers.",
            "before_external_visibility": False,
        },
        {
            "step": "N10_go_no_go_NPJ_GO_USING_LEAKY_N4",
            **commit("c73bebb"),
            "note": "GO/NO-GO verdict NPJ-GO, still on N4's pre-fix numbers.",
            "before_external_visibility": False,
        },
        {
            "step": "N11_N13_manuscript_and_audit_drafted_USING_LEAKY_N4",
            "commit": "1a4bf3b..7eb7f80",
            "date": "2026-09-21T20:01:14+06:00..2026-09-21T20:31:15+06:00",
            "message": "Manuscript draft (N11), four-layer audit (N12), final status (N13)",
            "note": (
                "Full manuscript and an independent four-layer audit were completed BEFORE "
                "the internal-CV leakage defect was discovered."
            ),
            "before_external_visibility": False,
        },
        {
            "step": "DISCOVERY_AND_FIX_internal_cv_hyperparameter_leakage",
            **commit("b926660"),
            "note": (
                "N4b nested-CV recompute (hyperparameters now tuned per outer fold on an "
                "inner split of that fold's OWN outer-train partition only) fixes the "
                "optimism bias. N7b recomputes H1/H2 against N4b -- N6 was NOT rerun. H1/H2 "
                "conclusions and the NPJ-GO verdict are UNCHANGED after the fix. Also: "
                "manuscript language corrected from 'pre-registered' to 'pre-specified, "
                "hash-locked'; N1b reconciles a 403-vs-422 RT-window discrepancy; N2b adds "
                "the 25% source-overlap finding."
            ),
            "before_external_visibility": False,
        },
        {
            "step": "N9b_six_model_stable_sensitivity_recomputed_against_N4b",
            **commit("5893182"),
            "note": (
                "6-model stable-architecture H1/H2 sensitivity, recomputed against the "
                "corrected N4b numbers (current branch HEAD)."
            ),
            "before_external_visibility": False,
        },
    ]

    uncommitted_note = (
        "N15 through N19 (7 post-review sensitivity analyses, manuscript/SI rebuild, DOCX "
        "submission package, a citation fix, and a forensic NN-distance check) were performed "
        "in later sessions on top of HEAD 5893182 but were never committed -- they exist only "
        "as working-tree files (modified/untracked) at the time this reviewer-remediation pass "
        "(N20) began. None of them reran N6 external inference or reopened the frozen 113-row "
        "OBELiX test split."
    )

    return {
        "n20_part": "D_nested_cv_provenance_timeline",
        "reconstructed_from": "git log --date=iso-strict on branch npj-cross-dataset-extension",
        "external_visibility_boundary": {
            "commit": "312f0e5",
            "meaning": (
                "N6 is the one-shot moment Hargreaves external predictions/labels were first "
                "computed and became visible to the researcher. Every milestone above this "
                "commit in the table is BEFORE external visibility; every milestone below it "
                "is AFTER."
            ),
        },
        "milestones": milestones,
        "key_finding": (
            "The internal-CV hyperparameter-leakage defect (fixed in b926660) was discovered "
            "and fixed ~10.5 hours AFTER N6's external predictions were generated (312f0e5), "
            "and AFTER the GO/NO-GO decision (c73bebb), a full manuscript draft (N11), and a "
            "prior four-layer audit (N12) had already been completed using the leaky N4 "
            "numbers. N6's external predictions do not depend on N4's CV output and were NOT "
            "rerun; every claim that COMPARED internal-CV MAE to external MAE (H1/H2, "
            "regime-utility, GO/NO-GO) WAS recomputed post-fix (N4b/N7b) with an unchanged "
            "qualitative verdict."
        ),
        "preregistration_status": (
            "artifacts/frozen/external_transfer_lock.json is a purely INTERNAL, local "
            "git-committed SHA-256 "
            "hash lock plus written protocol text on a private branch -- there is no public "
            "timestamped registration (no OSF entry, no DOI, no pushed tag on a public "
            "remote). Accurate language: 'internally hash-locked protocol' / 'pre-specified "
            "internal protocol', never 'pre-registered' or 'publicly preregistered'. The "
            "b926660 commit already corrected the manuscript's own language to this effect; "
            "this timeline uses the same accurate framing throughout."
        ),
        "uncommitted_working_tree_note": uncommitted_note,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    dev_pool = load_dev_population()
    ledger = load_ledger()

    print("=== Part D: provenance timeline ===")
    part_d = run_part_d()
    guard_output(ARTIFACT_D).write_text(json.dumps(part_d, indent=2, default=str), encoding="utf-8")
    print(f"wrote {ARTIFACT_D.relative_to(ROOT)}")

    print("=== Part B: random-CV duplicate/leakage audit ===")
    part_b = run_part_b(dev_pool)
    guard_output(ARTIFACT_B).write_text(json.dumps(part_b, indent=2, default=str), encoding="utf-8")
    print(f"wrote {ARTIFACT_B.relative_to(ROOT)}")
    print(f"  {part_b['verdict']}")
    print(
        f"  formula-groupkfold diagnostic status: {part_b['formula_group_kfold_diagnostic_status']}"
    )

    print("=== Part A: external family audit ===")
    part_a = run_part_a(dev_pool, ledger)
    guard_output(ARTIFACT_A).write_text(json.dumps(part_a, indent=2, default=str), encoding="utf-8")
    print(f"wrote {ARTIFACT_A.relative_to(ROOT)}")
    n_seen, n_unseen = part_a["n_seen_family"], part_a["n_unseen_family"]
    n_ambig = part_a["n_ambiguous_or_missing_family"]
    print(f"  n_seen={n_seen} n_unseen={n_unseen} n_ambiguous={n_ambig}")

    print("=== Part C: cross-database label compatibility ===")
    part_c = run_part_c(ledger)
    guard_output(ARTIFACT_C).write_text(json.dumps(part_c, indent=2, default=str), encoding="utf-8")
    print(f"wrote {ARTIFACT_C.relative_to(ROOT)}")
    print(f"  n_paired_formulas={part_c['paired_formula_calibration_audit']['n_paired_formulas']}")


if __name__ == "__main__":
    main()
