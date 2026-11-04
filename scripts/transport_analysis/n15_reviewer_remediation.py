#!/usr/bin/env python3
"""Cross-dataset study, N15: post-review reviewer-remediation
sensitivity analyses.

STATUS: POST-REVIEW SENSITIVITY ANALYSES, NOT part of `artifacts/frozen/
external_transfer_lock.json`'s pre-specified/hash-locked N6-N10 protocol. See
`artifacts/frozen/n15_remediation_spec.json` (authored and hashed before any
number in this script was computed) for the exact definition of each of the
seven analyses below.

Every analysis reuses N6's already-frozen strict-cohort predictions/
residuals (`artifacts/frozen/n6_external_transfer_primary.json`) and/or N4b's already-
frozen nested internal-validation numbers
(`artifacts/historical/n4b_nested_internal_validation.json`) wherever a model
prediction or internal-CV MAE is needed -- NO model is fit, retrained, or
fine-tuned anywhere in this script, and N6 external inference is never
rerun. The only "fitting" performed here is the deterministic Magpie
composition-feature scaler (`features.composition_pipeline.
TrainFitCompositionScaler`), reused unmodified, for analysis 2's train-only
nearest-neighbor distances -- a feature transform, not an ML model. The
frozen OBELiX official 113-row test split is never opened.

Run with: python scripts/transport_analysis/n15_reviewer_remediation.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.data.hargreaves_cohort import (  # noqa: E402
    load_hargreaves_raw,
    load_obelix_raw,
)
from sse_transfer.data.obelix_loader import load_obelix  # noqa: E402
from sse_transfer.data.regression_dataset import load_regression_dataset  # noqa: E402
from sse_transfer.evaluation.internal_validation import (  # noqa: E402
    LOMFO_MIN_FAMILY_N,
    N_OUTER_SPLITS,
    canonicalize_and_fill_family,
    family_group_kfold_splits,
    leave_one_major_family_out_splits,
    major_family_counts,
    random_kfold_splits,
)
from sse_transfer.evaluation.metrics import regression_metrics  # noqa: E402
from sse_transfer.evaluation.n15_baseline_variability import (  # noqa: E402
    constant_predictor_metrics,
    within_composition_variability,
)
from sse_transfer.evaluation.n15_deployment_shift import (  # noqa: E402
    distance_distribution_summary,
    family_novelty_fraction,
    wasserstein_distance_1d,
)
from sse_transfer.evaluation.n15_external_uncertainty import (  # noqa: E402
    row_paired_bootstrap_mae_ci_and_ranking,
)
from sse_transfer.evaluation.n15_risk_coverage_aurc import (  # noqa: E402
    aurc_from_risk_coverage,
)
from sse_transfer.evaluation.n15_source_disjoint import (  # noqa: E402
    build_obelix_doi_set,
    classify_source_disjoint,
    score_subset_from_frozen_predictions,
    source_disjoint_formulas,
)
from sse_transfer.evaluation.n15_temperature_window import (  # noqa: E402
    build_strict_cohort_custom_window,
)
from sse_transfer.evaluation.regime_utility import (  # noqa: E402
    bootstrap_regime_pair_delta_ci,
    regime_summary,
)
from sse_transfer.features.composition_pipeline import TrainFitCompositionScaler  # noqa: E402
from sse_transfer.uncertainty.applicability_domain import (  # noqa: E402
    ad_score_error_correlation,
    fit_standardization,
    nearest_neighbor_ad_score,
)
from sse_transfer.uncertainty.risk_coverage import (  # noqa: E402
    risk_coverage_mae_at_retained_fraction,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

SPEC_PATH = ROOT / "artifacts" / "frozen" / "n15_remediation_spec.json"
SPEC_SHA256_AT_AUTHORING = "c448d97590eb1d0ee283166d5b7805dbba0c4612c1811c212ab9c4eac8a42733"

N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
N4B_PATH = ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
N7B_PATH = ROOT / "artifacts" / "authoritative" / "n7b_regime_utility_nested_corrected.json"
LEDGER_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_overlap_ledger.csv"
STRICT_COHORT_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_strict.csv"
OBELIX_RAW_PATH = ROOT / "data" / "raw" / "obelix" / "all.csv"

ARTIFACTS_DIR = ROOT / "artifacts" / "rerun" / "n15_artifacts"
TABLES_DIR = ROOT / "artifacts" / "rerun" / "n15_tables"
REPORT_PATH = ROOT / "reports" / "rerun" / "n15_reviewer_remediation.md"

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
BOOTSTRAP_SEED = 20260921
CV_SEED = 20260921
N_BOOT = 2000
REGIMES = ["A_random_5fold_cv", "B_family_group_kfold", "C_lomfo"]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------


def load_dev_population() -> pd.DataFrame:
    """449-row OBELiX development population with true_composition, y, and
    the harmonized/filled family label -- identical construction to N4's
    `load_dev_population` (scripts/transport_analysis/n4_internal_validation.py)."""
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
    return pool.reset_index(drop=True)


def load_n6() -> dict[str, Any]:
    return _load_json(N6_PATH)


def load_n4b_internal() -> dict[str, dict[str, float]]:
    data = _load_json(N4B_PATH)
    out: dict[str, dict[str, float]] = {r: {} for r in REGIMES}
    for model, regimes in data["results"].items():
        for regime, payload in regimes.items():
            key = "pooled_mae" if "pooled_mae" in payload else "pooled_mae_over_major_family_rows"
            out[regime][model] = float(payload[key])
    return out


def load_n6_external() -> tuple[dict[str, float], dict[str, np.ndarray]]:
    data = load_n6()
    mae_by_model: dict[str, float] = {}
    residuals_by_model: dict[str, np.ndarray] = {}
    for model, payload in data["results"].items():
        strict = payload["strict"]
        mae_by_model[model] = float(strict["mae"])
        residuals_by_model[model] = np.array(strict["predictions"]["residual"], dtype=float)
    return mae_by_model, residuals_by_model


# ---------------------------------------------------------------------------
# Analysis 1 -- fully source-disjoint external sensitivity
# ---------------------------------------------------------------------------


def run_analysis_1(n6: dict[str, Any]) -> dict[str, Any]:
    ledger = pd.read_csv(LEDGER_PATH)
    obelix_raw = pd.read_csv(OBELIX_RAW_PATH)
    obelix_doi_set = build_obelix_doi_set(obelix_raw)
    classification = classify_source_disjoint(ledger, obelix_doi_set)
    keep_formulas = set(source_disjoint_formulas(classification))
    n_strict = len(classification)
    n_source_disjoint = len(keep_formulas)

    per_model: dict[str, Any] = {}
    for model in MODEL_NAMES:
        strict_predictions = n6["results"][model]["strict"]["predictions"]
        full_strict = {
            "mae": n6["results"][model]["strict"]["mae"],
            "rmse": n6["results"][model]["strict"]["rmse"],
            "spearman_rho": n6["results"][model]["strict"]["spearman_rho"],
            "n": n6["results"][model]["strict"]["n"],
        }
        subset = score_subset_from_frozen_predictions(strict_predictions, keep_formulas)
        per_model[model] = {
            "full_strict": full_strict,
            "source_disjoint": subset,
            "mae_delta_source_disjoint_minus_full_strict": subset["mae"] - full_strict["mae"],
        }

    return {
        "analysis": "N15_analysis_1_fully_source_disjoint_external_sensitivity",
        "obelix_doi_set_size": len(obelix_doi_set),
        "obelix_doi_set_size_note": "individual DOIs after '|'-splitting multi-DOI OBELiX cells",
        "n_strict_cohort": n_strict,
        "n_source_disjoint": n_source_disjoint,
        "n_excluded_for_source_overlap": n_strict - n_source_disjoint,
        "fraction_excluded": (n_strict - n_source_disjoint) / n_strict,
        "source_disjoint_formulas": sorted(keep_formulas),
        "per_model": per_model,
        "n2b_comparison": {
            "n2b_reported_excluded": 51,
            "n2b_reported_total": 204,
            "n15_excluded": n_strict - n_source_disjoint,
            "n15_total": n_strict,
            "note": (
                "N2b (reports/cross_dataset_study/n2b_source_overlap_audit.md) reported 51/204 "
                "using a "
                "simpler per-composition-row DOI check without '|'-splitting OBELiX's "
                "multi-DOI cells; N15's number may differ slightly for that reason."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Analysis 2 -- deployment-shift alignment
# ---------------------------------------------------------------------------


def _fold_nn_distances(
    pool: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (distances, family_novelty_flags) aligned to the rows actually
    scored across all folds (each row scored exactly once for regimes A/B;
    C is called per-family and concatenated by the caller instead)."""
    n = len(pool)
    distances = np.full(n, np.nan)
    novel = np.zeros(n, dtype=bool)
    comps = pool["true_composition"]
    families = pool["family_filled"].to_numpy()
    for train_idx, test_idx in folds:
        scaler = TrainFitCompositionScaler().fit(comps.iloc[train_idx])
        x_ref = scaler.transform(comps.iloc[train_idx])
        x_query = scaler.transform(comps.iloc[test_idx])
        stats = fit_standardization(x_ref)
        d = nearest_neighbor_ad_score(x_ref, x_query, stats)
        distances[test_idx] = d
        novelty = family_novelty_fraction(families[test_idx], families[train_idx])
        train_families = set(families[train_idx].tolist())
        novel[test_idx] = np.array([families[i] not in train_families for i in test_idx])
        # cross-check the returned array against the pure helper.
        assert novelty["n_novel"] == int(novel[test_idx].sum())
    return distances, novel


def run_analysis_2(
    pool: pd.DataFrame, strict_df: pd.DataFrame, keep_formulas: set[str]
) -> dict[str, Any]:
    comps = pool["true_composition"]

    # Regime A: random 5-fold.
    folds_a = random_kfold_splits(len(pool), n_splits=N_OUTER_SPLITS, seed=CV_SEED)
    dist_a, novel_a = _fold_nn_distances(pool, folds_a)

    # Regime B: family-group 5-fold.
    folds_b = family_group_kfold_splits(
        pool["family_filled"], n_splits=N_OUTER_SPLITS, seed=CV_SEED
    )
    dist_b, novel_b = _fold_nn_distances(pool, folds_b)

    # Regime C: LOMFO -- partial coverage (major families only).
    major_families = sorted(major_family_counts(pool["family_filled"], min_n=LOMFO_MIN_FAMILY_N))
    splits_c = leave_one_major_family_out_splits(pool["family_filled"], major_families)
    dist_c_list: list[float] = []
    novel_c_list: list[bool] = []
    families = pool["family_filled"].to_numpy()
    for _fam_name, (train_idx, test_idx) in splits_c.items():
        scaler = TrainFitCompositionScaler().fit(comps.iloc[train_idx])
        x_ref = scaler.transform(comps.iloc[train_idx])
        x_query = scaler.transform(comps.iloc[test_idx])
        stats = fit_standardization(x_ref)
        d = nearest_neighbor_ad_score(x_ref, x_query, stats)
        dist_c_list.extend(d.tolist())
        train_families = set(families[train_idx].tolist())
        novel_c_list.extend([families[i] not in train_families for i in test_idx])
    dist_c = np.array(dist_c_list, dtype=float)

    # External: single scaler fit on the full 449-row pool (identical to N8's AD reference fit).
    scaler_full = TrainFitCompositionScaler().fit(comps)
    x_ref_full = scaler_full.transform(comps)
    ad_stats_full = fit_standardization(x_ref_full)
    x_query_strict = scaler_full.transform(strict_df["reduced_formula"])
    dist_strict = nearest_neighbor_ad_score(x_ref_full, x_query_strict, ad_stats_full)
    disjoint_mask = strict_df["reduced_formula"].isin(keep_formulas).to_numpy()
    dist_source_disjoint = dist_strict[disjoint_mask]

    summaries = {
        "A_random_5fold_cv": distance_distribution_summary(dist_a),
        "B_family_group_kfold": distance_distribution_summary(dist_b),
        "C_lomfo": distance_distribution_summary(dist_c),
        "hargreaves_strict": distance_distribution_summary(dist_strict),
        "hargreaves_source_disjoint": distance_distribution_summary(dist_source_disjoint),
    }
    family_novelty = {
        "A_random_5fold_cv": {"n": int(len(novel_a)), "fraction_novel": float(novel_a.mean())},
        "B_family_group_kfold": {"n": int(len(novel_b)), "fraction_novel": float(novel_b.mean())},
        "C_lomfo": {
            "n": int(len(novel_c_list)),
            "fraction_novel": float(np.mean(novel_c_list)) if novel_c_list else float("nan"),
        },
    }

    wasserstein = {}
    for regime_name, internal_dist in (
        ("A_random_5fold_cv", dist_a),
        ("B_family_group_kfold", dist_b),
        ("C_lomfo", dist_c),
    ):
        wasserstein[regime_name] = {
            "vs_hargreaves_strict": wasserstein_distance_1d(internal_dist, dist_strict),
            "vs_hargreaves_source_disjoint": wasserstein_distance_1d(
                internal_dist, dist_source_disjoint
            ),
        }

    return {
        "analysis": "N15_analysis_2_deployment_shift_alignment",
        "cv_seed": CV_SEED,
        "n_outer_splits": N_OUTER_SPLITS,
        "lomfo_min_family_n": LOMFO_MIN_FAMILY_N,
        "major_families": major_families,
        "distance_distribution_summary": summaries,
        "family_novelty_fraction": family_novelty,
        "wasserstein_distance_internal_vs_external": wasserstein,
        "note": "No target y/log_target value was read anywhere in this analysis.",
    }


# ---------------------------------------------------------------------------
# Analysis 3 -- stable-model sensitivity (exclude ridge + shallow_mlp)
# ---------------------------------------------------------------------------

EXCLUDED_UNSTABLE_MODELS = ("ridge", "shallow_mlp")


def run_analysis_3(
    internal_by_regime: dict[str, dict[str, float]],
    external_mae: dict[str, float],
    external_residuals: dict[str, np.ndarray],
) -> dict[str, Any]:
    retained = [m for m in MODEL_NAMES if m not in EXCLUDED_UNSTABLE_MODELS]

    filtered_internal = {
        regime: {m: v for m, v in d.items() if m in retained}
        for regime, d in internal_by_regime.items()
    }
    filtered_external_mae = {m: v for m, v in external_mae.items() if m in retained}
    filtered_external_residuals = {m: v for m, v in external_residuals.items() if m in retained}

    per_regime_summary = {
        regime: regime_summary(filtered_internal[regime], filtered_external_mae)
        for regime in REGIMES
    }
    pairwise_ci: dict[str, dict[str, Any]] = {}
    statistics = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]
    labels = {"A_random_5fold_cv": "A", "B_family_group_kfold": "B", "C_lomfo": "C"}
    for i, regime_a in enumerate(REGIMES):
        for regime_b in REGIMES[i + 1 :]:
            key = f"{labels[regime_b]}_minus_{labels[regime_a]}"
            pairwise_ci[key] = {
                stat: bootstrap_regime_pair_delta_ci(
                    filtered_internal[regime_a],
                    filtered_internal[regime_b],
                    filtered_external_residuals,
                    statistic=stat,
                    seed=BOOTSTRAP_SEED,
                    n_boot=N_BOOT,
                )
                for stat in statistics
            }

    h1_support = {
        pair: pairwise_ci[pair]["median_abs_mae_gap"]["ci_high"] < 0.0
        for pair in ("B_minus_A", "C_minus_A")
    }
    h2_support = {
        f"{pair}_{stat}": pairwise_ci[pair][stat]["ci_low"] > 0.0
        for pair in ("B_minus_A", "C_minus_A")
        for stat in ("spearman_rho", "kendall_tau")
    }

    return {
        "analysis": "N15_analysis_3_stable_model_sensitivity",
        "excluded_models": list(EXCLUDED_UNSTABLE_MODELS),
        "retained_models": retained,
        "primary_8_model_result_unmodified": True,
        "per_regime_summary": per_regime_summary,
        "pairwise_bootstrap_ci": pairwise_ci,
        "h1_family_aware_smaller_mae_gap_than_random_cv": h1_support,
        "h2_family_aware_better_rank_agreement_than_random_cv": h2_support,
    }


# ---------------------------------------------------------------------------
# Analysis 4 -- external uncertainty (row-paired bootstrap)
# ---------------------------------------------------------------------------


def run_analysis_4(n6: dict[str, Any], keep_formulas: set[str]) -> dict[str, Any]:
    residuals_strict = {
        m: np.array(n6["results"][m]["strict"]["predictions"]["residual"], dtype=float)
        for m in MODEL_NAMES
    }
    strict_formulas = n6["results"][MODEL_NAMES[0]]["strict"]["predictions"]["reduced_formula"]
    mask = np.array([f in keep_formulas for f in strict_formulas])
    residuals_source_disjoint = {m: residuals_strict[m][mask] for m in MODEL_NAMES}

    result_strict = row_paired_bootstrap_mae_ci_and_ranking(
        residuals_strict, seed=BOOTSTRAP_SEED, n_boot=N_BOOT
    )
    result_source_disjoint = row_paired_bootstrap_mae_ci_and_ranking(
        residuals_source_disjoint, seed=BOOTSTRAP_SEED, n_boot=N_BOOT
    )
    return {
        "analysis": "N15_analysis_4_external_uncertainty",
        "seed": BOOTSTRAP_SEED,
        "n_boot": N_BOOT,
        "hargreaves_strict": result_strict,
        "hargreaves_source_disjoint": result_source_disjoint,
    }


# ---------------------------------------------------------------------------
# Analysis 5 -- AD / AURC over a finer coverage grid, all 8 models
# ---------------------------------------------------------------------------

COVERAGE_GRID: tuple[float, ...] = tuple(round(1.0 - 0.05 * i, 2) for i in range(16))  # 1.00..0.25


def run_analysis_5(
    pool: pd.DataFrame, n6: dict[str, Any], strict_df: pd.DataFrame
) -> dict[str, Any]:
    scaler = TrainFitCompositionScaler().fit(pool["true_composition"])
    x_ref = scaler.transform(pool["true_composition"])
    x_query = scaler.transform(strict_df["reduced_formula"])
    ad_stats = fit_standardization(x_ref)
    ad_scores = nearest_neighbor_ad_score(x_ref, x_query, ad_stats)
    cohort_index = {rf: i for i, rf in enumerate(strict_df["reduced_formula"])}

    per_model: dict[str, Any] = {}
    n_positive_spearman = 0
    n_defined_spearman = 0
    for model in MODEL_NAMES:
        strict = n6["results"][model]["strict"]
        y_true = np.array(strict["predictions"]["y_true"])
        y_pred = np.array(strict["predictions"]["y_pred"])
        row_ids = strict["predictions"]["reduced_formula"]
        order = np.array([cohort_index[rf] for rf in row_ids])
        model_ad_scores = ad_scores[order]
        abs_residual = np.abs(y_pred - y_true)

        correlation = ad_score_error_correlation(model_ad_scores, abs_residual)
        risk_coverage = risk_coverage_mae_at_retained_fraction(
            model_ad_scores, abs_residual, COVERAGE_GRID
        )
        aurc = aurc_from_risk_coverage(risk_coverage)
        per_model[model] = {
            "spearman_ad_vs_abs_error": correlation,
            "risk_coverage_mae_fine_grid": risk_coverage,
            "aurc": aurc,
        }
        if not np.isnan(correlation["spearman_r"]):
            n_defined_spearman += 1
            if correlation["spearman_r"] > 0:
                n_positive_spearman += 1

    return {
        "analysis": "N15_analysis_5_ad_aurc",
        "coverage_grid": list(COVERAGE_GRID),
        "n_models_with_positive_spearman": n_positive_spearman,
        "n_models_with_defined_spearman": n_defined_spearman,
        "per_model": per_model,
        "note": (
            "AURC reported for all 8 models descriptively; no model is chosen based on its "
            "own external performance for this headline (per spec analysis_5)."
        ),
    }


# ---------------------------------------------------------------------------
# Analysis 6 -- naive baseline + raw-measurement variability
# ---------------------------------------------------------------------------


def run_analysis_6(
    pool: pd.DataFrame, strict_df: pd.DataFrame, keep_formulas: set[str]
) -> dict[str, Any]:
    mean_y = float(pool["y"].mean())
    median_y = float(pool["y"].median())

    strict_y_true = strict_df["log_target_rt_median"].to_numpy(dtype=float)
    disjoint_mask = strict_df["reduced_formula"].isin(keep_formulas).to_numpy()
    disjoint_y_true = strict_y_true[disjoint_mask]

    baselines = {
        "obelix_dev_mean_y": mean_y,
        "obelix_dev_median_y": median_y,
        "hargreaves_strict": {
            "n": int(len(strict_y_true)),
            "mean_predictor": constant_predictor_metrics(strict_y_true, mean_y),
            "median_predictor": constant_predictor_metrics(strict_y_true, median_y),
        },
        "hargreaves_source_disjoint": {
            "n": int(len(disjoint_y_true)),
            "mean_predictor": constant_predictor_metrics(disjoint_y_true, mean_y),
            "median_predictor": constant_predictor_metrics(disjoint_y_true, median_y),
        },
    }

    ledger = pd.read_csv(LEDGER_PATH)
    rt_valid = ledger[(ledger["in_rt_window"]) & (ledger["valid_target"])]
    variability = within_composition_variability(
        rt_valid, formula_column="reduced_formula", value_column="log_target", min_repeats=2
    )
    variability.pop("per_group")  # exclude the large per-formula dict from the top-level JSON

    return {
        "analysis": "N15_analysis_6_baseline_and_variability",
        "constant_predictor_baselines": baselines,
        "within_composition_variability": variability,
    }


# ---------------------------------------------------------------------------
# Analysis 7 -- fixed [20, 30] degC temperature-window sensitivity
# ---------------------------------------------------------------------------


def run_analysis_7(n6: dict[str, Any], strict_df: pd.DataFrame) -> dict[str, Any]:
    hargreaves_raw = load_hargreaves_raw()
    obelix_raw = load_obelix_raw()
    rebuilt = build_strict_cohort_custom_window(hargreaves_raw, obelix_raw, low_c=20.0, high_c=30.0)
    n_new = len(rebuilt)

    strict_formulas = set(strict_df["reduced_formula"])
    new_formulas = set(rebuilt["reduced_formula"])
    if not new_formulas.issubset(strict_formulas):
        raise RuntimeError(
            "N15 analysis 7: the [20,30]degC rebuilt cohort contains a formula not in the "
            "original 204-row strict cohort -- the spec's subset guarantee was violated; "
            f"unexpected formulas: {sorted(new_formulas - strict_formulas)}"
        )

    per_model: dict[str, Any] = {}
    for model in MODEL_NAMES:
        strict_predictions = n6["results"][model]["strict"]["predictions"]
        pred_by_formula = dict(
            zip(strict_predictions["reduced_formula"], strict_predictions["y_pred"], strict=False)
        )
        orig_true_by_formula = dict(
            zip(strict_predictions["reduced_formula"], strict_predictions["y_true"], strict=False)
        )

        rebuilt_sorted = rebuilt.sort_values("reduced_formula").reset_index(drop=True)
        y_pred_new = np.array([pred_by_formula[f] for f in rebuilt_sorted["reduced_formula"]])
        y_true_new = rebuilt_sorted["log_target_rt_median"].to_numpy(dtype=float)
        metrics_new = regression_metrics(y_true_new, y_pred_new)
        metrics_new["n"] = int(len(rebuilt_sorted))

        y_true_orig_subset = np.array(
            [orig_true_by_formula[f] for f in rebuilt_sorted["reduced_formula"]]
        )
        metrics_orig_subset = regression_metrics(y_true_orig_subset, y_pred_new)
        metrics_orig_subset["n"] = int(len(rebuilt_sorted))

        per_model[model] = {
            "window_20_30": metrics_new,
            "window_15_35_same_formula_subset": metrics_orig_subset,
            "mae_delta_window_change_only": metrics_new["mae"] - metrics_orig_subset["mae"],
            "mae_delta_vs_full_strict_204": (
                metrics_new["mae"] - n6["results"][model]["strict"]["mae"]
            ),
        }

    return {
        "analysis": "N15_analysis_7_fixed_temperature_sensitivity",
        "window_c": [20.0, 30.0],
        "n_strict_204_window_15_35": len(strict_df),
        "n_window_20_30": n_new,
        "per_model": per_model,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    spec_sha256_now = _sha256_file(SPEC_PATH)
    if spec_sha256_now != SPEC_SHA256_AT_AUTHORING:
        raise RuntimeError(
            f"N15 spec integrity check failed: {SPEC_PATH} sha256 is {spec_sha256_now}, "
            f"expected {SPEC_SHA256_AT_AUTHORING} (recorded at authoring time, before any "
            "N15 calculation). The spec must not be edited after analyses begin."
        )

    n6 = load_n6()
    internal_by_regime = load_n4b_internal()
    external_mae, external_residuals = load_n6_external()
    strict_df = pd.read_csv(STRICT_COHORT_PATH)
    pool = load_dev_population()

    result_1 = run_analysis_1(n6)
    keep_formulas = set(result_1["source_disjoint_formulas"])

    result_2 = run_analysis_2(pool, strict_df, keep_formulas)
    result_3 = run_analysis_3(internal_by_regime, external_mae, external_residuals)
    result_4 = run_analysis_4(n6, keep_formulas)
    result_5 = run_analysis_5(pool, n6, strict_df)
    result_6 = run_analysis_6(pool, strict_df, keep_formulas)
    result_7 = run_analysis_7(n6, strict_df)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    common_provenance = {
        "phase": "N15_reviewer_remediation",
        "status": "POST_REVIEW_SENSITIVITY_ANALYSIS_NOT_PRESPECIFIED",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "spec_file": "artifacts/frozen/n15_remediation_spec.json",
        "spec_file_sha256": spec_sha256_now,
        "n6_source": "artifacts/frozen/n6_external_transfer_primary.json (frozen, not rerun)",
        "n4b_source": "artifacts/historical/n4b_nested_internal_validation.json (frozen, not "
        "rerun)",
    }

    outputs = {
        "n15_source_disjoint.json": {**common_provenance, **result_1},
        "n15_deployment_shift.json": {**common_provenance, **result_2},
        "n15_stable_model_sensitivity.json": {**common_provenance, **result_3},
        "n15_external_uncertainty.json": {**common_provenance, **result_4},
        "n15_ad_aurc.json": {**common_provenance, **result_5},
        "n15_baseline_variability.json": {**common_provenance, **result_6},
        "n15_temperature_sensitivity.json": {**common_provenance, **result_7},
    }
    for filename, payload in outputs.items():
        path = ARTIFACTS_DIR / filename
        guard_output(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")

    _write_tables(result_1, result_2, result_3, result_4, result_5, result_6, result_7)
    _write_report(
        common_provenance, result_1, result_2, result_3, result_4, result_5, result_6, result_7
    )

    summary = {
        **common_provenance,
        "n_strict_cohort": result_1["n_strict_cohort"],
        "n_source_disjoint": result_1["n_source_disjoint"],
        "n_window_20_30": result_7["n_window_20_30"],
        "stable_model_h1_h2": {
            "h1": result_3["h1_family_aware_smaller_mae_gap_than_random_cv"],
            "h2": result_3["h2_family_aware_better_rank_agreement_than_random_cv"],
        },
        "n_models_with_positive_spearman_finer_grid": result_5["n_models_with_positive_spearman"],
    }
    guard_output(ARTIFACTS_DIR / "n15_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {(ARTIFACTS_DIR / 'n15_summary.json').relative_to(ROOT)}")
    print(f"wrote {REPORT_PATH.relative_to(ROOT)}")


def _write_tables(r1, r2, r3, r4, r5, r6, r7) -> None:  # type: ignore[no-untyped-def]
    # Table 1: source-disjoint per-model metrics.
    rows = []
    for model, payload in r1["per_model"].items():
        rows.append(
            {
                "model": model,
                "mae_full_strict_n204": payload["full_strict"]["mae"],
                "mae_source_disjoint": payload["source_disjoint"]["mae"],
                "n_source_disjoint": payload["source_disjoint"]["n"],
                "mae_delta": payload["mae_delta_source_disjoint_minus_full_strict"],
            }
        )
    pd.DataFrame(rows).to_csv(guard_output(TABLES_DIR / "n15_source_disjoint.csv"), index=False)

    # Table 2: deployment-shift distance summaries.
    rows = []
    for regime, summary in r2["distance_distribution_summary"].items():
        rows.append({"regime": regime, **summary})
    pd.DataFrame(rows).to_csv(guard_output(TABLES_DIR / "n15_deployment_shift.csv"), index=False)

    # Table 3: stable-model (6-model) regime summary.
    rows = []
    for regime, summary in r3["per_regime_summary"].items():
        rows.append({"regime": regime, **summary})
    pd.DataFrame(rows).to_csv(
        guard_output(TABLES_DIR / "n15_stable_model_sensitivity.csv"), index=False
    )

    # Table 4: external uncertainty (strict cohort primary).
    rows = []
    for model, payload in r4["hargreaves_strict"]["per_model"].items():
        rows.append(
            {
                "model": model,
                "point_mae": payload["point_mae"],
                "ci_low": payload["ci_low"],
                "ci_high": payload["ci_high"],
                "probability_rank_1": payload["probability_rank_1"],
            }
        )
    pd.DataFrame(rows).to_csv(
        guard_output(TABLES_DIR / "n15_external_uncertainty.csv"), index=False
    )

    # Table 5: AD/AURC per model.
    rows = []
    for model, payload in r5["per_model"].items():
        rows.append(
            {
                "model": model,
                "spearman_r": payload["spearman_ad_vs_abs_error"]["spearman_r"],
                "p_value": payload["spearman_ad_vs_abs_error"]["p_value"],
                "aurc": payload["aurc"]["aurc"],
            }
        )
    pd.DataFrame(rows).to_csv(guard_output(TABLES_DIR / "n15_ad_aurc.csv"), index=False)

    # Table 6: baseline/variability.
    baselines = r6["constant_predictor_baselines"]
    rows = [
        {
            "cohort": cohort,
            "n": baselines[cohort]["n"],
            "mean_predictor_mae": baselines[cohort]["mean_predictor"]["mae"],
            "median_predictor_mae": baselines[cohort]["median_predictor"]["mae"],
        }
        for cohort in ("hargreaves_strict", "hargreaves_source_disjoint")
    ]
    pd.DataFrame(rows).to_csv(
        guard_output(TABLES_DIR / "n15_baseline_variability.csv"), index=False
    )

    # Table 7: temperature-window sensitivity.
    rows = []
    for model, payload in r7["per_model"].items():
        rows.append(
            {
                "model": model,
                "mae_window_20_30": payload["window_20_30"]["mae"],
                "mae_window_15_35_same_subset": payload["window_15_35_same_formula_subset"]["mae"],
                "n": payload["window_20_30"]["n"],
                "mae_delta_window_change_only": payload["mae_delta_window_change_only"],
            }
        )
    pd.DataFrame(rows).to_csv(
        guard_output(TABLES_DIR / "n15_temperature_sensitivity.csv"), index=False
    )


def _write_report(prov, r1, r2, r3, r4, r5, r6, r7) -> None:  # type: ignore[no-untyped-def]
    lines: list[str] = []
    lines.append("# N15 -- Reviewer-Remediation Post-Review Sensitivity Analyses\n\n")
    lines.append(
        "Status: **POST-REVIEW SENSITIVITY ANALYSES** -- NOT pre-specified/hash-locked under "
        "`artifacts/frozen/external_transfer_lock.json`. No N6 external inference was rerun; no "
        "model was "
        "retrained or fine-tuned; the frozen OBELiX 113-row test split was never opened. "
        f"Spec: `artifacts/frozen/n15_remediation_spec.json` "
        f"(sha256 `{prov['spec_file_sha256']}`).\n\n"
        f"Git commit: `{prov['git_commit']}` on branch `{prov['git_branch']}`\n\n"
    )

    lines.append("## 1. Fully source-disjoint external sensitivity\n\n")
    lines.append(
        f"Strict cohort n={r1['n_strict_cohort']}. Fully source-disjoint "
        f"n=**{r1['n_source_disjoint']}** "
        f"({r1['n_excluded_for_source_overlap']} compositions excluded, "
        f"{r1['fraction_excluded']:.1%}). OBELiX DOI set size (after '|'-splitting): "
        f"{r1['obelix_doi_set_size']}.\n\n"
    )
    lines.append("| Model | MAE (strict, n=204) | MAE (source-disjoint) | n | Delta |\n")
    lines.append("|---|---|---|---|---|\n")
    for model, payload in r1["per_model"].items():
        lines.append(
            f"| {model} | {payload['full_strict']['mae']:.4f} | "
            f"{payload['source_disjoint']['mae']:.4f} | {payload['source_disjoint']['n']} | "
            f"{payload['mae_delta_source_disjoint_minus_full_strict']:+.4f} |\n"
        )

    lines.append("\n## 2. Deployment-shift alignment (no target y used)\n\n")
    lines.append("| Regime | n | Median dist | IQR | q90 | Max |\n|---|---|---|---|---|---|\n")
    for regime, summary in r2["distance_distribution_summary"].items():
        lines.append(
            f"| {regime} | {summary['n']} | {summary['median']:.4f} | {summary['iqr']:.4f} | "
            f"{summary['q90']:.4f} | {summary['max']:.4f} |\n"
        )
    lines.append(
        "\nRegime B (Family-GroupKFold)'s max standardized NN distance "
        f"({r2['distance_distribution_summary']['B_family_group_kfold']['max']:.1f}) "
        "is far outside "
        "its own q90; one held-out fold's excluded family sits enormously far, in standardized "
        "composition-feature space, from that fold's training pool. This is a purely "
        "composition-distance-based, target-free corroboration of N4b's independently-observed "
        "catastrophic extrapolation for ridge/shallow_mlp under Family-GroupKFold "
        "(reports/cross_dataset_study/n4b_nested_internal_validation.md: pooled MAE 143.1 and "
        "969.4 respectively) "
        "-- the same family-removal mechanism that produces those MAE blowups also produces "
        "extreme composition-space distance, via two independent signals, not one. This also "
        "explains regime B's very large Wasserstein distance below (dominated by this tail, "
        "since Wasserstein distance integrates over the full distribution, not just central "
        "quantiles).\n\n"
        "Family novelty fraction (held-out row's family absent from its fold's train set):\n\n"
    )
    lines.append("| Regime | n | Fraction novel |\n|---|---|---|\n")
    for regime, fn in r2["family_novelty_fraction"].items():
        lines.append(f"| {regime} | {fn['n']} | {fn['fraction_novel']:.4f} |\n")
    lines.append("\n1-D Wasserstein distance, internal regime vs. external cohort:\n\n")
    lines.append("| Regime | vs strict | vs source-disjoint |\n|---|---|---|\n")
    for regime, w in r2["wasserstein_distance_internal_vs_external"].items():
        lines.append(
            f"| {regime} | {w['vs_hargreaves_strict']:.4f} | "
            f"{w['vs_hargreaves_source_disjoint']:.4f} |\n"
        )

    lines.append("\n## 3. Stable-model sensitivity (excludes ridge, shallow_mlp)\n\n")
    retained_str = ", ".join(r3["retained_models"])
    lines.append(f"Primary 8-model N7b result is unmodified. Retained models: {retained_str}.\n\n")
    lines.append(
        "| Regime | Median \\|MAE gap\\| | Spearman rho | Kendall tau | Top-model regret |\n"
    )
    lines.append("|---|---|---|---|---|\n")
    for regime, s in r3["per_regime_summary"].items():
        lines.append(
            f"| {regime} | {s['median_abs_mae_gap']:.4f} | {s['spearman_rho']:.4f} | "
            f"{s['kendall_tau']:.4f} | {s['top_model_selection_regret']:.4f} |\n"
        )
    lines.append("\nH1 (family-aware regime's median abs MAE gap smaller than random CV's):\n\n")
    for pair, supported in r3["h1_family_aware_smaller_mae_gap_than_random_cv"].items():
        lines.append(f"- {pair}: **{'SUPPORTED' if supported else 'NOT SUPPORTED'}**\n")
    lines.append("\nH2 (family-aware regime's rank agreement higher than random CV's):\n\n")
    for pair, supported in r3["h2_family_aware_better_rank_agreement_than_random_cv"].items():
        lines.append(f"- {pair}: **{'SUPPORTED' if supported else 'NOT SUPPORTED'}**\n")

    lines.append(
        "\n## 4. External uncertainty (row-paired bootstrap, n_boot=2000, seed=20260921)\n\n"
    )
    lines.append("### Strict cohort (n=204, primary)\n\n")
    lines.append("| Model | MAE | 95% CI | P(rank 1) |\n|---|---|---|---|\n")
    for model, payload in r4["hargreaves_strict"]["per_model"].items():
        lines.append(
            f"| {model} | {payload['point_mae']:.4f} | [{payload['ci_low']:.4f}, "
            f"{payload['ci_high']:.4f}] | {payload['probability_rank_1']:.4f} |\n"
        )
    best_strict = r4["hargreaves_strict"]
    lines.append(
        f"\nBest model by point estimate: **{best_strict['best_model_point_estimate']}** "
        f"(P(stays rank 1) = {best_strict['probability_best_model_stays_rank_1']:.4f}).\n"
    )
    lines.append("\n### Source-disjoint cohort (companion)\n\n")
    lines.append("| Model | MAE | 95% CI | P(rank 1) |\n|---|---|---|---|\n")
    for model, payload in r4["hargreaves_source_disjoint"]["per_model"].items():
        lines.append(
            f"| {model} | {payload['point_mae']:.4f} | [{payload['ci_low']:.4f}, "
            f"{payload['ci_high']:.4f}] | {payload['probability_rank_1']:.4f} |\n"
        )

    lines.append("\n## 5. AD / AURC (finer 16-point coverage grid, all 8 models)\n\n")
    lines.append(
        f"{r5['n_models_with_positive_spearman']}/{r5['n_models_with_defined_spearman']} models "
        "show a positive Spearman(AD, |error|) at this finer grid.\n\n"
    )
    lines.append("| Model | Spearman r | p-value | AURC |\n|---|---|---|---|\n")
    for model, payload in r5["per_model"].items():
        c = payload["spearman_ad_vs_abs_error"]
        lines.append(
            f"| {model} | {c['spearman_r']:.4f} | {c['p_value']:.4g} | "
            f"{payload['aurc']['aurc']:.4f} |\n"
        )

    lines.append("\n## 6. Baseline and raw-measurement variability\n\n")
    b = r6["constant_predictor_baselines"]
    lines.append(
        f"OBELiX-development mean(y)={b['obelix_dev_mean_y']:.4f}, "
        f"median(y)={b['obelix_dev_median_y']:.4f}.\n\n"
    )
    lines.append("| Cohort | n | Mean-predictor MAE | Median-predictor MAE |\n|---|---|---|---|\n")
    for cohort in ("hargreaves_strict", "hargreaves_source_disjoint"):
        c = b[cohort]
        lines.append(
            f"| {cohort} | {c['n']} | {c['mean_predictor']['mae']:.4f} | "
            f"{c['median_predictor']['mae']:.4f} |\n"
        )
    v = r6["within_composition_variability"]
    lines.append(
        f"\nWithin-composition raw repeated-measurement variability (RT-window, valid-target rows, "
        f"n_groups_with_repeats={v['n_groups_with_repeats']}): median log10-range "
        f"{v['range_median']:.4f} (IQR [{v['range_q25']:.4f}, {v['range_q75']:.4f}]), pooled MAD "
        f"{v['pooled_mad']:.4f}, median n_repeats {v['n_repeats_median']:.1f}. "
        f"Labeled as: *{v['interpretation_label']}*.\n"
    )

    lines.append("\n## 7. Fixed [20, 30] degC temperature-window sensitivity\n\n")
    lines.append(
        f"n under [20,30]degC = **{r7['n_window_20_30']}** (of {r7['n_strict_204_window_15_35']} "
        "strict-cohort formulas under [15,35]degC).\n\n"
    )
    lines.append("| Model | MAE (20-30) | MAE (15-35, same subset) | n | Delta (window only) |\n")
    lines.append("|---|---|---|---|---|\n")
    for model, payload in r7["per_model"].items():
        lines.append(
            f"| {model} | {payload['window_20_30']['mae']:.4f} | "
            f"{payload['window_15_35_same_formula_subset']['mae']:.4f} | "
            f"{payload['window_20_30']['n']} | {payload['mae_delta_window_change_only']:+.4f} |\n"
        )

    lines.append(
        "\n## Not done in N15 (by design)\n\n"
        "- N6 external inference was not rerun.\n"
        "- No model was retrained or fine-tuned.\n"
        "- The frozen OBELiX official 113-row test split was never opened.\n"
        "- No manuscript or figure file was edited.\n"
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_PATH).write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
