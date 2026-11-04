#!/usr/bin/env python3
"""Cross-dataset study, Phase 2 Stage 0: glass-subset (n=36) pairwise
between-regime bootstrap difference CIs.

Author-review amendment context: Phase 1 (Stage 1, `phase1_glass_split.json`)
computed point-estimate regime-utility statistics (median |MAE gap|, Spearman
rho, Kendall tau, top-model-selection regret) for the UNSEEN_GLASS subset
(n=36) for regimes A (random 5-fold CV), B (Family-GroupKFold), C (LOMFO),
and D (formula-GroupKFold), but did not compute bootstrap CIs for the
between-regime deltas on that subset -- only point values existed. A
subsequent author amendment asked for CIs on the glass subset specifically,
so any manuscript claim about the glass-subset regime-utility pattern can be
scoped to what the data actually supports (e.g. "anticipates error
magnitude, not model selection" if the median-gap delta CI excludes zero but
the rank/regret delta CIs do not).

No external inference is rerun (N6's frozen `strict.predictions` are the
only source of external residuals, restricted here to the 36 glass/
glass-ceramic formulas from `phase1_glass_split.json`). No model is
retrained. The frozen 113-row OBELiX `test` split is never opened. Internal
(N4b/N20b) pooled-MAE-per-model numbers are reused unchanged; only which
external rows are bootstrap-resampled varies, matching the same convention
as `bootstrap_regime_pair_delta_ci`'s existing full-cohort use in N7b/Phase 1
Stage 2 item 4.

Run with: python scripts/transport_analysis/phase2_0_glass_subset_regime_cis.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.evaluation.regime_utility import (  # noqa: E402
    bootstrap_regime_pair_delta_ci,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
N4B_PATH = ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
N20B_PATH = ROOT / "artifacts" / "authoritative" / "n20b_formula_group_kfold_corrected.json"
GLASS_SPLIT_PATH = ROOT / "artifacts" / "authoritative" / "phase1_glass_split_corrected.json"

OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "phase2_glass_subset_regime_cis_corrected.json"

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
STATISTICS: tuple[str, ...] = (
    "median_abs_mae_gap",
    "spearman_rho",
    "kendall_tau",
    "top_model_selection_regret",
)
BOOTSTRAP_SEED = 20260921
N_BOOT = 2000


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def load_n4b_internal() -> dict[str, dict[str, float]]:
    data = _load_json(N4B_PATH)
    out: dict[str, dict[str, float]] = {
        "A_random_5fold_cv": {},
        "B_family_group_kfold": {},
        "C_lomfo": {},
    }
    for model, regimes in data["results"].items():
        for regime, payload in regimes.items():
            key = "pooled_mae" if "pooled_mae" in payload else "pooled_mae_over_major_family_rows"
            out[regime][model] = float(payload[key])
    return out


def load_n20b_internal() -> dict[str, float]:
    data = _load_json(N20B_PATH)
    return {m: float(data["results"][m]["pooled_mae"]) for m in MODEL_NAMES if m in data["results"]}


def glass_residuals_by_model(glass_formulas: set[str]) -> dict[str, np.ndarray]:
    n6 = _load_json(N6_PATH)
    reference_formulas: list[str] | None = None
    residuals: dict[str, np.ndarray] = {}
    for model in MODEL_NAMES:
        predictions = n6["results"][model]["strict"]["predictions"]
        formulas: list[str] = predictions["reduced_formula"]
        if reference_formulas is None:
            reference_formulas = formulas
        elif formulas != reference_formulas:
            raise RuntimeError(
                f"row order mismatch: model {model!r} strict.predictions.reduced_formula "
                "does not match the reference model's row order -- cannot apply a single "
                "positional glass-subset mask across models"
            )
        y_true = np.asarray(predictions["y_true"], dtype=float)
        y_pred = np.asarray(predictions["y_pred"], dtype=float)
        mask = np.array([f in glass_formulas for f in formulas])
        if mask.sum() != len(glass_formulas):
            raise RuntimeError(
                f"expected {len(glass_formulas)} glass rows in model {model!r}'s strict "
                f"predictions, found {int(mask.sum())}"
            )
        residuals[model] = (y_pred - y_true)[mask]
    return residuals


def main() -> None:
    glass_split = _load_json(GLASS_SPLIT_PATH)
    glass_formulas = set(glass_split["unseen_glass_formulas"])
    if len(glass_formulas) != 36:
        raise RuntimeError(f"expected 36 unseen-glass formulas, got {len(glass_formulas)}")

    print("=== Phase 2 Stage 0: glass-subset (n=36) pairwise regime-delta CIs ===")
    residuals = glass_residuals_by_model(glass_formulas)
    n_rows = len(next(iter(residuals.values())))
    print(f"  n_glass_rows={n_rows}")

    n4b_internal = load_n4b_internal()
    n20b_internal = load_n20b_internal()

    pairs = {
        "B_minus_A": (n4b_internal["A_random_5fold_cv"], n4b_internal["B_family_group_kfold"]),
        "C_minus_A": (n4b_internal["A_random_5fold_cv"], n4b_internal["C_lomfo"]),
        "D_minus_A": (n4b_internal["A_random_5fold_cv"], n20b_internal),
    }

    results: dict[str, Any] = {}
    for pair_name, (internal_a, internal_b) in pairs.items():
        pair_out: dict[str, Any] = {}
        for statistic in STATISTICS:
            pair_out[statistic] = bootstrap_regime_pair_delta_ci(
                internal_a,
                internal_b,
                residuals,
                statistic=statistic,
                seed=BOOTSTRAP_SEED,
                n_boot=N_BOOT,
            )
        results[pair_name] = pair_out
        gap_ci = pair_out["median_abs_mae_gap"]
        print(
            f"  {pair_name}: median_abs_mae_gap delta={gap_ci['point_delta']:.4f} "
            f"CI=[{gap_ci['ci_low']:.4f}, {gap_ci['ci_high']:.4f}]"
        )

    def _excludes_zero(ci: dict[str, Any]) -> bool:
        return bool(ci["ci_low"] > 0.0) or bool(ci["ci_high"] < 0.0)

    verdict_by_pair: dict[str, Any] = {}
    for pair_name, pair_out in results.items():
        gap_excludes_zero_favoring_b = _excludes_zero(pair_out["median_abs_mae_gap"]) and (
            pair_out["median_abs_mae_gap"]["point_delta"] < 0.0
        )
        ranking_or_regret_excludes_zero = any(
            _excludes_zero(pair_out[s])
            for s in ("spearman_rho", "kendall_tau", "top_model_selection_regret")
        )
        if gap_excludes_zero_favoring_b and not ranking_or_regret_excludes_zero:
            claim = "anticipates error magnitude, not model selection"
        elif gap_excludes_zero_favoring_b and ranking_or_regret_excludes_zero:
            claim = (
                "anticipates error magnitude AND at least one ranking/regret statistic, "
                "CI-supported"
            )
        elif not gap_excludes_zero_favoring_b:
            claim = "median-gap advantage for family-aware regime is not CI-supported at n=36"
        else:
            claim = "mixed/inconclusive"
        verdict_by_pair[pair_name] = {
            "median_gap_ci_excludes_zero_favoring_family_aware": gap_excludes_zero_favoring_b,
            "any_ranking_or_regret_ci_excludes_zero": ranking_or_regret_excludes_zero,
            "claim": claim,
        }

    output = {
        "phase": "phase2_stage0_glass_subset_regime_cis",
        "git_commit": _git("rev-parse", "HEAD"),
        "n_glass_rows": n_rows,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_boot": N_BOOT,
        "pairwise_bootstrap_ci": results,
        "verdict_by_pair": verdict_by_pair,
        "note": (
            "CI-excludes-zero uses a strict no-overlap standard (ci_low>0 or ci_high<0), matching "
            "the convention used elsewhere in this study (N21 Section B / SI-29) for calling a "
            "delta 'CI-supported' rather than merely a favorable point estimate."
        ),
        "provenance": (
            "Internal (N4b regimes A/B/C, N20b regime D) pooled-MAE-per-model numbers reused "
            "unchanged from artifacts/historical/n4b_nested_internal_validation.json and "
            "artifacts/historical/n20b_formula_group_kfold.json. External residuals are N6's "
            "frozen "
            "strict.predictions, restricted by formula membership to the 36 UNSEEN_GLASS formulas "
            "from artifacts/historical/phase1_glass_split.json. bootstrap_regime_pair_delta_ci is "
            "reused "
            "unchanged (same function, same seed 20260921, same n_boot 2000) from "
            "src/sse_transfer/evaluation/regime_utility.py."
        ),
    }
    guard_output(OUTPUT_PATH).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")
    for pair_name, v in verdict_by_pair.items():
        print(f"  {pair_name} verdict: {v['claim']}")


if __name__ == "__main__":
    main()
