#!/usr/bin/env python3
"""Cross-dataset study, Phase 1 (Stage 0 + Stage 1): pre-declaration
and the seen/unseen-glass/unseen-crystalline external-cohort split.

Task brief context: an author-review pass asking (a) whether the existing
family-aware seen/unseen (138/55) split (`artifacts/historical/reviewer_family_audit.json`,
N20 Part A) should be refined by separating the 55 UNSEEN formulas into
glass/glass-ceramic (amorphous/partially-amorphous, no crystal structure) vs
crystalline-but-unseen-family, since "unseen chemistry" and "unseen material
class" are physically different claims and the manuscript has so far only
made the coarser one; and (b) whether a family-aware CV regime's internal MAE
anticipates the harder external subsets better than random CV, per subset.

STAGE 0 (pre-declaration): the subset-size-dependent statistical treatment
and the three-way IF/ELIF/ELSE narrative-framing rule are written to
`artifacts/historical/phase1_predeclared.json` and hashed BEFORE any subset MAE, regime
gap, or AD-score distribution in this module is computed -- this script's own
`main()` writes that file first, uses only the frozen file it wrote (never an
in-memory dict) for every downstream decision, and never re-runs Stage 0
after seeing Stage 1's numbers (EXPERIMENT_PROTOCOL.md "never select ... to
produce a better score").

Because n_crystalline_unseen = 19 turns out to be below the pre-declared
n>=20 threshold for even an MAE+CI treatment, the framing rule's third branch
(report descriptively, state unresolved) fires for that subset by
construction, before any regime-comparison number for it is inspected -- this
script literally skips computing regime-ranking stats for that subset rather
than computing and discarding them, so no such number exists to peek at.

No Hargreaves external inference is rerun (N6's frozen `strict.predictions`
per model are the only source of external residuals). No model is fit here
except the existing, already-tested `TrainFitCompositionScaler` transform used
to build the (unchanged) N8 composition-space AD feature space. The frozen
113-row OBELiX `test` split is never opened. N4b (nested internal-CV) and
N20b (formula-GroupKFold nested internal-CV) pooled-MAE-per-model are reused
unchanged as the "internal" side of every regime-utility statistic.

Run with: python scripts/transport_analysis/phase1_0_1_predeclare_and_glass_split.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.data.hargreaves_cohort import safe_reduced_formula  # noqa: E402
from sse_transfer.data.obelix_loader import load_obelix  # noqa: E402
from sse_transfer.data.regression_dataset import load_regression_dataset  # noqa: E402
from sse_transfer.evaluation.metrics import bootstrap_metric_ci  # noqa: E402
from sse_transfer.evaluation.n15_source_disjoint import (  # noqa: E402
    score_subset_from_frozen_predictions,
)
from sse_transfer.evaluation.regime_utility import regime_summary  # noqa: E402
from sse_transfer.evaluation.reviewer_remediation import assign_formula_family  # noqa: E402
from sse_transfer.features.composition_pipeline import TrainFitCompositionScaler  # noqa: E402
from sse_transfer.uncertainty.applicability_domain import (  # noqa: E402
    fit_standardization,
    nearest_neighbor_ad_score,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

LEDGER_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_overlap_ledger.csv"
STRICT_COHORT_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_strict.csv"
N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
N4B_PATH = ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
N20B_PATH = ROOT / "artifacts" / "authoritative" / "n20b_formula_group_kfold_corrected.json"
FAMILY_AUDIT_PATH = (
    ROOT / "artifacts" / "authoritative" / "n20_artifacts" / "reviewer_family_audit.json"
)

PREDECLARED_PATH = ROOT / "artifacts" / "rerun" / "phase1_predeclared_corrected.json"
GLASS_SPLIT_PATH = ROOT / "artifacts" / "rerun" / "phase1_glass_split_corrected.json"

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
BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
GLASS_FAMILIES = frozenset({"Glass", "Glass-Ceramic"})


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Stage 0: pre-declaration
# ---------------------------------------------------------------------------


def build_predeclared_spec() -> dict[str, Any]:
    return {
        "phase": "phase1_stage0_predeclared",
        "git_commit": _git("rev-parse", "HEAD"),
        "written_before_any_subset_statistic_was_computed": True,
        "subset_size_thresholds": {
            "n_ge_20": "MAE + RMSE + R2 + bootstrap_95pct_CI",
            "n_ge_30": "MAE+CI treatment above, PLUS regime-ranking statistics "
            "(median_abs_mae_gap, spearman_rho, kendall_tau, top_model_selection_regret)",
            "below_20": "descriptive only (n, mean, median, spread) -- no bootstrap CI, "
            "no regime-ranking statistic is computed for a subset this small",
        },
        "conditional_framing_rule": {
            "branch_1": {
                "condition": "crystalline-unseen subset (n>=20, so a regime comparison exists) "
                "shows family-aware internal-CV median |MAE gap| < random-CV's median |MAE gap| "
                "against that subset's external MAE",
                "framing": "family-holdout anticipates unseen-chemistry error magnitude",
            },
            "branch_2": {
                "condition": "the same smaller-gap pattern holds only for the glass/glass-ceramic "
                "subset, not for crystalline-unseen",
                "framing": "unseen material class (amorphous), not chemistry novelty",
            },
            "branch_3": {
                "condition": "crystalline-unseen n is below the n>=20 threshold",
                "framing": "report descriptively; state unresolved",
            },
            "evaluation_order": "branch_3 is checked FIRST (a sample-size gate on what can be "
            "computed at all), then branch_1, then branch_2 -- if branch_3 fires for "
            "crystalline-unseen, branches 1/2 can still independently apply to the glass subset "
            "using its own (independently thresholded) regime comparison, but the crystalline "
            "subset itself is never given a regime-ranking verdict regardless of what its raw "
            "numbers would have shown.",
        },
        "frozen_inputs": {
            "n6_external_transfer_sha256": _sha256(N6_PATH),
            "n4b_nested_internal_validation_sha256": _sha256(N4B_PATH),
            "n20b_formula_group_kfold_sha256": _sha256(N20B_PATH),
            "reviewer_family_audit_sha256": _sha256(FAMILY_AUDIT_PATH),
            "hargreaves_overlap_ledger_sha256": _sha256(LEDGER_PATH),
        },
        "hard_constraints": [
            "no external inference rerun",
            "no retraining of final models",
            "no new model or dataset",
            "frozen predictions/artifacts only",
            "report adverse results",
            "no commit, no push",
        ],
    }


# ---------------------------------------------------------------------------
# Stage 1: glass split
# ---------------------------------------------------------------------------


def load_dev_population_formulas() -> pd.Series:
    """449-row OBELiX development population's canonical (pymatgen-reduced)
    formula -- reused unchanged construction from N8/N20's own loaders."""
    dataset = load_regression_dataset()
    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)
    raw_by_id = raw.set_index("ID")
    dev_ids = pd.concat([dataset.train["ID"], dataset.validation["ID"]], ignore_index=True)
    reduced_raw = raw_by_id.loc[dev_ids, "Reduced Composition"].reset_index(drop=True)
    return reduced_raw.apply(safe_reduced_formula)


def load_dev_compositions_full() -> pd.Series:
    dataset = load_regression_dataset()
    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)
    raw_by_id = raw.set_index("ID")
    dev_ids = pd.concat([dataset.train["ID"], dataset.validation["ID"]], ignore_index=True)
    return raw_by_id.loc[dev_ids, "True Composition"].reset_index(drop=True)


def load_n4b_internal() -> dict[str, dict[str, float]]:
    data = _load_json(N4B_PATH)
    out: dict[str, dict[str, float]] = {r: {} for r in REGIMES}
    for model, regimes in data["results"].items():
        for regime, payload in regimes.items():
            key = "pooled_mae" if "pooled_mae" in payload else "pooled_mae_over_major_family_rows"
            out[regime][model] = float(payload[key])
    return out


def load_n20b_internal() -> dict[str, float]:
    data = _load_json(N20B_PATH)
    return {m: float(data["results"][m]["pooled_mae"]) for m in MODEL_NAMES if m in data["results"]}


def chemical_system(formula: str) -> str | None:
    try:
        elements = frozenset(str(e) for e in Composition(formula).elements)
    except Exception:
        return None
    return "-".join(sorted(elements))


def compute_ad_scores(strict_cohort: pd.DataFrame) -> np.ndarray:
    """Same construction as N8's own AD score (reused unchanged): standardize
    on the 449-row OBELiX dev population's composition features, nearest-
    neighbor Euclidean distance for each strict-cohort query row."""
    obelix_dev_compositions = load_dev_compositions_full()
    scaler = TrainFitCompositionScaler().fit(obelix_dev_compositions)
    x_ref = scaler.transform(obelix_dev_compositions)
    x_query = scaler.transform(strict_cohort["reduced_formula"])
    stats = fit_standardization(x_ref)
    return nearest_neighbor_ad_score(x_ref, x_query, stats)


def subset_metrics_for_models(
    n6: dict[str, Any], formulas: set[str], *, min_n_for_ci: int
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for model in MODEL_NAMES:
        predictions = n6["results"][model]["strict"]["predictions"]
        metrics = score_subset_from_frozen_predictions(predictions, formulas)
        if metrics["n"] >= min_n_for_ci:
            y_true = np.array(predictions["y_true"])
            y_pred = np.array(predictions["y_pred"])
            mask = np.array([f in formulas for f in predictions["reduced_formula"]])
            metrics["mae_bootstrap_ci"] = bootstrap_metric_ci(
                y_true[mask], y_pred[mask], "mae", seed=BOOTSTRAP_SEED, n_boot=N_BOOT
            )
        else:
            metrics["bootstrap_ci_note"] = (
                f"n={metrics['n']} < {min_n_for_ci}: bootstrap CI not computed "
                "(below pre-declared Stage 0 threshold)."
            )
        out[model] = metrics
    return out


def regime_utility_for_subset(
    internal_by_regime: dict[str, dict[str, float]],
    formula_gkf_internal: dict[str, float],
    external_mae_by_model: dict[str, float],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for regime, internal_by_model in internal_by_regime.items():
        out[regime] = regime_summary(internal_by_model, external_mae_by_model)
    out["D_formula_group_kfold"] = regime_summary(formula_gkf_internal, external_mae_by_model)
    return out


def main() -> None:
    print("=== Stage 0: pre-declaration ===")
    predeclared = build_predeclared_spec()
    guard_output(PREDECLARED_PATH).write_text(json.dumps(predeclared, indent=2), encoding="utf-8")
    predeclared_sha256 = _sha256(PREDECLARED_PATH)
    print(f"wrote {PREDECLARED_PATH.relative_to(ROOT)}")
    print(f"  sha256: {predeclared_sha256}")

    print("=== Stage 1: glass split ===")
    ledger = pd.read_csv(LEDGER_PATH)
    retained = ledger.loc[ledger["exclusion_reason"] == "RETAINED"].copy()
    if retained["reduced_formula"].nunique() != 204:
        raise RuntimeError(
            f"expected 204 strict-cohort formulas, got {retained['reduced_formula'].nunique()}"
        )
    formula_family = assign_formula_family(retained)

    family_audit = _load_json(FAMILY_AUDIT_PATH)
    seen_formulas = family_audit["seen_family_formulas"]
    unseen_formulas = family_audit["unseen_family_formulas"]
    if len(seen_formulas) != 138 or len(unseen_formulas) != 55:
        raise RuntimeError(
            f"expected 138 SEEN / 55 UNSEEN from reviewer_family_audit.json, "
            f"got {len(seen_formulas)} / {len(unseen_formulas)}"
        )

    unseen_glass = sorted(
        f for f in unseen_formulas if formula_family[f]["assigned_family"] in GLASS_FAMILIES
    )
    unseen_crystalline = sorted(
        f for f in unseen_formulas if formula_family[f]["assigned_family"] not in GLASS_FAMILIES
    )
    if len(unseen_glass) + len(unseen_crystalline) != len(unseen_formulas):
        raise RuntimeError("glass/crystalline partition of UNSEEN does not sum to 55")

    subsets = {
        "SEEN": set(seen_formulas),
        "UNSEEN_ALL": set(unseen_formulas),
        "UNSEEN_GLASS": set(unseen_glass),
        "UNSEEN_CRYSTALLINE": set(unseen_crystalline),
    }
    subset_n = {k: len(v) for k, v in subsets.items()}

    n6 = _load_json(N6_PATH)
    per_model_subset_metrics: dict[str, dict[str, Any]] = {}
    for label, formulas in subsets.items():
        min_n_for_ci = 20
        per_model_subset_metrics[label] = subset_metrics_for_models(
            n6, formulas, min_n_for_ci=min_n_for_ci
        )

    n4b_internal = load_n4b_internal()
    n20b_internal = load_n20b_internal()

    regime_utility_by_subset: dict[str, Any] = {}
    for label in ("SEEN", "UNSEEN_ALL", "UNSEEN_GLASS"):
        n_subset = subset_n[label]
        if n_subset < 30:
            regime_utility_by_subset[label] = {
                "note": f"n={n_subset} < 30: regime-ranking statistics not computed per "
                "Stage 0 pre-declared threshold."
            }
            continue
        external_mae = {m: per_model_subset_metrics[label][m]["mae"] for m in MODEL_NAMES}
        regime_utility_by_subset[label] = regime_utility_for_subset(
            n4b_internal, n20b_internal, external_mae
        )
    regime_utility_by_subset["UNSEEN_CRYSTALLINE"] = {
        "note": f"n={subset_n['UNSEEN_CRYSTALLINE']} < 20: below even the MAE+CI threshold; "
        "no regime-ranking statistic computed (Stage 0 branch_3 fires by construction)."
    }

    # AD score distributions per subset (composition-space nearest-neighbor
    # distance to the 449-row OBELiX dev population; N8's method, unchanged).
    strict_cohort = pd.read_csv(STRICT_COHORT_PATH)
    ad_scores = compute_ad_scores(strict_cohort)
    formula_to_ad = dict(zip(strict_cohort["reduced_formula"], ad_scores, strict=True))
    ad_distribution_by_subset: dict[str, Any] = {}
    for label, formulas in subsets.items():
        vals = np.array([formula_to_ad[f] for f in formulas if f in formula_to_ad])
        ad_distribution_by_subset[label] = {
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=0)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "p90": float(np.percentile(vals, 90)),
        }

    # Descriptive: fraction of unseen-glass rows sharing a chemical system
    # (pymatgen element set) with an OBELiX dev-population row.
    dev_formulas = load_dev_population_formulas()
    obelix_systems = {chemical_system(f) for f in dev_formulas if chemical_system(f) is not None}
    glass_systems = {f: chemical_system(f) for f in unseen_glass}
    n_glass_sharing_system = sum(
        1 for sysid in glass_systems.values() if sysid is not None and sysid in obelix_systems
    )
    n_glass_parseable = sum(1 for sysid in glass_systems.values() if sysid is not None)
    unseen_glass_chemical_system_overlap = {
        "n_unseen_glass": len(unseen_glass),
        "n_parseable": n_glass_parseable,
        "n_sharing_chemical_system_with_obelix_dev": n_glass_sharing_system,
        "fraction_sharing_chemical_system_with_obelix_dev": (
            n_glass_sharing_system / n_glass_parseable if n_glass_parseable else float("nan")
        ),
        "note": "Chemical system = exact pymatgen element set (order-independent), computed "
        "against the 449-row OBELiX development population only (never the frozen 113-row "
        "test split). Sharing a chemical system is a much weaker condition than SEEN family "
        "status -- e.g. an oxide glass can share every element with a crystalline OBELiX "
        "perovskite while having no structural family match at all.",
    }

    # Framing verdict, applied mechanically per Stage 0's pre-declared rule.
    glass_gap = regime_utility_by_subset.get("UNSEEN_GLASS", {})
    verdict: dict[str, Any] = {"branch_3_fires_for_crystalline_unseen": True}
    if isinstance(glass_gap, dict) and "A_random_5fold_cv" in glass_gap:
        a_gap = glass_gap["A_random_5fold_cv"]["median_abs_mae_gap"]
        b_gap = glass_gap["B_family_group_kfold"]["median_abs_mae_gap"]
        c_gap = glass_gap["C_lomfo"]["median_abs_mae_gap"]
        family_aware_smaller = (b_gap < a_gap) or (c_gap < a_gap)
        verdict["unseen_glass_median_abs_mae_gap"] = {
            "A_random_5fold_cv": a_gap,
            "B_family_group_kfold": b_gap,
            "C_lomfo": c_gap,
        }
        verdict["branch_2_fires_glass_only_pattern"] = family_aware_smaller
        verdict["framing"] = (
            "unseen material class (amorphous), not chemistry novelty"
            if family_aware_smaller
            else "no family-aware advantage detected for the glass subset either; "
            "crystalline-unseen is unresolved by construction (n<20) -- neither branch 1 "
            "nor branch 2 fires, this remains a negative/inconclusive result for the "
            "glass-split refinement, reported as such, not forced into a framing."
        )
    else:
        verdict["framing"] = "glass subset also below regime-ranking threshold; fully descriptive."

    output = {
        "phase": "phase1_stage1_glass_split",
        "predeclared_spec_sha256": predeclared_sha256,
        "git_commit": _git("rev-parse", "HEAD"),
        "n_strict_cohort_formulas": 204,
        "n_seen_family": len(seen_formulas),
        "n_unseen_family_all": len(unseen_formulas),
        "n_unseen_glass_or_glass_ceramic": len(unseen_glass),
        "n_unseen_crystalline": len(unseen_crystalline),
        "unseen_glass_formulas": unseen_glass,
        "unseen_crystalline_formulas": unseen_crystalline,
        "unseen_crystalline_raw_family_breakdown": {
            fam: sum(1 for f in unseen_crystalline if formula_family[f]["assigned_family"] == fam)
            for fam in sorted({formula_family[f]["assigned_family"] for f in unseen_crystalline})
        },
        "per_model_subset_metrics": per_model_subset_metrics,
        "regime_utility_by_subset": regime_utility_by_subset,
        "ad_score_distribution_by_subset": ad_distribution_by_subset,
        "unseen_glass_chemical_system_overlap_with_obelix_dev": (
            unseen_glass_chemical_system_overlap
        ),
        "stage0_framing_rule_verdict": verdict,
    }
    guard_output(GLASS_SPLIT_PATH).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {GLASS_SPLIT_PATH.relative_to(ROOT)}")
    print(
        f"  n_seen={len(seen_formulas)} n_unseen_glass={len(unseen_glass)} "
        f"n_unseen_crystalline={len(unseen_crystalline)}"
    )
    print(f"  framing verdict: {verdict['framing']}")


if __name__ == "__main__":
    main()
