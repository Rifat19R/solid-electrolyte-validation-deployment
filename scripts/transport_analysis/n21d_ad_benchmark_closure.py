#!/usr/bin/env python3
"""Cross-dataset study, N21 Section D: applicability-domain (AD)
benchmark closure (reviewer-closure task, Section D).

Main question: does the composition-space nearest-neighbor AD score (N8)
provide risk stratification beyond a random-rejection baseline, and is its
signal merely family novelty (seen vs. unseen chemistry family, N20 Part A)?

Everything here is computed from already-frozen predictions/features:
  - AD score: `uncertainty.applicability_domain.nearest_neighbor_ad_score`,
    recomputed identically to N8 (same `TrainFitCompositionScaler` fit on the
    449-row OBELiX development population, same deterministic feature
    transform -- not a model fit, not new inference; N8 did not persist its
    per-row scores, only summary statistics, so they are regenerated here
    from the same frozen source composition strings).
  - Family novelty: `artifacts/historical/reviewer_family_audit.json` (N20 Part A,
    already frozen).
  - Cross-model prediction dispersion: std of the 8 models' y_pred per row
    from `artifacts/frozen/n6_external_transfer_primary.json` (already-frozen
    predictions only). This substitutes for a literal "five-seed" dispersion
    score: N6's per-seed predictions were averaged before being written to
    disk and the per-row 5-seed std was never persisted (only its pooled
    mean/median), so recovering it would require re-fitting each of N6's 40
    (8 model x 5 seed) external fits -- forbidden this pass ("do NOT rerun
    N6 external inference"). Cross-model disagreement is the closest
    zero-refit proxy for "ensemble-style predictive dispersion" available
    from already-frozen artifacts; this substitution is stated here and in
    the report, not silently made.
  - kNN-density / KDE comparator: predefined BEFORE viewing any result (per
    task instruction) as a Gaussian KDE (`scipy.stats.gaussian_kde`, default
    Scott bandwidth) fit on the same standardized reference feature space,
    score = negative log-density at each query row (higher = more novel).
  - Random-rejection baseline: bootstrap distribution (n_boot=2000,
    seed=20260921) of MAE at each retained fraction from uniformly random
    row subsets (no score involved at all).

Coverage grid fixed in advance to N8's own primary grid (1.0, 0.75, 0.5,
0.25) for direct comparability with the already-reported headline result --
not tuned or chosen after seeing any of this script's own numbers.

Run:
    python scripts/transport_analysis/n21d_ad_benchmark_closure.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.data.obelix_loader import load_obelix  # noqa: E402
from sse_transfer.data.regression_dataset import load_regression_dataset  # noqa: E402
from sse_transfer.evaluation.n15_risk_coverage_aurc import (  # noqa: E402
    aurc_from_risk_coverage,
)
from sse_transfer.features.composition_pipeline import TrainFitCompositionScaler  # noqa: E402
from sse_transfer.uncertainty.applicability_domain import (  # noqa: E402
    fit_standardization,
    nearest_neighbor_ad_score,
)
from sse_transfer.uncertainty.risk_coverage import (  # noqa: E402
    risk_coverage_mae_at_retained_fraction,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
FAMILY_AUDIT_PATH = (
    ROOT / "artifacts" / "authoritative" / "n20_artifacts" / "reviewer_family_audit.json"
)
STRICT_COHORT_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_strict.csv"
OUTPUT_JSON = ROOT / "artifacts" / "rerun" / "n21d_ad_benchmark_closure_corrected.json"
REPORT_MD = ROOT / "reports" / "rerun" / "n21d_ad_benchmark_closure.md"

RETAINED_FRACTIONS: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
BOOTSTRAP_SEED = 20260921
N_BOOT = 2000


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def load_obelix_dev_compositions() -> pd.Series:
    dataset = load_regression_dataset()
    raw = load_obelix("all").reset_index(drop=True)
    raw["ID"] = raw["ID"].astype(str)
    raw_by_id = raw.set_index("ID")
    dev_ids = pd.concat([dataset.train["ID"], dataset.validation["ID"]], ignore_index=True)
    return raw_by_id.loc[dev_ids, "True Composition"].reset_index(drop=True)


def random_rejection_bootstrap(
    abs_residual: np.ndarray, fractions: tuple[float, ...], *, seed: int, n_boot: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = len(abs_residual)
    out = []
    for f in fractions:
        k = max(1, round(n * f))
        maes = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.choice(n, size=k, replace=False)
            maes[b] = float(np.mean(abs_residual[idx]))
        out.append(
            {
                "retained_fraction": f,
                "n_retained": k,
                "mae_mean": float(np.mean(maes)),
                "mae_ci_low": float(np.percentile(maes, 2.5)),
                "mae_ci_high": float(np.percentile(maes, 97.5)),
            }
        )
    return out


def spearman_signed(score: np.ndarray, abs_err: np.ndarray) -> dict[str, float]:
    if np.allclose(score, score[0]):
        return {"spearman_r": float("nan"), "p_value": float("nan")}
    rho, p = scipy_stats.spearmanr(score, abs_err)
    return {"spearman_r": float(rho), "p_value": float(p)}


def main() -> None:
    n6 = json.loads(N6_PATH.read_text(encoding="utf-8"))
    family = json.loads(FAMILY_AUDIT_PATH.read_text(encoding="utf-8"))
    strict_cohort = pd.read_csv(STRICT_COHORT_PATH)

    family_status: dict[str, str] = {}
    for f in family["seen_family_formulas"]:
        family_status[f] = "SEEN"
    for f in family["unseen_family_formulas"]:
        family_status[f] = "UNSEEN"
    for f in family["ambiguous_or_missing_family_formulas"]:
        family_status[f] = "AMBIGUOUS"

    obelix_dev_compositions = load_obelix_dev_compositions()
    scaler = TrainFitCompositionScaler().fit(obelix_dev_compositions)
    x_ref = scaler.transform(obelix_dev_compositions)
    x_query = scaler.transform(strict_cohort["reduced_formula"])

    ad_stats = fit_standardization(x_ref)
    ad_scores_by_row = nearest_neighbor_ad_score(x_ref, x_query, ad_stats)

    z_ref = (x_ref.to_numpy(dtype=float) - ad_stats["mean"]) / ad_stats["std"]
    z_query = (x_query.to_numpy(dtype=float) - ad_stats["mean"]) / ad_stats["std"]
    # 132 standardized features vs. 449 reference rows gives a singular/near-singular
    # full-covariance Gaussian KDE (collinear Magpie features); predefined fix (chosen
    # for numerical conditioning, before any KDE result was viewed): PCA fit on the
    # reference rows only, retaining components explaining 95% of reference variance,
    # then KDE in that reduced space. Reference-only fit; query rows are projected,
    # never used to fit the PCA.
    kde_keep = np.std(z_ref, axis=0) > 1e-8
    z_ref_nz = z_ref[:, kde_keep]
    z_query_nz = z_query[:, kde_keep]
    ref_mean = z_ref_nz.mean(axis=0)
    u, s, vt = np.linalg.svd(z_ref_nz - ref_mean, full_matrices=False)
    explained = (s**2) / np.sum(s**2)
    n_components = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
    components = vt[:n_components]
    pca_ref = (z_ref_nz - ref_mean) @ components.T
    pca_query = (z_query_nz - ref_mean) @ components.T
    kde = scipy_stats.gaussian_kde(pca_ref.T)
    log_density = kde.logpdf(pca_query.T)
    kde_novelty_score_by_row = -np.asarray(log_density, dtype=float)
    kde_n_pca_components = n_components

    cohort_index = {rf: i for i, rf in enumerate(strict_cohort["reduced_formula"])}
    family_score_by_row = np.array(
        [
            {"SEEN": 0.0, "AMBIGUOUS": 1.0, "UNSEEN": 2.0}[family_status.get(rf, "AMBIGUOUS")]
            for rf in strict_cohort["reduced_formula"]
        ]
    )

    # cross-model dispersion: std of the 8 models' y_pred per row, aligned by formula
    pred_rows: list[np.ndarray] = []
    for _model_name, payload in n6["results"].items():
        strict = payload["strict"]
        order = np.array([cohort_index[rf] for rf in strict["predictions"]["reduced_formula"]])
        y_pred_aligned = np.empty(len(order))
        y_pred_aligned[order] = np.array(strict["predictions"]["y_pred"])
        pred_rows.append(y_pred_aligned)
    pred_matrix = np.stack(pred_rows)  # (8, 204), row order = strict_cohort order
    cross_model_dispersion_by_row = pred_matrix.std(axis=0)

    per_model: dict[str, Any] = {}
    for model_name, payload in n6["results"].items():
        strict = payload["strict"]
        row_ids = strict["predictions"]["reduced_formula"]
        order = np.array([cohort_index[rf] for rf in row_ids])
        y_true = np.array(strict["predictions"]["y_true"])
        y_pred = np.array(strict["predictions"]["y_pred"])
        abs_residual = np.abs(y_pred - y_true)  # already in row_ids order

        ad_scores = ad_scores_by_row[order]
        kde_scores = kde_novelty_score_by_row[order]
        fam_scores = family_score_by_row[order]
        disp_scores = cross_model_dispersion_by_row[order]

        comparators = {
            "ad_nn_distance": ad_scores,
            "kde_novelty": kde_scores,
            "family_novelty": fam_scores,
            "cross_model_dispersion": disp_scores,
        }
        comparator_results: dict[str, Any] = {}
        for name, scores in comparators.items():
            rc = risk_coverage_mae_at_retained_fraction(scores, abs_residual, RETAINED_FRACTIONS)
            aurc = aurc_from_risk_coverage(rc)
            corr = spearman_signed(scores, abs_residual)
            comparator_results[name] = {
                "risk_coverage_mae": rc,
                "aurc": aurc["aurc"],
                "spearman_r": corr["spearman_r"],
                "spearman_p": corr["p_value"],
            }

        random_rc = random_rejection_bootstrap(
            abs_residual, RETAINED_FRACTIONS, seed=BOOTSTRAP_SEED, n_boot=N_BOOT
        )
        random_aurc = aurc_from_risk_coverage(
            [{"retained_fraction": r["retained_fraction"], "mae": r["mae_mean"]} for r in random_rc]
        )
        comparator_results["random_rejection"] = {
            "risk_coverage_mae_bootstrap": random_rc,
            "aurc": random_aurc["aurc"],
        }

        per_model[model_name] = comparator_results

    # summary: for each real comparator, in how many of the 8 models does it beat
    # (lower AURC than) random rejection, and beat family_novelty alone?
    summary: dict[str, Any] = {}
    for name in ["ad_nn_distance", "kde_novelty", "cross_model_dispersion"]:
        beats_random = sum(
            1
            for m in per_model
            if per_model[m][name]["aurc"] < per_model[m]["random_rejection"]["aurc"]
        )
        beats_family = sum(
            1
            for m in per_model
            if per_model[m][name]["aurc"] < per_model[m]["family_novelty"]["aurc"]
        )
        mean_aurc = float(np.mean([per_model[m][name]["aurc"] for m in per_model]))
        n_positive_spearman = sum(
            1
            for m in per_model
            if not np.isnan(per_model[m][name]["spearman_r"])
            and per_model[m][name]["spearman_r"] > 0
        )
        summary[name] = {
            "mean_aurc_across_8_models": mean_aurc,
            "n_models_beating_random_rejection_aurc": beats_random,
            "n_models_beating_family_novelty_aurc": beats_family,
            "n_models_positive_spearman": n_positive_spearman,
        }
    summary["family_novelty"] = {
        "mean_aurc_across_8_models": float(
            np.mean([per_model[m]["family_novelty"]["aurc"] for m in per_model])
        ),
        "n_seen": family["n_seen_family"],
        "n_unseen": family["n_unseen_family"],
        "n_ambiguous": family["n_ambiguous_or_missing_family"],
    }
    summary["random_rejection"] = {
        "mean_aurc_across_8_models": float(
            np.mean([per_model[m]["random_rejection"]["aurc"] for m in per_model])
        )
    }

    manifest = {
        "phase": "N21_section_d_ad_benchmark_closure",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "n6_source": str(N6_PATH.relative_to(ROOT)),
        "n6_rerun": False,
        "family_audit_source": str(FAMILY_AUDIT_PATH.relative_to(ROOT)),
        "retained_fractions": list(RETAINED_FRACTIONS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_boot": N_BOOT,
        "kde_bandwidth_method": "scott (scipy default)",
        "kde_zero_variance_dims_dropped": int((~kde_keep).sum()),
        "kde_pca_n_components_95pct_variance": kde_n_pca_components,
        "cross_model_dispersion_note": (
            "Substitute for literal five-seed dispersion: N6's per-seed predictions were "
            "averaged before being persisted and per-row 5-seed std was never saved; "
            "recomputing it would require re-fitting N6's 40 external fits, forbidden this "
            "pass. std of the 8 already-frozen model predictions per row is used instead."
        ),
        "per_model": per_model,
        "summary": summary,
    }
    guard_output(OUTPUT_JSON).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {OUTPUT_JSON.relative_to(ROOT)}")
    _write_report(manifest)
    print(f"wrote {REPORT_MD.relative_to(ROOT)}")


def _write_report(manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# N21 Section D — Applicability-Domain Benchmark Closure\n\n")
    lines.append(f"Git commit: `{manifest['git_commit']}` on branch `{manifest['git_branch']}`\n\n")
    lines.append(
        f"Coverage grid (fixed in advance): {manifest['retained_fractions']}. N6 was not "
        f"rerun. KDE: {manifest['kde_bandwidth_method']}, "
        f"{manifest['kde_zero_variance_dims_dropped']} near-zero-variance reference dims "
        f"dropped, then PCA to {manifest['kde_pca_n_components_95pct_variance']} components "
        "(95% reference variance, reference-only fit) before fitting.\n\n"
    )
    lines.append(
        f"**Cross-model-dispersion substitution note:** "
        f"{manifest['cross_model_dispersion_note']}\n\n"
    )

    lines.append("## Summary: mean AURC across 8 models (lower = better risk stratification)\n\n")
    lines.append(
        "| Comparator | Mean AURC | Beats random rejection (n models) | "
        "Beats family novelty (n models) | Positive Spearman (n models) |\n"
    )
    lines.append("|---|---|---|---|---|\n")
    s = manifest["summary"]
    for name in ["ad_nn_distance", "kde_novelty", "cross_model_dispersion"]:
        d = s[name]
        lines.append(
            f"| {name} | {d['mean_aurc_across_8_models']:.4f} | "
            f"{d['n_models_beating_random_rejection_aurc']}/8 | "
            f"{d['n_models_beating_family_novelty_aurc']}/8 | "
            f"{d['n_models_positive_spearman']}/8 |\n"
        )
    lines.append(
        f"| family_novelty | {s['family_novelty']['mean_aurc_across_8_models']:.4f} | - | - | - |\n"
    )
    lines.append(
        f"| random_rejection | {s['random_rejection']['mean_aurc_across_8_models']:.4f} | "
        f"- | - | - |\n"
    )
    lines.append(
        f"\nfamily_novelty: SEEN n={s['family_novelty']['n_seen']}, "
        f"UNSEEN n={s['family_novelty']['n_unseen']}, "
        f"AMBIGUOUS n={s['family_novelty']['n_ambiguous']} (scored 0/2/1 respectively).\n"
    )

    lines.append("\n## Per-model AURC, all five comparators\n\n")
    lines.append(
        "| Model | AD (NN-dist) | KDE novelty | Family novelty | "
        "Cross-model dispersion | Random rejection |\n"
    )
    lines.append("|---|---|---|---|---|---|\n")
    for model, d in manifest["per_model"].items():
        lines.append(
            f"| {model} | {d['ad_nn_distance']['aurc']:.4f} | {d['kde_novelty']['aurc']:.4f} | "
            f"{d['family_novelty']['aurc']:.4f} | {d['cross_model_dispersion']['aurc']:.4f} | "
            f"{d['random_rejection']['aurc']:.4f} |\n"
        )

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_MD).write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
