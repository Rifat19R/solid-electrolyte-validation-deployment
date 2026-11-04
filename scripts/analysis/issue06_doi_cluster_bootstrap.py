#!/usr/bin/env python3
"""Reviewer remediation Issue 06: DOI/source-cluster bootstrap.

Spec: reports/remediation/specs/issue06_doi_cluster_bootstrap.md (hashed before
this script was written; see reports/remediation/specs/HASHES.txt).

Pure post-hoc analysis over already-frozen external predictions
(artifacts/frozen/n6_external_transfer_primary.json) and the existing overlap ledger
(data/processed/external_cohort/hargreaves_overlap_ledger.csv). No
model is fit, no new inference is run, no target is read beyond what N6
already froze.

Run with: python scripts/analysis/issue06_doi_cluster_bootstrap.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.evaluation.regime_utility import (  # noqa: E402
    cluster_bootstrap_regime_pair_delta_ci,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

LEDGER_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_overlap_ledger.csv"
N4B_PATH = ROOT / "artifacts" / "authoritative" / "n4b_nested_internal_validation_corrected.json"
N6_PATH = ROOT / "artifacts" / "authoritative" / "n6_external_transfer_corrected.json"
OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "issue06_doi_cluster_bootstrap_corrected.json"
REPORT_PATH = ROOT / "reports" / "rerun" / "issue06_doi_cluster_bootstrap.md"

BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
STATISTICS = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]
REGIMES = ["B_family_group_kfold", "C_lomfo"]
REGIME_LABELS = {"B_family_group_kfold": "B", "C_lomfo": "C"}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_formula_clusters() -> tuple[dict[str, int], int, int]:
    """Connected-component clusters over the bipartite {formula} x {DOI}
    graph of the 213 strict-cohort pre-aggregation retained rows. Returns
    (formula -> cluster_id, n_formulas, n_dois), asserting the SI Note 4
    counts (204 formulas, 91 DOIs) as a sanity check."""
    df = pd.read_csv(LEDGER_PATH)
    strict_pre_agg = df[
        df["in_rt_window"]
        & df["valid_target"]
        & (df["exclusion_reason"] == "RETAINED")
        & (df["overlap_classification"] == "NO_MATCH")
    ]
    n_formulas = strict_pre_agg["reduced_formula"].nunique()
    n_dois = strict_pre_agg["source"].nunique()
    if n_formulas != 204 or n_dois != 91:
        raise RuntimeError(
            f"acs06: expected 204 formulas / 91 DOIs (SI Note 4), got {n_formulas}/{n_dois}"
        )

    uf = UnionFind()
    for _, row in strict_pre_agg.iterrows():
        formula_node = f"F::{row['reduced_formula']}"
        doi_node = f"D::{row['source']}"
        uf.union(formula_node, doi_node)

    formula_to_cluster: dict[str, str] = {}
    for formula in strict_pre_agg["reduced_formula"].unique():
        formula_to_cluster[formula] = uf.find(f"F::{formula}")

    cluster_ids = sorted(set(formula_to_cluster.values()))
    cluster_index = {c: i for i, c in enumerate(cluster_ids)}
    formula_to_cluster_id = {f: cluster_index[c] for f, c in formula_to_cluster.items()}
    return formula_to_cluster_id, n_formulas, n_dois


def load_n6_external() -> tuple[dict[str, float], dict[str, np.ndarray], list[str]]:
    data = json.loads(N6_PATH.read_text(encoding="utf-8"))
    mae_by_model: dict[str, float] = {}
    residuals_by_model: dict[str, np.ndarray] = {}
    formula_order: list[str] | None = None
    for model, payload in data["results"].items():
        strict = payload["strict"]
        mae_by_model[model] = float(strict["mae"])
        residuals_by_model[model] = np.array(strict["predictions"]["residual"], dtype=float)
        this_order = list(strict["predictions"]["reduced_formula"])
        if formula_order is None:
            formula_order = this_order
        elif formula_order != this_order:
            raise RuntimeError(f"acs06: row order mismatch for model {model!r}")
    assert formula_order is not None
    return mae_by_model, residuals_by_model, formula_order


def main() -> None:
    formula_to_cluster_id, n_formulas, n_dois = build_formula_clusters()
    external_mae, external_residuals, formula_order = load_n6_external()

    row_cluster_id = np.array([formula_to_cluster_id[f] for f in formula_order])
    n_clusters = len(set(row_cluster_id.tolist()))
    cluster_sizes = pd.Series(row_cluster_id).value_counts()

    n4b = json.loads(N4B_PATH.read_text(encoding="utf-8"))
    models = sorted(n4b["results"].keys())
    internal_by_regime = {
        regime: {m: float(n4b["results"][m][regime]["pooled_mae"]) for m in models}
        if regime != "C_lomfo"
        else {
            m: float(n4b["results"][m][regime]["pooled_mae_over_major_family_rows"]) for m in models
        }
        for regime in ["A_random_5fold_cv", *REGIMES]
    }

    # Row-level vs cluster-level external MAE CI, per model.
    row_level_mae_ci: dict[str, dict[str, float]] = {}
    cluster_level_mae_ci: dict[str, dict[str, float]] = {}
    rng_row = np.random.default_rng(BOOTSTRAP_SEED)
    rng_cluster = np.random.default_rng(BOOTSTRAP_SEED)
    n_rows = len(formula_order)
    unique_clusters = np.unique(row_cluster_id)
    cluster_row_idx = {c: np.where(row_cluster_id == c)[0] for c in unique_clusters}
    for model, arr in external_residuals.items():
        row_boot = np.array(
            [np.mean(np.abs(arr[rng_row.integers(0, n_rows, size=n_rows)])) for _ in range(N_BOOT)]
        )
        cluster_boot = []
        for _ in range(N_BOOT):
            sampled = rng_cluster.choice(unique_clusters, size=len(unique_clusters), replace=True)
            idx = np.concatenate([cluster_row_idx[c] for c in sampled])
            cluster_boot.append(np.mean(np.abs(arr[idx])))
        cluster_boot = np.array(cluster_boot)
        row_level_mae_ci[model] = {
            "point": float(np.mean(np.abs(arr))),
            "ci_low": float(np.quantile(row_boot, 0.025)),
            "ci_high": float(np.quantile(row_boot, 0.975)),
        }
        cluster_level_mae_ci[model] = {
            "point": float(np.mean(np.abs(arr))),
            "ci_low": float(np.quantile(cluster_boot, 0.025)),
            "ci_high": float(np.quantile(cluster_boot, 0.975)),
        }

    pairwise_cluster: dict[str, dict[str, dict]] = {}
    for regime in REGIMES:
        label = REGIME_LABELS[regime]
        pairwise_cluster[f"{label}_minus_A"] = {
            stat: cluster_bootstrap_regime_pair_delta_ci(
                internal_by_regime["A_random_5fold_cv"],
                internal_by_regime[regime],
                external_residuals,
                row_cluster_id,
                statistic=stat,
                seed=BOOTSTRAP_SEED,
                n_boot=N_BOOT,
            )
            for stat in STATISTICS
        }

    # Load the existing row-level pairwise CIs (N7b) for side-by-side comparison.
    n7b = json.loads(
        (
            ROOT / "artifacts" / "authoritative" / "n7b_regime_utility_nested_corrected.json"
        ).read_text()
    )

    output = {
        "phase": "acs_issue06_doi_cluster_bootstrap",
        "spec_path": "reports/remediation/specs/issue06_doi_cluster_bootstrap.md",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "ledger_source": str(LEDGER_PATH.relative_to(ROOT)),
        "n6_source": str(N6_PATH.relative_to(ROOT)),
        "n_formulas": n_formulas,
        "n_dois": n_dois,
        "n_clusters": n_clusters,
        "max_cluster_size": int(cluster_sizes.max()),
        "cluster_size_distribution": cluster_sizes.value_counts().sort_index().to_dict(),
        "external_mae_row_level_ci": row_level_mae_ci,
        "external_mae_cluster_level_ci": cluster_level_mae_ci,
        "pairwise_bootstrap_ci_cluster_level": pairwise_cluster,
        "pairwise_bootstrap_ci_row_level_reference": n7b["pairwise_bootstrap_ci"],
    }
    guard_output(OUTPUT_PATH).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")
    print(f"n_clusters={n_clusters}, max_cluster_size={cluster_sizes.max()}")

    lines = [
        "# Issue 06 — DOI/source-cluster bootstrap",
        "",
        "**Spec:** `reports/remediation/specs/issue06_doi_cluster_bootstrap.md`.",
        "",
        f"204 strict-cohort formulas collapse into **{n_clusters} source clusters** "
        f"(largest = {int(cluster_sizes.max())} rows) via connected components over the "
        f"formula–DOI bipartite graph of the 213 pre-aggregation retained rows "
        f"(91 unique DOIs) -- reproduces SI Note 4's counts exactly.",
        "",
        "## External MAE: row-level vs. cluster-level 95% bootstrap CI",
        "",
        "| Model | Point | Row-level CI | Cluster-level CI | Cluster CI wider |",
        "| --- | --- | --- | --- | --- |",
    ]
    for model in sorted(row_level_mae_ci):
        r = row_level_mae_ci[model]
        c = cluster_level_mae_ci[model]
        r_width = r["ci_high"] - r["ci_low"]
        c_width = c["ci_high"] - c["ci_low"]
        wider = "Yes" if c_width > r_width else "No"
        lines.append(
            f"| {model} | {r['point']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | "
            f"[{c['ci_low']:.4f}, {c['ci_high']:.4f}] | {wider} |"
        )
    lines.append("")
    lines.append("## Regime-utility pairwise deltas vs. A: row-level vs. cluster-level CI")
    lines.append("")
    lines.append(
        "| Comparison | Statistic | Point | Row-level CI (excl. 0) | "
        "Cluster CI (excl. 0) | Downgraded |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for key, stats in pairwise_cluster.items():
        row_key = key  # same naming convention (e.g. "B_minus_A")
        for stat, res in stats.items():
            row_ref = n7b["pairwise_bootstrap_ci"].get(row_key, {}).get(stat)
            if row_ref is None:
                continue
            row_excl = row_ref["ci_low"] * row_ref["ci_high"] > 0
            cluster_excl = res["ci_low"] * res["ci_high"] > 0
            downgraded = "Yes" if row_excl and not cluster_excl else "No"
            lines.append(
                f"| {key.replace('_minus_A', ' minus A')} | {stat} | {res['point_delta']:.4f} | "
                f"[{row_ref['ci_low']:.4f}, {row_ref['ci_high']:.4f}] "
                f"({'excl' if row_excl else 'incl'} 0) | "
                f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}] "
                f"({'excl' if cluster_excl else 'incl'} 0) | "
                f"{downgraded} |"
            )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    guard_output(REPORT_PATH).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
