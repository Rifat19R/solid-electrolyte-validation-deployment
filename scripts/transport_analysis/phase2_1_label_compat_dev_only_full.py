#!/usr/bin/env python3
"""Cross-dataset study, Phase 2 Stage 1: full aggregate paired-formula
label-compatibility statistics for the 449-row OBELiX DEVELOPMENT population
only (166 paired formulas), completing Phase 1 Stage 2 item 1.

Phase 1 Stage 2 item 1 (`scripts/transport_analysis/phase1_2_factual_fixes.py`) ran the exact
same `paired_label_compatibility` call used here, restricted to the 449-row
dev population, but its saved artifact (`artifacts/historical/phase1_factual_fixes.json`)
only kept the DIFFERENT-SOURCE / SAME-SOURCE group breakdown, not the
aggregate n=166 statistics (mean/median discrepancy, RMSE, Spearman rho,
Pearson r, bootstrap CIs, >1/>2 log10-unit counts) needed to replace the old
full-599-row n=200 statistics "everywhere" in the manuscript per an author
amendment. This script re-runs the identical computation (same function,
same frozen inputs, same seed) purely to persist the aggregate fields that
were computed but not saved -- no new statistic, no new data source, no
external inference, no retraining. The frozen 113-row OBELiX `test` split is
never opened.

Run with: python scripts/transport_analysis/phase2_1_label_compat_dev_only_full.py
"""

from __future__ import annotations

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
    MATCH_EXACT_DOI,
    load_obelix_raw,
    safe_reduced_formula,
)
from sse_transfer.data.obelix_loader import (  # noqa: E402
    LABEL_STATUS_EXACT,
    censoring_bound,
    label_status,
)
from sse_transfer.data.regression_dataset import load_regression_dataset  # noqa: E402
from sse_transfer.evaluation.reviewer_remediation import (  # noqa: E402
    paired_label_compatibility,
)
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

LEDGER_PATH = ROOT / "data" / "processed" / "external_cohort" / "hargreaves_overlap_ledger.csv"
OUTPUT_PATH = ROOT / "artifacts" / "rerun" / "phase2_label_compat_dev_only_full.json"

BOOTSTRAP_SEED = 20260921
N_BOOT = 2000
MIN_N_FOR_CI = 10


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def dev_population_ids() -> set[str]:
    dataset = load_regression_dataset()
    ids = pd.concat([dataset.train["ID"], dataset.validation["ID"]], ignore_index=True)
    return set(ids.astype(str))


def main() -> None:
    dev_ids = dev_population_ids()
    obelix_raw = load_obelix_raw()
    obelix_raw["ID"] = obelix_raw["ID"].astype(str)
    obelix_dev = obelix_raw.loc[obelix_raw["ID"].isin(dev_ids)].copy()
    if len(obelix_dev) != 449:
        raise RuntimeError(f"expected 449 dev rows, got {len(obelix_dev)}")

    conductivity = obelix_dev["Ionic conductivity (S cm-1)"]
    status = label_status(conductivity)
    exact_mask = status == LABEL_STATUS_EXACT
    obelix_exact = obelix_dev.loc[exact_mask].copy()
    obelix_exact["canonical_formula"] = obelix_exact["Reduced Composition"].apply(
        safe_reduced_formula
    )
    obelix_exact["y"] = np.log10(censoring_bound(conductivity.loc[exact_mask]))
    obelix_exact = obelix_exact.dropna(subset=["canonical_formula"])

    obelix_y_by_formula: dict[str, float] = {}
    for formula, group in obelix_exact.groupby("canonical_formula"):
        obelix_y_by_formula[str(formula)] = float(group["y"].median())

    ledger = pd.read_csv(LEDGER_PATH)
    overlap_ledger = ledger.loc[
        ledger["exclusion_reason"].isin(
            ["EXACT_COMPOSITION_AND_DOI_MATCH_WITH_OBELIX", "COMPOSITION_ONLY_MATCH_WITH_OBELIX"]
        )
    ].copy()
    hargreaves_by_formula: dict[str, float] = {}
    hargreaves_meta: dict[str, dict[str, Any]] = {}
    for formula, group in overlap_ledger.groupby("reduced_formula"):
        hargreaves_by_formula[str(formula)] = float(group["log_target"].median())
        hargreaves_meta[str(formula)] = {
            "overlap_classification": sorted(group["overlap_classification"].unique().tolist())
        }

    compatibility = paired_label_compatibility(
        obelix_y_by_formula,
        hargreaves_by_formula,
        seed=BOOTSTRAP_SEED,
        n_boot=N_BOOT,
        min_n_for_ci=MIN_N_FOR_CI,
    )
    shared = compatibility["reduced_formulas"]
    n_same_source = sum(
        1 for f in shared if hargreaves_meta[f]["overlap_classification"] == [MATCH_EXACT_DOI]
    )
    n_different_source = len(shared) - n_same_source

    output = {
        "phase": "phase2_stage1_label_compat_dev_only_full",
        "git_commit": _git("rev-parse", "HEAD"),
        "scope": (
            "OBELiX side restricted to the 449-row development population (train+validation) "
            "only; the frozen 113-row test split was never read for this comparison. Reruns "
            "the identical computation as phase1_2_factual_fixes.py item 1, persisting the full "
            "aggregate fields that pass omitted."
        ),
        "n_dev_rows": 449,
        "n_same_source_exact_doi": n_same_source,
        "n_different_source_composition_only": n_different_source,
        "aggregate": compatibility,
    }
    guard_output(OUTPUT_PATH).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"  n_paired_formulas={compatibility['n_paired_formulas']}")
    print(f"  mean_absolute_discrepancy={compatibility['mean_absolute_discrepancy']:.4f}")
    print(f"  median_absolute_discrepancy={compatibility['median_absolute_discrepancy']:.6f}")
    print(f"  spearman_rho={compatibility['spearman_rho']:.4f}")
    print(f"  pearson_r={compatibility['pearson_r']:.4f}")


if __name__ == "__main__":
    main()
