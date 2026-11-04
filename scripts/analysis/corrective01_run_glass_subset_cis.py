#!/usr/bin/env python3
"""Corrective 01: rerun phase2_0_glass_subset_regime_cis.py's pairwise
between-regime bootstrap delta CIs for the UNSEEN_GLASS subset (n=36)
against the corrected N6/N4B/N20B views (feeds main-text Fig. 4a / the
'For the glass subset...' paragraph).

Run with: python scripts/analysis/corrective01_run_glass_subset_cis.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "transport_analysis"))

CORR = ROOT / "artifacts" / "authoritative"
RERUN = ROOT / "artifacts" / "rerun"  # outputs; the authoritative record is never rewritten
N6_CORRECTED = CORR / "n6_external_transfer_corrected.json"
N4B_CORRECTED = CORR / "n4b_nested_internal_validation_corrected.json"
N20B_CORRECTED = CORR / "n20b_formula_group_kfold_corrected.json"
GLASS_SPLIT_CORRECTED = CORR / "phase1_glass_split_corrected.json"

for p in [N6_CORRECTED, N4B_CORRECTED, N20B_CORRECTED, GLASS_SPLIT_CORRECTED]:
    assert p.exists(), f"missing {p}"

mod = importlib.import_module("phase2_0_glass_subset_regime_cis")
mod.N6_PATH = N6_CORRECTED
mod.N4B_PATH = N4B_CORRECTED
mod.N20B_PATH = N20B_CORRECTED
mod.GLASS_SPLIT_PATH = GLASS_SPLIT_CORRECTED
mod.OUTPUT_PATH = RERUN / "phase2_glass_subset_regime_cis_corrected.json"

if __name__ == "__main__":
    mod.main()
    print(f"[corrective01] wrote {mod.OUTPUT_PATH.relative_to(ROOT)}")
