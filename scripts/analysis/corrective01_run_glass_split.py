#!/usr/bin/env python3
"""Corrective 01: rerun phase1_0_1_predeclare_and_glass_split.py's Stage 0/1
(pre-declaration + SEEN/UNSEEN/glass/crystalline per-model subset metrics,
incl. Table 4's bootstrap CIs) against the corrected N6/N4B/N20B views.
Reuses the historical `artifacts/historical/reviewer_family_audit.json` for the
SEEN/UNSEEN/glass/crystalline formula classification itself, which is a
family-label/crosswalk construction independent of any model's predictions
(confirmed by reading `subset_metrics_for_models`/`main`: the classification
never touches N6/N4B/N20B). Only the per-model subset MAE/CI table (which
does read N6 predictions) is therefore affected, and only in the
pretrained_element_embedding row -- the other 7 models' numbers must come
out byte-identical.

Does not touch artifacts/historical/phase1_predeclared.json or
artifacts/historical/phase1_glass_split.json.

Run with: python scripts/analysis/corrective01_run_glass_split.py
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

for p in [N6_CORRECTED, N4B_CORRECTED, N20B_CORRECTED]:
    assert p.exists(), f"missing {p}"

mod = importlib.import_module("phase1_0_1_predeclare_and_glass_split")
mod.N6_PATH = N6_CORRECTED
mod.N4B_PATH = N4B_CORRECTED
mod.N20B_PATH = N20B_CORRECTED
mod.PREDECLARED_PATH = RERUN / "phase1_predeclared_corrected.json"
mod.GLASS_SPLIT_PATH = RERUN / "phase1_glass_split_corrected.json"

if __name__ == "__main__":
    mod.main()
    print(f"[corrective01] wrote {mod.GLASS_SPLIT_PATH.relative_to(ROOT)}")
