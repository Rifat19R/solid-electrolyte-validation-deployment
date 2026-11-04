#!/usr/bin/env python3
"""Corrective 01, step 2: run N6's one-shot external transfer for
`pretrained_element_embedding` only, using the corrected ladder file
(artifacts/authoritative/model_ladder_corrected.json)
instead of the historical frozen ladder.

This imports `n6_external_transfer` as a module and monkeypatches its
module-level `FROZEN_LADDER_PATH` constant before calling
`run_one_model("pretrained_element_embedding")` -- `run_one_model` reads
`FROZEN_LADDER_PATH` at call time (a global lookup inside the function body),
so this substitution is safe and does not require editing
`n6_external_transfer.py` itself. Critically, this NEVER calls
`merge_fragments`/`check_run_authorization` -- it writes only the one
per-model fragment file
(`artifacts/fragments/n6/pretrained_element_embedding.json`), leaving
the locked historical `artifacts/frozen/n6_external_transfer_primary.json` (run_count=1,
authorized=true) completely untouched.

Run with: python scripts/analysis/corrective01_run_n6_pretrained_element_embedding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "transport_analysis"))

import n6_external_transfer as n6  # noqa: E402

CORRECTED_LADDER = ROOT / "artifacts" / "authoritative" / "model_ladder_corrected.json"

if __name__ == "__main__":
    assert CORRECTED_LADDER.exists(), (
        f"missing {CORRECTED_LADDER}; run corrective01_retune_... first"
    )
    n6.FROZEN_LADDER_PATH = CORRECTED_LADDER
    print(f"[corrective01] FROZEN_LADDER_PATH monkeypatched to {CORRECTED_LADDER}")
    n6.run_one_model("pretrained_element_embedding")
