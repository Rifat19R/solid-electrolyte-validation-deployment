#!/usr/bin/env python3
"""Corrective 01: merge the 7 untouched historical Issue-09-Variant-B
fragments with the 1 corrected `pretrained_element_embedding` fragment into
a NEW, non-historical artifact, then recompute regime-utility statistics
for B' (acs09b_regime_utility.py's own logic) against the corrected N4B/N6.

Does NOT call acs09b_missing_singleton_groupkfold.merge_fragments() (which
writes to the historical artifacts/historical/issue09b_missing_singleton_groupkfold.json)
or acs09b_regime_utility.main() unpatched (which reads the historical
B'/N4B/N6 paths). Reuses both modules' own logic via monkeypatched
module-level path constants, same pattern as corrective01_merge_*.py.

Run with: python scripts/analysis/corrective01_merge_issue09b.py
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from sse_transfer.models.ladder import MODEL_NAMES  # noqa: E402
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

FRAGMENTS_DIR = ROOT / "artifacts" / "fragments" / "issue09b"
CORR = ROOT / "artifacts" / "authoritative"
RERUN = ROOT / "artifacts" / "rerun"  # outputs; the authoritative record is never rewritten
OUT_REPORTS = ROOT / "reports" / "rerun" / "corrective01_reports"
CORR.mkdir(parents=True, exist_ok=True)
OUT_REPORTS.mkdir(parents=True, exist_ok=True)

N4B_CORRECTED = CORR / "n4b_nested_internal_validation_corrected.json"
N6_CORRECTED = CORR / "n6_external_transfer_corrected.json"
BPRIME_CORRECTED = RERUN / "issue09b_missing_singleton_groupkfold_corrected.json"

for p in [N4B_CORRECTED, N6_CORRECTED]:
    assert p.exists(), f"missing {p}"


def merge_fragments_corrected() -> dict[str, Any]:
    mod = importlib.import_module("issue09b_missing_singleton_groupkfold")
    fragments: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        frag_path = FRAGMENTS_DIR / f"{model_name}.json"
        fragments[model_name] = json.loads(frag_path.read_text(encoding="utf-8"))

    first = fragments[MODEL_NAMES[0]]
    for model_name, frag in fragments.items():
        if frag["obelix_dev_population_n"] != first["obelix_dev_population_n"]:
            raise RuntimeError(f"Fragment mismatch: {model_name}")

    results = {name: frag["missing_singleton_group_kfold"] for name, frag in fragments.items()}
    total_runtime = sum(frag["runtime_seconds"] for frag in fragments.values())

    n4b = json.loads(N4B_CORRECTED.read_text(encoding="utf-8"))
    lomfo_major_families_excl_missing = {
        k: v for k, v in n4b["major_families"].items() if k != "__MISSING__"
    }

    manifest = {
        "phase": "corrective01_acs_issue09b_missing_singleton_groupkfold",
        "spec_path": (
            "reports/remediation/specs/corrective01_pretrained_element_embedding_bugfix.md"
        ),
        "historical_spec_path": "reports/remediation/specs/issue09_missing_family_sensitivity.md",
        "git_commit": mod._git("rev-parse", "HEAD"),
        "git_branch": mod._git("rev-parse", "--abbrev-ref", "HEAD"),
        "obelix_dev_population_n": first["obelix_dev_population_n"],
        "obelix_test_untouched_row_count": first["obelix_test_untouched_row_count"],
        "n_missing_rows_now_singleton": 18,
        "hargreaves_data_read": False,
        "n6_external_rerun": False,
        "cv_seed": mod.CV_SEED,
        "n_outer_splits": mod.N_OUTER_SPLITS,
        "inner_tuning_stop_fraction": mod.INNER_TUNING_STOP_FRACTION,
        "results": results,
        "lomfo_major_families_excluding_missing": lomfo_major_families_excl_missing,
        "lomfo_common_support_n_excluding_missing": sum(lomfo_major_families_excl_missing.values()),
        "total_runtime_seconds": total_runtime,
        "corrective_note": (
            "Only pretrained_element_embedding's fragment was regenerated "
            "(reports/remediation/specs/corrective01_pretrained_element_embedding_bugfix.md); "
            "the other 7 models' fragments are byte-identical to the historical "
            "artifacts/fragments/issue09b/*.json."
        ),
    }
    guard_output(BPRIME_CORRECTED).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {BPRIME_CORRECTED.relative_to(ROOT)}")
    return manifest


def run_regime_utility() -> None:
    mod = importlib.import_module("issue09b_regime_utility")
    mod.BPRIME_PATH = BPRIME_CORRECTED
    mod.N4B_PATH = N4B_CORRECTED
    mod.N6_PATH = N6_CORRECTED
    mod.OUTPUT_PATH = RERUN / "issue09b_regime_utility_corrected.json"
    mod.REPORT_PATH = OUT_REPORTS / "issue09b_missing_singleton_groupkfold_corrected.md"
    mod.main()
    print(f"[corrective01] wrote {mod.OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    merge_fragments_corrected()
    run_regime_utility()
