#!/usr/bin/env python3
"""Corrective 01: merge the 7 untouched historical N6 fragments with the 1
corrected `pretrained_element_embedding` fragment into a NEW, clearly
non-primary artifact. Does NOT call `n6_external_transfer.merge_fragments`
(which writes to the locked historical path) -- this is a separate,
minimal reimplementation of the same merge logic pointed at a new output
path, so the historical `artifacts/frozen/n6_external_transfer_primary.json`
(run_count=1, authorized=true) is never touched.

Run with: python scripts/analysis/corrective01_merge_n6.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.models.ladder import MODEL_NAMES  # noqa: E402
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

FRAGMENTS_DIR = ROOT / "artifacts" / "fragments" / "n6"
OUT_DIR = ROOT / "artifacts" / "rerun"
OUT_PATH = OUT_DIR / "n6_external_transfer_corrected.json"
HISTORICAL_LOCK = ROOT / "artifacts" / "frozen" / "external_transfer_lock.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], text=True, capture_output=True, check=False, cwd=ROOT
    ).stdout.strip()


def main() -> None:
    fragments: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        frag_path = FRAGMENTS_DIR / f"{model_name}.json"
        fragments[model_name] = json.loads(frag_path.read_text(encoding="utf-8"))

    first = fragments[MODEL_NAMES[0]]
    for model_name, frag in fragments.items():
        if frag["obelix_dev_population_n"] != first["obelix_dev_population_n"]:
            raise RuntimeError(
                f"Fragment mismatch: {model_name} used a different dev-population size."
            )

    results = {
        name: {"strict": frag["strict"], "relaxed": frag["relaxed"]}
        for name, frag in fragments.items()
    }
    ranking = sorted(MODEL_NAMES, key=lambda m: results[m]["strict"]["mae"])
    for rank, name in enumerate(ranking, start=1):
        results[name]["strict"]["rank_by_strict_mae"] = rank

    manifest = {
        "phase": "corrective01_N6_external_transfer_CORRECTED_NON_PRIMARY",
        "authorized": "N/A_CORRECTIVE_NON_PRIMARY",
        "note": (
            "This is NOT a second primary one-shot N6 result. It combines the "
            "7 untouched historical per-model fragments (ridge, random_forest, "
            "extra_trees, xgboost, catboost, shallow_mlp, composition_attention "
            "-- byte-identical to artifacts/frozen/n6_external_transfer_primary.json's own "
            "fragments) with ONE corrected fragment for "
            "pretrained_element_embedding (re-fit after fixing the pymatgen "
            "Element bug in features/element_embeddings.py; see "
            "reports/remediation/specs/corrective01_pretrained_element_embedding_bugfix.md). "
            "The historical artifacts/frozen/n6_external_transfer_primary.json "
            "(run_count=1, authorized=true) is untouched by this script."
        ),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "lock_file": "artifacts/frozen/external_transfer_lock.json",
        "lock_file_sha256": json.loads(
            (ROOT / "artifacts" / "frozen" / "n6_external_transfer_primary.json").read_text(
                encoding="utf-8"
            )
        )["lock_file_sha256"],
        "obelix_dev_population_n": first["obelix_dev_population_n"],
        "obelix_test_untouched_row_count": first["obelix_test_untouched_row_count"],
        "seeds": [17, 29, 43, 71, 101],
        "model_ranking_by_strict_mae": ranking,
        "results": results,
        "corrected_model": "pretrained_element_embedding",
        "unchanged_models": [m for m in MODEL_NAMES if m != "pretrained_element_embedding"],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    guard_output(OUT_PATH).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    print("\n=== Corrected strict-cohort ranking ===")
    for rank, name in enumerate(ranking, start=1):
        r = results[name]["strict"]
        print(f"{rank}. {name}: MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} rho={r['spearman_rho']}")


if __name__ == "__main__":
    main()
