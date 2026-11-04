#!/usr/bin/env python3
"""Rebuild the frozen external cohorts from the raw sources and verify them.

Runs `sse_transfer.data.hargreaves_cohort.build_cohorts` (the N2 pipeline:
15-35 degC window, valid target, strict/relaxed overlap exclusion against
OBELiX, median log10 conductivity per reduced formula) and serializes the
strict cohort, relaxed cohort and row-level overlap ledger exactly as they
were frozen. Every output is checked against the SHA-256 recorded in the
external-transfer lock before it is accepted.

Existing files are never overwritten: present files are only verified. A
missing file (for example after a lawful re-download of the raw Hargreaves
database into a fresh copy of the release) is written to a temporary file,
verified against the lock hash, and only then moved into place.

The raw inputs are `data/raw/hargreaves_conductivity/LiIonDatabase.csv` and
`data/raw/obelix/all.csv`; see `data/RECONSTRUCTION.md` for retrieval.

Run with: python scripts/analysis/build_external_cohort.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.data.hargreaves_cohort import build_cohorts  # noqa: E402

LOCK_PATH = ROOT / "artifacts" / "frozen" / "external_transfer_lock.json"
COHORT_DIR = ROOT / "data" / "processed" / "external_cohort"
EXPECTED_COUNTS = {
    "hargreaves_strict.csv": 204,
    "hargreaves_relaxed.csv": 239,
    "hargreaves_overlap_ledger.csv": 820,
}


def locked_hashes(lock_path: Path = LOCK_PATH) -> dict[str, str]:
    cohort = json.loads(lock_path.read_text(encoding="utf-8"))["frozen_inputs"]["external_cohort"]
    return {
        "hargreaves_strict.csv": cohort["strict_cohort_file_sha256"],
        "hargreaves_relaxed.csv": cohort["relaxed_cohort_file_sha256"],
        "hargreaves_overlap_ledger.csv": cohort["overlap_ledger_file_sha256"],
    }


def materialize(name: str, payload: bytes, expected_sha256: str, cohort_dir: Path) -> str:
    """Verify an existing file, or write a missing one only if it matches the lock."""
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            f"{name}: rebuilt bytes hash to {digest}, lock expects {expected_sha256}. "
            "The raw inputs or the pipeline differ from the frozen state."
        )
    target = cohort_dir / name
    if target.exists():
        existing = hashlib.sha256(target.read_bytes()).hexdigest()
        if existing != expected_sha256:
            raise RuntimeError(f"{target} exists but does not match the lock hash; not touching it")
        return "verified"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(target)
    return "written"


def main() -> None:
    result = build_cohorts()
    frames = {
        "hargreaves_strict.csv": result.strict_cohort,
        "hargreaves_relaxed.csv": result.relaxed_cohort,
        "hargreaves_overlap_ledger.csv": result.ledger,
    }
    expected = locked_hashes()
    for name, frame in frames.items():
        if len(frame) != EXPECTED_COUNTS[name]:
            raise RuntimeError(f"{name}: {len(frame)} rows, expected {EXPECTED_COUNTS[name]}")
        status = materialize(name, frame.to_csv(index=False).encode(), expected[name], COHORT_DIR)
        print(f"{name}: {len(frame)} rows, sha256 matches lock ({status})")
    print(f"stage counts: {result.stage_counts}")


if __name__ == "__main__":
    main()
