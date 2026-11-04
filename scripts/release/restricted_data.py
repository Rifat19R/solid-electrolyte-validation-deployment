#!/usr/bin/env python3
"""Hargreaves-derived files that the public reproducibility package withholds.

The Hargreaves et al. database is used under research-use terms that do not
grant redistribution (RIGHTS_AND_LICENSING.md). The package therefore ships
no file from which the external targets can be read back:

* WITHHELD files are not shipped at all: the raw database, the row-level
  overlap ledger and the two formula-level cohort CSVs (they carry the
  median log10 conductivity of every external formula).
* REDACTED files are shipped with the per-row external target (`y_true`)
  and the per-row residual (`y_pred - y_true`, from which the target
  follows) set to null. These are the one-shot external-transfer records
  and their per-model fragments; every aggregate (MAE, RMSE, R2, Spearman,
  ranks) and every prediction is kept.

`provenance/restricted_data.json` records the SHA-256 of every original file
and of every redacted copy. A user who obtains the raw database under the
source terms restores the original bytes with

    python scripts/release/restricted_data.py restore

which rebuilds the cohort files (scripts/analysis/build_external_cohort.py,
lock-verified), fills the redacted fields from the rebuilt strict/relaxed
cohorts and writes a file only if the result hashes to the recorded original.

Commands:
    status         print whether the restricted material is withheld or restored
    restore        rebuild and verify the restricted material (needs the raw database)
    write-record   regenerate provenance/restricted_data.json (repository only)
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "provenance" / "restricted_data.json"
COHORT_DIR = "data/processed/external_cohort"
RAW_HARGREAVES = "data/raw/hargreaves_conductivity/LiIonDatabase.csv"

WITHHELD = (
    RAW_HARGREAVES,
    f"{COHORT_DIR}/hargreaves_overlap_ledger.csv",
    f"{COHORT_DIR}/hargreaves_strict.csv",
    f"{COHORT_DIR}/hargreaves_relaxed.csv",
)
_MODELS = (
    "catboost",
    "composition_attention",
    "extra_trees",
    "pretrained_element_embedding",
    "random_forest",
    "ridge",
    "shallow_mlp",
    "xgboost",
)
REDACTED = (
    "artifacts/frozen/n6_external_transfer_primary.json",
    "artifacts/authoritative/n6_external_transfer_corrected.json",
    *(f"artifacts/fragments/n6/{m}.json" for m in _MODELS),
)
REDACTED_FIELDS = ("y_true", "residual")
_ROW_BLOCK = {"reduced_formula", "y_pred", *REDACTED_FIELDS}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _row_blocks(node: Any) -> list[dict[str, Any]]:
    """Every per-row prediction block (reduced_formula/y_pred/y_true/residual)."""
    if isinstance(node, dict):
        if _ROW_BLOCK <= node.keys():
            return [node]
        return [b for v in node.values() for b in _row_blocks(v)]
    if isinstance(node, list):
        return [b for v in node for b in _row_blocks(v)]
    return []


def _dump(doc: Any) -> bytes:
    # Same serialization as scripts/transport_analysis/n6_external_transfer.py.
    return json.dumps(doc, indent=2, default=str).encode("utf-8")


def redact(payload: bytes) -> bytes:
    doc = json.loads(payload)
    if _dump(doc) != payload:
        raise ValueError("record does not round-trip through its serializer")
    blocks = _row_blocks(doc)
    if not blocks:
        raise ValueError("no per-row prediction block found")
    for block in blocks:
        for field in REDACTED_FIELDS:
            block[field] = None
    return _dump(doc)


def restore(redacted: bytes, cohort_targets: dict[str, float]) -> bytes:
    """Refill the redacted fields; the residual is recomputed as n6 computed it."""
    doc = copy.deepcopy(json.loads(redacted))
    for block in _row_blocks(doc):
        block["y_true"] = [cohort_targets[f] for f in block["reduced_formula"]]
        block["residual"] = [p - t for p, t in zip(block["y_pred"], block["y_true"], strict=True)]
    return _dump(doc)


def cohort_targets(root: Path) -> dict[str, float]:
    """reduced_formula -> log_target_rt_median from the rebuilt cohort CSVs.

    Parsed with pandas' default float parser, as n6 read them."""
    import pandas as pd

    targets: dict[str, float] = {}
    for name in ("hargreaves_strict.csv", "hargreaves_relaxed.csv"):
        frame = pd.read_csv(root / COHORT_DIR / name)
        for formula, value in zip(
            frame["reduced_formula"], frame["log_target_rt_median"].tolist(), strict=True
        ):
            if targets.setdefault(formula, value) != value:
                raise ValueError(f"{formula}: strict and relaxed targets differ")
    return targets


def load_record(root: Path = ROOT) -> dict[str, Any]:
    return json.loads((root / "provenance" / "restricted_data.json").read_text("utf-8"))


def state(root: Path = ROOT) -> str:
    """'restored' if every restricted file holds its original bytes, else 'withheld'.

    Raises if a file is in neither recorded state (unknown bytes)."""
    record = load_record(root)
    restored = True
    for rel, entry in record["withheld"].items():
        path = root / rel
        if not path.is_file():
            restored = False
        elif sha256_bytes(path.read_bytes()) != entry["sha256"]:
            raise RuntimeError(f"{rel}: bytes match no recorded state")
    for rel, entry in record["redacted"].items():
        digest = sha256_bytes((root / rel).read_bytes())
        if digest == entry["redacted_sha256"]:
            restored = False
        elif digest != entry["original_sha256"]:
            raise RuntimeError(f"{rel}: bytes match no recorded state")
    return "restored" if restored else "withheld"


def build_record(root: Path = ROOT) -> dict[str, Any]:
    withheld = {rel: {"sha256": sha256_bytes((root / rel).read_bytes())} for rel in WITHHELD}
    redacted = {}
    for rel in REDACTED:
        original = (root / rel).read_bytes()
        redacted[rel] = {
            "original_sha256": sha256_bytes(original),
            "redacted_sha256": sha256_bytes(redact(original)),
            "redacted_fields": list(REDACTED_FIELDS),
        }
    return {
        "description": (
            "Hargreaves-derived files withheld from, or redacted in, the public "
            "reproducibility package (RIGHTS_AND_LICENSING.md). Restore with "
            "`python scripts/release/restricted_data.py restore` after obtaining "
            "the raw database (data/RECONSTRUCTION.md)."
        ),
        "withheld": withheld,
        "redacted": redacted,
    }


def run_restore(root: Path = ROOT) -> None:
    record = load_record(root)
    raw = root / RAW_HARGREAVES
    if not raw.is_file():
        raise SystemExit(f"missing {RAW_HARGREAVES}; see data/RECONSTRUCTION.md")
    if sha256_bytes(raw.read_bytes()) != record["withheld"][RAW_HARGREAVES]["sha256"]:
        raise SystemExit(f"{RAW_HARGREAVES} is a different version (hash mismatch)")
    subprocess.run(
        [sys.executable, str(root / "scripts" / "analysis" / "build_external_cohort.py")],
        cwd=root,
        check=True,
    )
    for rel, entry in record["withheld"].items():
        if sha256_bytes((root / rel).read_bytes()) != entry["sha256"]:
            raise SystemExit(f"{rel}: rebuilt bytes do not match the record")
    targets = cohort_targets(root)
    for rel, entry in record["redacted"].items():
        path = root / rel
        current = path.read_bytes()
        if sha256_bytes(current) == entry["original_sha256"]:
            print(f"{rel}: original (verified)")
            continue
        if sha256_bytes(current) != entry["redacted_sha256"]:
            raise SystemExit(f"{rel}: bytes match no recorded state; not touching it")
        restored = restore(current, targets)
        if sha256_bytes(restored) != entry["original_sha256"]:
            raise SystemExit(f"{rel}: restored bytes do not match the recorded original")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(restored)
        tmp.replace(path)
        print(f"{rel}: restored (sha256 matches the recorded original)")
    print(f"restricted material: {state(root)}")


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "status"
    if command == "status":
        print(f"restricted material: {state()}")
    elif command == "restore":
        run_restore()
    elif command == "write-record":
        if not (ROOT / "release" / "README.md").is_file():
            raise SystemExit("write-record runs in the repository, not in the package")
        RECORD.write_text(json.dumps(build_record(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {RECORD.relative_to(ROOT)}")
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
