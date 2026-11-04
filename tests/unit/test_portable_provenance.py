"""Provenance records must use repo-relative POSIX paths, never machine paths."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MACHINE_PATH = re.compile(r"(/mnt/[a-z]/|/home/[^/\s\"']+/|[A-Za-z]:\\\\(?:Users|Rifat))")
# Frozen historical records keep the paths they were written with.
HISTORICAL_PREFIXES = (
    "reports/cross_dataset_study/",
    "reports/history/",
    "reports/validation/",
    "reports/phase",
    "scripts/historical/",
    "_quarantine_unauthorized_fork_output/",
)
# Files that define the machine-path pattern itself (same set the package
# builder exempts); every other file is scanned.
PATTERN_DEFINITIONS = {"scripts/release/build_package.py", "tests/unit/test_portable_provenance.py"}


def _load_freeze_split():
    spec = importlib.util.spec_from_file_location(
        "freeze_split", ROOT / "scripts" / "freeze_split.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sidecar_records_repo_relative_posix_path(tmp_path: Path) -> None:
    freeze_split = _load_freeze_split()
    inside = ROOT / "data" / "processed" / "v1" / "obelix_index.json"
    assert freeze_split.portable_path(inside) == "data/processed/v1/obelix_index.json"
    outside = tmp_path / "split.json"
    outside.write_text("{}", encoding="utf-8")
    sidecar = json.loads(freeze_split.write_sidecar(outside).read_text(encoding="utf-8"))
    assert sidecar["file"] == "split.json"
    assert not MACHINE_PATH.search(sidecar["file"])


def test_existing_sidecars_are_portable_and_match() -> None:
    import hashlib

    for sidecar in ROOT.glob("data/processed/**/*.sha256.json"):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        assert not Path(record["file"]).is_absolute(), sidecar
        target = ROOT / record["file"]
        assert hashlib.sha256(target.read_bytes()).hexdigest() == record["sha256"], sidecar


def test_no_machine_paths_in_current_tracked_text(current_files: list[str]) -> None:
    offenders = []
    for rel in current_files:
        if rel in PATTERN_DEFINITIONS:
            continue
        if rel.startswith(HISTORICAL_PREFIXES) or Path(rel).suffix in {
            ".png",
            ".pdf",
            ".docx",
            ".bundle",
        }:
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if MACHINE_PATH.search(text):
            offenders.append(rel)
    assert offenders == []
