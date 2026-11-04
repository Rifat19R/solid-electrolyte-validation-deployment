"""The cohort reconstruction entrypoint verifies against the lock and never overwrites."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts" / "analysis" / "build_external_cohort.py"
    spec = importlib.util.spec_from_file_location("build_external_cohort", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lock_hashes_are_the_frozen_cohort_hashes() -> None:
    hashes = _module().locked_hashes()
    assert hashes["hargreaves_strict.csv"].startswith("1bda972d")
    assert hashes["hargreaves_relaxed.csv"].startswith("dac166ee")
    assert hashes["hargreaves_overlap_ledger.csv"].startswith("a1b4d0f4")


def test_missing_file_is_written_only_when_hash_matches(tmp_path: Path) -> None:
    mod = _module()
    payload = b"reduced_formula\nLi\n"
    good = hashlib.sha256(payload).hexdigest()
    with pytest.raises(RuntimeError, match="lock expects"):
        mod.materialize("x.csv", payload, "0" * 64, tmp_path)
    assert not (tmp_path / "x.csv").exists()
    assert mod.materialize("x.csv", payload, good, tmp_path) == "written"
    assert (tmp_path / "x.csv").read_bytes() == payload


def test_existing_file_is_verified_never_overwritten(tmp_path: Path) -> None:
    mod = _module()
    payload = b"a\n1\n"
    good = hashlib.sha256(payload).hexdigest()
    (tmp_path / "x.csv").write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="not touching it"):
        mod.materialize("x.csv", payload, good, tmp_path)
    assert (tmp_path / "x.csv").read_bytes() == b"tampered\n"
