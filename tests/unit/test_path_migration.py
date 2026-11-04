"""Regression tests for the journal-neutral path migration.

Frozen records keep their original path strings; these tests prove every
hashed path they cite still resolves (through provenance/path_migration_map.json)
to a file with the recorded SHA-256, and that no current path reintroduces a
journal-specific token.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
from pathlib import Path

from sse_transfer.utils.path_migration import MIGRATION_MAP_PATH, resolve_repo_path

ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "artifacts" / "frozen" / "external_transfer_lock.json"
LOCK_SHA256 = "76510b372bca29624e597aa8f9cffdd7ee32bb33962dff40bb85b41cfa519b98"
FORBIDDEN = re.compile(r"npj|acs|jcim", re.IGNORECASE)
# No exemptions: the untrusted quarantine payload was removed in the minimization pass.
NOT_CURRENT_PREFIXES: tuple[str, ...] = ()
# Lock-hashed files that are deliberately never redistributed (derived row by
# row from the Hargreaves database; see RIGHTS_AND_LICENSING.md).
RESTRICTED_LOCK_PATHS = {
    f"data/processed/external_cohort/{name}"
    for name in ("hargreaves_overlap_ledger.csv", "hargreaves_strict.csv", "hargreaves_relaxed.csv")
}


@functools.cache
def _build_package():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "build_package", ROOT / "scripts" / "release" / "build_package.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _in_package_scope(rel: str) -> bool:
    return bool(_build_package().in_package_scope(rel))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.cache
def _redacted() -> dict[str, dict[str, str]]:
    record = ROOT / "provenance" / "restricted_data.json"
    return dict(json.loads(record.read_text(encoding="utf-8"))["redacted"])


def _original_sha256(rel: str) -> str:
    """SHA-256 of the original bytes: a recorded redacted copy (public package)
    stands for its recorded original (scripts/release/restricted_data.py)."""
    digest = _sha256(ROOT / rel)
    entry = _redacted().get(rel)
    if entry and digest == entry["redacted_sha256"]:
        return entry["original_sha256"]
    return digest


def _lock_hashed_paths(node: object) -> list[tuple[str, str]]:
    """Every (path, sha256) pair the lock records, at any depth."""
    pairs: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and f"{key}_sha256" in node:
                pairs.append((value, node[f"{key}_sha256"]))
            elif key == "file" and isinstance(value, str) and "sha256" in node:
                pairs.append((value, node["sha256"]))
            elif key == "code" and isinstance(value, str) and "code_sha256" in node:
                pairs.append((value, node["code_sha256"]))
            pairs.extend(_lock_hashed_paths(value))
    return pairs


def test_lock_is_byte_identical() -> None:
    assert _sha256(LOCK_PATH) == LOCK_SHA256


def test_every_lock_hashed_path_resolves_to_matching_bytes() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    pairs = sorted(set(_lock_hashed_paths(lock)))
    assert len(pairs) >= 10
    for old_path, expected in pairs:
        current = ROOT / resolve_repo_path(old_path)
        assert current.is_file(), f"{old_path} -> {current} missing"
        assert _sha256(current) == expected, f"{old_path} -> {current} hash mismatch"


def test_lock_hashed_paths_in_package_scope_match_and_only_restricted_are_out() -> None:
    """Portable counterpart of the test above: every lock-hashed file the
    reproducibility package ships must match its recorded hash, and the only
    lock-hashed files outside the package are the declared restricted ones."""
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    pairs = sorted(set(_lock_hashed_paths(lock)))
    assert len(pairs) >= 10
    out_of_scope = set()
    for old_path, expected in pairs:
        rel = resolve_repo_path(old_path)
        if not _in_package_scope(rel):
            out_of_scope.add(rel)
            continue
        current = ROOT / rel
        assert current.is_file(), f"{old_path} -> {current} missing"
        assert _sha256(current) == expected, f"{old_path} -> {current} hash mismatch"
    assert out_of_scope == RESTRICTED_LOCK_PATHS


def _promotions() -> dict[str, dict[str, str]]:
    doc = json.loads((ROOT / "provenance" / "authoritative_promotions.json").read_text("utf-8"))
    return {record["file"]: record for record in doc["promotions"]}


def test_migration_map_entries_in_package_scope_resolve() -> None:
    """Package-scope counterpart of the full-repository check in
    tests/repository/test_repository_records.py, restricted to entries whose
    current path the reproducibility package ships; same assertions."""
    entries = json.loads(MIGRATION_MAP_PATH.read_text(encoding="utf-8"))["entries"]
    promoted = _promotions()
    assert len({e["old"] for e in entries}) == len(entries)
    checked = 0
    for entry in entries:
        new = ROOT / entry["new"]
        assert not FORBIDDEN.search(entry["new"]), entry["new"]
        if entry.get("removed"):
            assert not new.exists() and "git history" in entry["reason"], entry["new"]
            continue
        if not _in_package_scope(entry["new"]):
            continue
        checked += 1
        assert new.is_file(), entry["new"]
        if entry["new"] in promoted:
            record = promoted[entry["new"]]
            assert record["previous_sha256"] == entry["sha256"], entry["new"]
            # The archived pre-promotion bytes live under artifacts/historical/,
            # which is outside the package; the full-repository test checks them.
            if _in_package_scope(record["previous_archived_as"]):
                assert _sha256(ROOT / record["previous_archived_as"]) == entry["sha256"]
            assert _original_sha256(entry["new"]) == record["new_sha256"], entry["new"]
        elif entry["new"].startswith(("artifacts/", "data/")):
            assert _original_sha256(entry["new"]) == entry["sha256"], entry["new"]
    assert checked > 0


def test_remediation_spec_hashes_still_match() -> None:
    specs = ROOT / "reports" / "remediation" / "specs"
    for line in (specs / "HASHES.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        target = ROOT / resolve_repo_path(name) if "/" in name else specs / name
        assert _sha256(target) == expected, name


def test_no_current_tracked_path_has_journal_token(current_files: list[str]) -> None:
    offenders = [
        p for p in current_files if FORBIDDEN.search(p) and not p.startswith(NOT_CURRENT_PREFIXES)
    ]
    assert offenders == []


def test_no_current_working_tree_path_has_journal_token() -> None:
    current_roots = ["artifacts", "data/manifests", "data/processed", "manuscript", "provenance"]
    current_roots += ["reports", "scripts", "src", "submission", "tests", "environment"]
    offenders = []
    for name in current_roots:
        base = ROOT / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            rel = path.relative_to(ROOT).as_posix()
            if "__pycache__" not in rel and FORBIDDEN.search(rel):
                offenders.append(rel)
    assert offenders == []
