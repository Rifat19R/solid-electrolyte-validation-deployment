"""Schema and hash checks for this study's data manifests.

Covers OBELiX (`obelix.json`), the Hargreaves et al. raw database
(`hargreaves_conductivity.json`) and the external-cohort build record
(`external_cohort.json`). The manifests of the closed earlier project's
pretraining datasets are checked in tests/closed_project/.

The Hargreaves raw file and its derived cohort files are not redistributed
(RIGHTS_AND_LICENSING.md); the two tests that hash those files on disk are the
documented restricted-data exceptions of the reproducibility package.
"""

import hashlib
import json
from pathlib import Path

import pytest

from sse_transfer.utils.path_migration import resolve_repo_path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "data" / "manifests"
LOCK_PATH = ROOT / "artifacts" / "frozen" / "external_transfer_lock.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(name: str) -> dict:
    return json.loads((MANIFEST_DIR / name).read_text(encoding="utf-8"))


REQUIRED_FILE_KEYS = {"filename", "path", "sha256"}
RAW_MANIFESTS = ["obelix.json", "hargreaves_conductivity.json"]


@pytest.mark.parametrize("filename", RAW_MANIFESTS)
def test_raw_manifest_has_provenance_keys(filename: str) -> None:
    manifest = _load(filename)
    for key in ("dataset_name", "version", "role", "retrieval_date", "license", "files"):
        assert manifest.get(key), f"{filename} missing {key}"
    assert "paper_doi" in manifest or "paper_reference" in manifest
    assert any(k.startswith("source_url") or k == "source_repository" for k in manifest)


@pytest.mark.parametrize("filename", [*RAW_MANIFESTS, "external_cohort.json"])
def test_manifest_files_entries_have_required_keys(filename: str) -> None:
    manifest = _load(filename)
    assert isinstance(manifest["files"], list) and manifest["files"]
    for entry in manifest["files"]:
        missing = REQUIRED_FILE_KEYS - entry.keys()
        assert not missing, f"{filename} file entry missing keys: {missing}"
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)


def test_obelix_manifest_room_temperature_scope_is_verified() -> None:
    """Phase 0C closure: OBELiX target temperature scope must be verified against
    the primary source DOI (10.1039/D5DD00441A), not assumed. See DATA_CONTRACT.md
    dataset role D and the canonical label y = log10(sigma_RT / S cm^-1)."""
    manifest = _load("obelix.json")
    assert manifest["temperature_scope"] == "ROOM_TEMPERATURE"
    assert manifest["verification_status"] == "VERIFIED_BY_PRIMARY_SOURCE"
    assert manifest["paper_doi"] == "10.1039/D5DD00441A"
    verification = manifest["temperature_scope_verification"]
    assert verification["primary_source_doi"] == "10.1039/D5DD00441A"
    assert "room temperature" in verification["verified_claim"].lower()
    assert "no explicit per-row temperature column" in verification["row_level_caveat"]


def test_obelix_manifest_sha256_matches_file_on_disk() -> None:
    for entry in _load("obelix.json")["files"]:
        path = ROOT / entry["path"]
        assert path.is_file(), f"missing {entry['path']} (run scripts/fetch_raw_data.py)"
        assert path.stat().st_size == entry["size_bytes"], f"size mismatch for {path}"
        assert _sha256(path) == entry["sha256"], f"sha256 mismatch for {path}"


def test_hargreaves_manifest_records_source_counts_and_terms() -> None:
    manifest = _load("hargreaves_conductivity.json")
    assert manifest["paper_doi"] == "10.1038/s41524-022-00951-z"
    assert manifest["original_count"] == manifest["parsed_count"] == 820
    assert manifest["censoring"]["n_censored_rows"] == 0
    # Not an open licence: the terms restrict commercial re-use and require citation.
    assert "commercial" in manifest["license"].lower()
    assert "NOT a standard open license" in manifest["license"]
    [entry] = manifest["files"]
    assert entry["filename"] == "LiIonDatabase.csv"
    assert entry["path"] == "data/raw/hargreaves_conductivity/LiIonDatabase.csv"
    assert entry["size_bytes"] == 76402


def test_external_cohort_manifest_is_consistent_with_raw_manifest_and_lock() -> None:
    cohort = _load("external_cohort.json")
    raw_sha = _load("hargreaves_conductivity.json")["files"][0]["sha256"]
    assert raw_sha in cohort["derived_from"]
    assert cohort["rt_window_c"] == [15.0, 35.0]
    stages = cohort["stage_counts"]
    assert stages["raw_rows"] == 820
    assert stages["strict_independent_rows"] == 204
    assert stages["relaxed_independent_rows"] == 239
    assert stages["strict_independent_rows"] <= stages["strict_pre_aggregation_rows"]
    assert cohort["gate_n2"]["result"] == "PASS"
    assert cohort["gate_n2"]["strict_independent_rows"] >= cohort["gate_n2"]["threshold"]
    rows = {e["filename"]: e["rows"] for e in cohort["files"]}
    assert rows == {
        "hargreaves_strict.csv": 204,
        "hargreaves_relaxed.csv": 239,
        "hargreaves_overlap_ledger.csv": 820,
    }
    frozen = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["frozen_inputs"]["external_cohort"]
    by_name = {e["filename"]: e["sha256"] for e in cohort["files"]}
    assert by_name["hargreaves_strict.csv"] == frozen["strict_cohort_file_sha256"]
    assert by_name["hargreaves_relaxed.csv"] == frozen["relaxed_cohort_file_sha256"]
    assert by_name["hargreaves_overlap_ledger.csv"] == frozen["overlap_ledger_file_sha256"]
    assert frozen["strict_cohort_n"] == 204 and frozen["relaxed_cohort_n"] == 239
    manifest_path = ROOT / resolve_repo_path(frozen["cohort_build_manifest"])
    assert manifest_path == MANIFEST_DIR / "external_cohort.json"
    assert _sha256(manifest_path) == frozen["cohort_build_manifest_sha256"]
    raw_manifest = ROOT / resolve_repo_path(frozen["raw_source_manifest"])
    assert _sha256(raw_manifest) == frozen["raw_source_manifest_sha256"]


def test_hargreaves_raw_file_matches_manifest() -> None:
    """Restricted data: needs the raw Hargreaves database (data/RECONSTRUCTION.md)."""
    [entry] = _load("hargreaves_conductivity.json")["files"]
    path = ROOT / entry["path"]
    assert path.is_file(), f"missing {entry['path']} (see data/RECONSTRUCTION.md)"
    assert path.stat().st_size == entry["size_bytes"]
    assert _sha256(path) == entry["sha256"]


def test_external_cohort_files_match_manifest() -> None:
    """Restricted data: needs the cohort files rebuilt from the raw Hargreaves
    database (scripts/analysis/build_external_cohort.py)."""
    for entry in _load("external_cohort.json")["files"]:
        path = ROOT / resolve_repo_path(entry["path"])
        assert path.is_file(), f"missing {path.relative_to(ROOT)} (see data/RECONSTRUCTION.md)"
        assert _sha256(path) == entry["sha256"], entry["filename"]
        with path.open(encoding="utf-8") as fh:
            assert sum(1 for _ in fh) - 1 == entry["rows"]
