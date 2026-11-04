#!/usr/bin/env python3
"""Run the test suite inside an extracted reproducibility package.

The whole shipped `tests/` tree is collected and run. The package does not
contain the closed earlier project's tests, the development editor-hook tests
or the repository-only record checks (`tests/closed_project/`,
`tests/dev_tooling/`, `tests/repository/`; see TEST_COVERAGE_EXCEPTIONS.md),
so nothing is ignored and collection has no errors.

The only tests not run are the restricted-data tests listed below. They need
the raw Hargreaves database or files derived from it, which the package
withholds or redacts (provenance/restricted_data.json). They are deselected
only while that material is withheld; after a lawful restore
(`python scripts/release/restricted_data.py restore`) every test runs and
nothing is deselected. The list is exhaustive: any other failure or any skip
fails the run, and a listed target that no longer exists is an error.

Run with: python scripts/release/run_package_tests.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import restricted_data  # noqa: E402

_RAW = "needs the raw Hargreaves database (data/RECONSTRUCTION.md)"
_COHORT = "needs the withheld Hargreaves cohort files (rebuilt by restore)"
_HC = "tests/unit/test_hargreaves_cohort.py::test_build_cohorts_"
_I10 = "tests/unit/test_issue10_formula_balanced_sensitivity.py::test_"
_MS = "tests/unit/test_manifest_schema.py::test_"

# nodeid -> reason. Category A only: restricted Hargreaves-derived material.
# Synthetic or package-scope counterparts that do run are named in
# TEST_COVERAGE_EXCEPTIONS.md.
RESTRICTED: dict[str, str] = {
    _HC + "gate_n2_passes_and_relaxed_is_superset_size": _RAW,
    _HC + "strict_cohort_has_no_overlap_with_obelix": _RAW,
    _MS + "hargreaves_raw_file_matches_manifest": _RAW,
    _MS + "external_cohort_files_match_manifest": _COHORT,
    _I10 + "formula_duplication_stats_matches_frozen_strict_cohort": _COHORT,
    _I10 + "loader_functions_do_not_mutate_frozen_strict_cohort_file": _COHORT,
    "tests/unit/test_path_migration.py::test_every_lock_hashed_path_resolves_to_matching_bytes": (
        _COHORT + "; the lock also hashes the overlap ledger"
    ),
}


def stale_targets() -> list[str]:
    stale = []
    for nodeid in RESTRICTED:
        path, name = nodeid.split("::", 1)
        name = name.split("[", 1)[0]
        file = ROOT / path
        if not file.is_file() or not re.search(
            rf"^def {re.escape(name)}\(", file.read_text("utf-8"), re.M
        ):
            stale.append(nodeid)
    return stale


def main() -> int:
    stale = stale_targets()
    if stale:
        for target in stale:
            print(f"ERROR stale exclusion (target not found): {target}")
        return 2
    state = restricted_data.state(ROOT)
    deselected = RESTRICTED if state == "withheld" else {}
    print(f"restricted material: {state}; deselected restricted-data tests: {len(deselected)}")
    args = [sys.executable, "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider", "tests"]
    for nodeid, reason in deselected.items():
        print(f"not run [A] {nodeid}: {reason}")
        args += ["--deselect", nodeid]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        code = subprocess.run(
            [*args, f"--junitxml={report}"], cwd=ROOT, env=env, check=False
        ).returncode
        suite = ET.parse(report).getroot()
        cases = list(suite.iter("testcase"))
        skipped = sorted(
            f"{case.get('classname')}::{case.get('name')}"
            for case in cases
            if case.find("skipped") is not None
        )
        bad = sum(1 for c in cases if c.find("failure") is not None or c.find("error") is not None)
    for nodeid in skipped:
        print(f"ERROR unexpected skip: {nodeid}")
    if skipped:
        code = code or 3
    # Last line (read by verify_package.py): counts and the policy verdict.
    passed = len(cases) - len(skipped) - bad
    print(
        f"package test policy: {'PASS' if code == 0 else 'FAIL'}; {passed} passed, "
        f"{len(skipped)} skipped, {bad} failed/errors, {len(deselected)} deselected "
        f"(restricted data {state})"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
