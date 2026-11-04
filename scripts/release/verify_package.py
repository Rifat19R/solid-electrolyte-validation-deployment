#!/usr/bin/env python3
"""Validate the reproducibility ZIP from a fresh extraction.

Checks, in a new temporary directory: the ZIP hash against its .sha256 file;
safe member names; every file against MANIFEST_SHA256.txt; release preflight;
protected anchors; table provenance; numeric audit; test collection (no
errors); the package test suite; the presence of the four submission
documents; the exclusion rules (no restricted, historical, quarantine,
journal-named or machine-path content); and that every Hargreaves-derived
file is withheld or carries its recorded redacted hash.

Run with: python scripts/release/verify_package.py [ZIP]
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZIP_DEFAULT = ROOT / "release" / "SSE_Reproducibility_Submission.zip"
FORBIDDEN_PATH = re.compile(
    r"npj|acs|jcim|turnitin|quarantine|historical|pre_fix|LiIonDatabase|overlap_ledger|\.venv|\.git/",
    re.I,
)
SUBMISSION = [
    "Main_Manuscript.docx",
    "Supporting_Information.docx",
    "Cover_Letter.docx",
    "Data_and_Code_Availability.docx",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    zip_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ZIP_DEFAULT
    results: list[tuple[str, bool, str]] = []
    expected = (zip_path.parent / f"{zip_path.name}.sha256").read_text().split()[0]
    results.append(("zip sha256", sha256(zip_path) == expected, expected))
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            unsafe = [n for n in names if n.startswith("/") or ".." in Path(n).parts]
            results.append(("safe member names", not unsafe, f"{len(names)} members"))
            zf.extractall(tmp)
        pkg = Path(tmp) / "SSE_Reproducibility"
        manifest = (pkg / "MANIFEST_SHA256.txt").read_text().splitlines()
        bad = [ln for ln in manifest if sha256(pkg / ln.split("  ", 1)[1]) != ln.split()[0]]
        listed = {ln.split("  ", 1)[1] for ln in manifest}
        on_disk = {p.relative_to(pkg).as_posix() for p in pkg.rglob("*") if p.is_file()}
        unlisted = on_disk - listed - {"MANIFEST_SHA256.txt"}
        results.append(("manifest hashes", not bad and not unlisted, f"{len(manifest)} files"))
        forbidden = sorted(p for p in on_disk if FORBIDDEN_PATH.search(p))
        results.append(("exclusions", not forbidden, ", ".join(forbidden[:5]) or "none"))
        sys.path.insert(0, str(pkg / "scripts" / "release"))
        import restricted_data  # noqa: PLC0415

        try:
            state = restricted_data.state(pkg)
        except RuntimeError as exc:
            state = str(exc)
        shipped_withheld = [w for w in restricted_data.WITHHELD if (pkg / w).exists()]
        results.append(
            (
                "restricted data withheld/redacted",
                state == "withheld" and not shipped_withheld,
                f"{state}; {len(restricted_data.REDACTED)} redacted records",
            )
        )
        missing_docs = [d for d in SUBMISSION if not (pkg / "submission" / d).is_file()]
        results.append(
            ("submission documents", not missing_docs, ", ".join(missing_docs) or "all 4")
        )
        env = dict(os.environ, PYTHONPATH=str(pkg / "src"))
        for label, cmd in [
            ("preflight", ["scripts/preflight.py", "--release"]),
            ("protected anchors", ["scripts/release/verify_anchors.py"]),
            ("table provenance", ["scripts/manuscript/build_tables.py", "--check-only"]),
            ("numeric audit", ["scripts/analysis/numeric_audit.py"]),
            ("test collection", ["-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]),
            ("package tests", ["scripts/release/run_package_tests.py"]),
        ]:
            proc = subprocess.run(
                [sys.executable, *cmd],
                cwd=pkg,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            tail = (proc.stdout.strip().splitlines() or [proc.stderr.strip()[-200:]])[-1]
            results.append((label, proc.returncode == 0, tail[:120]))
    for label, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}")
    failed = [r for r in results if not r[1]]
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
