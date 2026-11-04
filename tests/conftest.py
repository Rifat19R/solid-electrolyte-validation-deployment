"""Shared pytest fixtures."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def current_files() -> list[str]:
    """Repo-relative POSIX paths of every current file.

    In a git checkout: `git ls-files` (tracked files). In an extracted
    reproducibility package (no git metadata): every file listed in the
    package's own MANIFEST_SHA256.txt, which `verify_package.py` checks
    against the files on disk. Fails, never skips, when neither exists.
    """
    if (ROOT / ".git").exists():
        return subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    manifest = ROOT / "MANIFEST_SHA256.txt"
    if not manifest.is_file():
        pytest.fail("neither a git checkout nor a package MANIFEST_SHA256.txt is present")
    paths = [ln.split("  ", 1)[1] for ln in manifest.read_text("utf-8").splitlines() if ln]
    assert paths, "empty package manifest"
    return paths
