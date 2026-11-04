#!/usr/bin/env python3
"""Fast project preflight. Safe to run before any experiment.

`--release` checks an extracted reproducibility package instead of a
development checkout (no git or editor-tooling files are expected there).
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "CLAUDE.md",
    "RESEARCH_CONTRACT.md",
    "DATA_CONTRACT.md",
    "EXPERIMENT_PROTOCOL.md",
    "DEFINITION_OF_DONE.md",
    "configs/data.yaml",
    "configs/model.yaml",
    "configs/experiment.yaml",
    ".claude/settings.json",
]


RELEASE_REQUIRED = [
    "README.md",
    "LICENSE",
    "MANIFEST_SHA256.txt",
    "RESEARCH_CONTRACT.md",
    "DATA_CONTRACT.md",
    "EXPERIMENT_PROTOCOL.md",
    "configs/data.yaml",
    "data/RECONSTRUCTION.md",
    "artifacts/frozen/external_transfer_lock.json",
    "artifacts/authoritative/n6_external_transfer_corrected.json",
    "provenance/path_migration_map.json",
]


def main() -> int:
    release = "--release" in sys.argv
    failures: list[str] = []
    if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
        failures.append(f"Python must be 3.12; found {platform.python_version()}")

    for rel in RELEASE_REQUIRED if release else REQUIRED:
        if not (ROOT / rel).exists():
            failures.append(f"Missing required file: {rel}")

    if not release:
        try:
            json.loads((ROOT / ".claude/settings.json").read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"Invalid .claude/settings.json: {exc}")

        # .venv itself is expected after setup, but it must be gitignored.
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        if ".venv/" not in gitignore:
            failures.append(".venv/ is not ignored by git")

    if failures:
        print("PREFLIGHT: FAIL")
        for item in failures:
            print(f"- {item}")
        return 1

    print("PREFLIGHT: PASS")
    print(f"Python: {platform.python_version()}")
    print("Planning contract: v2.0 (supersedes v1.0, see CHANGELOG.md)")
    print("New DFT/NEB/AIMD/MD: PROHIBITED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
