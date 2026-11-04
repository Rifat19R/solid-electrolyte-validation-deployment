#!/usr/bin/env python3
"""Hash an existing split file so subsequent runs can prove split identity.

The sidecar records the file as a repo-relative POSIX path (never an
absolute machine path), so it stays valid in any checkout or extracted
release.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path, root: Path = ROOT) -> str:
    """Repo-relative POSIX path for files inside the repository, else the file name."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.name


def write_sidecar(path: Path) -> Path:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    out = path.with_suffix(path.suffix + ".sha256.json")
    out.write_text(
        json.dumps({"file": portable_path(path), "sha256": digest}, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/freeze_split.py path/to/split.json")
    print(write_sidecar(Path(sys.argv[1])))
