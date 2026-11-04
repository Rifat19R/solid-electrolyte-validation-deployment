#!/usr/bin/env python3
"""Print SHA-256 for one or more files."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if len(sys.argv) < 2:
    raise SystemExit("Usage: python scripts/hash_file.py FILE [FILE ...]")

for arg in sys.argv[1:]:
    p = Path(arg)
    print(f"{sha256(p)}  {p}")
