"""Resolve pre-migration repository paths cited by frozen provenance records.

Frozen records (the external-transfer lock, hash-locked specs, historical
reports) are never edited, so they keep citing the paths that existed when
they were written. ``provenance/path_migration_map.json`` maps each such old
path to its current location; this module is the only lookup used for that.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION_MAP_PATH = REPO_ROOT / "provenance" / "path_migration_map.json"


@lru_cache(maxsize=1)
def _old_to_new() -> dict[str, str]:
    data = json.loads(MIGRATION_MAP_PATH.read_text(encoding="utf-8"))
    mapping = {entry["old"]: entry["new"] for entry in data["entries"]}
    if len(set(mapping.values())) != len(mapping):
        raise RuntimeError("path_migration_map.json maps two old paths to one new path")
    return mapping


def resolve_repo_path(path: str) -> str:
    """Return the current repo-relative POSIX path for ``path``.

    Paths that were not migrated are returned unchanged. A path that was
    migrated returns its new location; old directory prefixes are not
    guessed, only exact file entries are resolved.
    """
    return _old_to_new().get(path, path)
