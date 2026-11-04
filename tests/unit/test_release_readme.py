"""The package README's contents table must describe exactly what the package ships."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATED = {
    "VERSION.txt",
    "MANIFEST_SHA256.txt",
    "environment/requirements-lock.txt",
    "environment/python-version.txt",
}


def _shipped() -> set[str]:
    """Package-relative paths of every shipped file."""
    # The source repository keeps the package README at release/README.md; an
    # extracted package (with or without its own git history) does not.
    if (ROOT / "release" / "README.md").is_file():
        spec = importlib.util.spec_from_file_location(
            "build_package", ROOT / "scripts" / "release" / "build_package.py"
        )
        assert spec is not None and spec.loader is not None
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        return set(builder.selected_sources().values()) | GENERATED
    manifest = (ROOT / "MANIFEST_SHA256.txt").read_text("utf-8").splitlines()
    return {ln.split("  ", 1)[1] for ln in manifest if ln} | {"MANIFEST_SHA256.txt"}


def _readme_paths() -> list[str]:
    readme = ROOT / "release" / "README.md"
    if not readme.is_file():
        readme = ROOT / "README.md"
    text = readme.read_text("utf-8")
    table = text.split("## Contents", 1)[1].split("\n## ", 1)[0]
    paths = []
    for line in table.splitlines():
        if line.startswith("| `"):
            first_cell = line.split("|")[1]
            paths += re.findall(r"`([^`]+)`", first_cell)
    return [p.rstrip("/") for p in paths]


def _covers(entry: str, rel: str) -> bool:
    return rel == entry or rel.startswith(entry + "/")


def test_every_readme_path_is_shipped() -> None:
    shipped = _shipped()
    missing = [e for e in _readme_paths() if not any(_covers(e, rel) for rel in shipped)]
    assert missing == []


def test_every_shipped_file_is_described_in_the_readme() -> None:
    entries = _readme_paths()
    undescribed = sorted(rel for rel in _shipped() if not any(_covers(e, rel) for e in entries))
    assert undescribed == []
