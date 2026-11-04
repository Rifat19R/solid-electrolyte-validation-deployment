"""The manuscript table builder rebuilds Tables 1-4 and fails closed on any mismatch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts" / "manuscript" / "build_tables.py"
    spec = importlib.util.spec_from_file_location("manuscript_build_tables", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_manuscript_tables_match_authoritative_artifacts() -> None:
    mod = _module()
    built = {name: builder() for name, builder in mod.BUILDERS.items()}
    assert mod.verify(built) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "| B: Family-GroupKFold | Eight models (pre-specified) | 0.6810 |",
            "| B: Family-GroupKFold | Eight models (pre-specified) | 0.5930 |",
        ),
        ("[−0.0562, −0.0019]", "[−0.0562, −0.0028]"),
        (
            "| Strict external cohort (primary) | 204 |",
            "| Strict external cohort (primary) | 205 |",
        ),
    ],
)
def test_stale_or_edited_number_fails(tmp_path: Path, old: str, new: str) -> None:
    mod = _module()
    text = mod.MAIN_MD.read_text(encoding="utf-8")
    assert old in text
    tampered = tmp_path / "00_main.md"
    tampered.write_text(text.replace(old, new), encoding="utf-8")
    mod.MAIN_MD = tampered
    built = {name: builder() for name, builder in mod.BUILDERS.items()}
    assert mod.verify(built), "a changed table value must be reported"
