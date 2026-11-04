"""Regression test: the submission package must never contain "§".

Ports the guard from a retired journal-specific test of the earlier
submission track (removed in the repository migration -- see
reports/history/repo_migration_final.md). A prior DOCX build in that track
reintroduced "§2.5"/"§2.4" into rendered text even though the markdown
source had zero "§" characters, via a hardcoded "§" literal inside a
builder script's text-replacement table that a markdown-only sweep never
caught. This test checks all three places a "§"
could reappear for the current build: the markdown sources, the DOCX
builder scripts (source-level, so a bad edit fails before anyone even
rebuilds), and -- when the built DOCX files are present on disk -- the
rendered DOCX text itself.
"""

from __future__ import annotations

from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[2]

MARKDOWN_SOURCES = sorted((ROOT / "manuscript" / "source").glob("*.md"))

BUILDER_SCRIPTS = sorted((ROOT / "scripts" / "manuscript").glob("*.py"))

BUILT_DOCX_FILES = sorted((ROOT / "submission").glob("*.docx"))


def _docx_text(path: Path) -> str:
    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def test_no_section_symbol_in_markdown_sources() -> None:
    for path in MARKDOWN_SOURCES:
        text = path.read_text(encoding="utf-8")
        assert "§" not in text, f"{path} contains '§'"


def test_no_section_symbol_in_builder_source() -> None:
    for path in BUILDER_SCRIPTS:
        text = path.read_text(encoding="utf-8")
        assert "§" not in text, f"{path} contains '§' -- a builder cannot reintroduce it"


def test_no_section_symbol_in_built_docx() -> None:
    for path in BUILT_DOCX_FILES:
        if not path.exists():
            continue
        text = _docx_text(path)
        assert "§" not in text, f"{path} contains '§'"
