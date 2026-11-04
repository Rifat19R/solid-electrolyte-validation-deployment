"""Plain-text view of the manuscript markdown sources.

The sources carry inline run formatting as `<b>`, `<i>`, `<sup>` and `<sub>`
tags (rendered by the DOCX builder). Any check that compares manuscript text
with numbers or phrases must read it through `plain_text`.
"""

from __future__ import annotations

import re
from pathlib import Path

INLINE_TAG = re.compile(r"</?(?:b|i|sup|sub)>")


def plain_text(text: str) -> str:
    return INLINE_TAG.sub("", text)


def read_plain(path: Path) -> str:
    return plain_text(path.read_text(encoding="utf-8"))
