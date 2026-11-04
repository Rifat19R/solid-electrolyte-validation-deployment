#!/usr/bin/env python3
"""Build the four submission documents from the markdown sources, fail-closed.

Sources (manuscript/source/):
  00_main.md                       -> submission/Main_Manuscript.docx
  01_si.md                         -> submission/Supporting_Information.docx
  02_cover_letter.md               -> submission/Cover_Letter.docx
  03_data_and_code_availability.md -> submission/Data_and_Code_Availability.docx

Page setup, fonts and footer page numbering come from the body-free templates in
manuscript/templates/. Inline run formatting is written in the sources as
<b>, <i>, <sup> and <sub> tags; `code` spans render as plain text. Figures are
embedded from manuscript/figures/ immediately before their caption.

After building, the documents are checked and the build fails (exit 1) if a
stale title or phrase returns, required terminology is missing, an embedded
figure differs from its canonical PNG, a manuscript table disagrees with the
authoritative artifacts, or a document does not reopen as valid OOXML.

Run with: python scripts/manuscript/build_submission.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import zipfile
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "manuscript" / "source"
TEMPLATES = ROOT / "manuscript" / "templates"
FIGURES = ROOT / "manuscript" / "figures"
OUT = ROOT / "submission"

TITLE = (
    "Validation–Deployment Alignment Shapes How Internal Evaluation Predicts External "
    "Generalization in Lithium Solid-Electrolyte Machine Learning"
)
MAIN_FIGURES = {
    1: "fig1_schematic.png",
    2: "fig2_external_performance.png",
    3: "fig3_regime_utility.png",
    4: "fig4_seen_glass_crystalline.png",
    5: "fig5_extrapolation_failure.png",
    6: "fig6_risk_coverage_comparison.png",
}
SI_FIGURES = {1: "si/si_fig1_shift_alignment.png", 2: "si/si_fig2_parity.png"}

# Phrases that described pre-correction results or the old repository layout.
STALE_PHRASES = [
    "never opened",
    "Not rerun at any point",
    "the external inference was not rerun at any point",
    "Why N6 was never rerun",
    "downgrades 4 of 8",
    "two of which no longer exclude zero",
    "measurement-row estimand",
    "[−0.0562, −0.0028]",
    "ACS/JCIM",
    "manuscript_acs",
    "JCIM_",
    "npj_n",
    "scripts/acs",
    "reports/acs",
    "corrective01_pretrained_element_embedding",
    "verify_provenance_tables",
    "PLACEHOLDER",
    "\u00a7",  # section sign
    "Shapes External Generalization in",
    "NPJ-GO",
    "The formula-level external cohorts (204 strict and 239 relaxed records",
    "the formula-level external cohorts, model predictions",
    "the one-shot external transfer was not rerun",
    "The one-shot external transfer was not rerun",
    "archived with the code release",
    "preserved unmodified for provenance at",
    "archived separately",
    "Every reported comparison follows a written protocol frozen",
]
REQUIRED = {
    "Main_Manuscript.docx": [
        TITLE,
        "pooled elemental-property model",
        "feature-lookup bug",
        "Supplementary Note 27",
        "Use of AI assistance",
        "Data availability",
        "Code availability",
        "Competing interests",
        "Author contributions",
        "Corresponding author",
        "Supporting Information",
    ],
    "Supporting_Information.docx": [
        TITLE,
        "Supplementary Note 27",
        "Supplementary Note 33",
        "Md. Rifat Khandaker¹*, Mohammad Asaduzzaman Chowdhury²",
        "Department of Chemical Engineering, Dhaka University",
        "Department of Mechanical Engineering, Dhaka University",
        "*Corresponding author: Md. Rifat Khandaker",
        "not included in the curated public release",
    ],
    "Cover_Letter.docx": [
        TITLE,
        "Journal of Chemical Information and Modeling",
        "The primary external-transfer protocol was frozen and SHA-256 hash-locked before "
        "external targets were read.",
    ],
    "Data_and_Code_Availability.docx": [
        "RECONSTRUCTION.md",
        "not included in the public release",
        "restricted_data.py",
    ],
}

# Tables whose estimated height is at most this fraction of the text height are never
# broken across pages; taller tables break between rows so no near-empty page is left.
KEEP_WHOLE_MAX_FRACTION = 0.4
# Body lines this short are labels (e.g. "Strict cohort (n=204):") kept with the next block.
SHORT_LABEL_MAX_CHARS = 120

TAG = re.compile(r"(</?(?:b|i|sup|sub)>|`)")
CAPTION = re.compile(r"^(?P<label>(?:Supplementary )?(?:Fig\.|Table) (?P<num>\d+) \|)(?P<rest>.*)$")


# ------------------------------------------------------------------ rendering


def add_runs(paragraph, text: str, *, size: float | None = None, bold: bool = False) -> None:
    """Render inline-tagged text into runs."""
    state = {"b": bold, "i": False, "sup": False, "sub": False}
    for token in TAG.split(text.replace("\\|", "|")):
        if not token or token == "`":
            continue
        m = re.fullmatch(r"<(/?)(b|i|sup|sub)>", token)
        if m:
            state[m.group(2)] = m.group(1) == "" or (m.group(2) == "b" and bold)
            continue
        run = paragraph.add_run(token)
        run.bold = state["b"] or None
        run.italic = state["i"] or None
        if state["sup"]:
            run.font.superscript = True
        if state["sub"]:
            run.font.subscript = True
        if size is not None:
            run.font.size = Pt(size)


def para(
    doc, text: str = "", *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, bold=False, italic=False
):
    p = doc.add_paragraph()
    p.alignment = align
    if text:
        add_runs(p, text, size=size, bold=bold)
        if italic:
            for r in p.runs:
                r.italic = True
    return p


def caption(doc, line: str, size: float) -> None:
    """Bold label and first sentence, then the rest of the caption in regular weight."""
    m = CAPTION.match(line)
    assert m
    rest = m.group("rest")
    split = re.search(r"\.(\s|$)", rest)
    head, tail = (rest[: split.end()], rest[split.end() :]) if split else (rest, "")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.keep_with_next = m.group("label").startswith(
        ("Table", "Supplementary Table")
    )
    add_runs(p, (m.group("label") + head).rstrip() + (" " if tail else ""), size=size, bold=True)
    if tail:
        add_runs(p, tail, size=size)


def table(doc, rows: list[str]) -> None:
    cells = [[c.strip() for c in re.split(r"(?<!\\)\|", r.strip())[1:-1]] for r in rows]
    cells = [r for r in cells if not set("".join(r)) <= set("-: ")]
    t = doc.add_table(rows=len(cells), cols=max(len(r) for r in cells))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Rows never break across a page; the header row repeats on continuation pages.
    for i, tr in enumerate(t.rows):
        tr_pr = tr._tr.get_or_add_trPr()
        cant_split = OxmlElement("w:cantSplit")
        cant_split.set(qn("w:val"), "true")
        tr_pr.append(cant_split)
        if i == 0:
            repeat = OxmlElement("w:tblHeader")
            repeat.set(qn("w:val"), "true")
            tr_pr.append(repeat)
    widths, size = column_layout(doc, cells)
    n = len(cells)
    keep_whole = table_height(cells, widths, size) <= KEEP_WHOLE_MAX_FRACTION * text_height(doc)
    for i, row in enumerate(cells):
        for j, value in enumerate(row):
            p = t.rows[i].cells[j].paragraphs[0]
            add_runs(p, value, size=size, bold=(i == 0))
            # A short table is kept whole. A taller one may break between rows, but never
            # directly after its header or before its last two rows.
            if keep_whole:
                p.paragraph_format.keep_with_next = i < n - 1
            else:
                p.paragraph_format.keep_with_next = i < 2 or n - 3 <= i < n - 1
    t.autofit = False
    for j, width in enumerate(widths):
        t.columns[j].width = Emu(width)
        for cell in t.columns[j].cells:
            cell.width = Emu(width)
    doc.add_paragraph()


def text_em(text: str, *, bold: bool) -> float:
    """Approximate Times New Roman advance width of `text`, in em."""
    em = 0.0
    for ch in text:
        if ch.isdigit() or ch in "+−–$_":
            em += 0.5
        elif ch in "@%mwMW":
            em += 0.92
        elif ch.isupper():
            em += 0.69
        elif ch.islower():
            em += 0.47
        elif ch.isspace():
            em += 0.25
        else:
            em += 0.36
    return em * (1.06 if bold else 1.0)


def text_height(doc) -> int:
    section = doc.sections[-1]
    return section.page_height - section.top_margin - section.bottom_margin


def table_height(cells: list[list[str]], widths: list[int], size: float) -> int:
    """Estimated rendered table height (EMU): wrapped lines per row times line height."""
    pad = Inches(0.17)
    line = Pt(size * 1.2)
    total = 0
    for i, row in enumerate(cells):
        lines = 1
        for j, value in enumerate(row):
            usable = max(widths[j] - pad, 1)
            run = text_em(TAG.sub("", value), bold=i == 0) * Pt(size)
            lines = max(lines, -(-int(run) // usable))
        total += lines * line + Pt(4)
    return total


def column_layout(doc, cells: list[list[str]]) -> tuple[list[int], float]:
    """Column widths (EMU) and font size that keep words whole inside cells.

    Equal-width columns break words such as model names mid-word in wide tables.
    Each column gets at least its longest unbreakable word plus cell padding; the
    remaining text width is shared in proportion to how much text the column holds.
    A table whose words cannot fit at 9 pt is set at 8 pt; it never exceeds the page.
    """
    section = doc.sections[-1]
    text_width = section.page_width - section.left_margin - section.right_margin
    pad = Inches(0.17)  # left + right cell margins, with a small safety allowance
    ncols = max(len(r) for r in cells)
    for size in (9.0, 8.0):
        em = Pt(size)
        floor, desire = [pad] * ncols, [pad] * ncols
        for i, row in enumerate(cells):
            for j, value in enumerate(row):
                plain = TAG.sub("", value).replace("\\|", "|")
                # Lines break at spaces and after dashes. Long file paths wrap at the cell
                # edge whatever the width, so they are measured per path segment.
                words = [w for w in re.split(r"\s+|(?<=[/–])|(?<=[A-Za-z]-)", plain) if w] or [""]
                longest = max(text_em(w, bold=i == 0) for w in words)
                floor[j] = max(floor[j], int(longest * em) + pad)
                desire[j] = max(desire[j], int(min(text_em(plain, bold=i == 0), 14) * em) + pad)
        if sum(floor) <= text_width:
            break
    if sum(desire) <= text_width:
        widths = [int(d * text_width / sum(desire)) for d in desire]
    elif sum(floor) <= text_width:
        spare = text_width - sum(floor)
        extra = [d - f for d, f in zip(desire, floor, strict=True)]
        total_extra = sum(extra) or 1
        widths = [f + int(e * spare / total_extra) for f, e in zip(floor, extra, strict=True)]
    else:
        widths = [int(f * text_width / sum(floor)) for f in floor]
    return widths, size


def figure(doc, path: Path, width_in: float) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width_in))


def build_article(src: Path, template: Path, out: Path, *, si: bool) -> None:
    doc = docx.Document(str(template))
    lines = [ln.rstrip() for ln in src.read_text(encoding="utf-8").split("\n")]
    figures = SI_FIGURES if si else MAIN_FIGURES
    fig_width = 6.35 if si else 6.45
    in_refs = False
    front = True
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            table(doc, block)
            continue
        if line.startswith("#"):
            front = False
            level = len(line) - len(line.lstrip("#"))
            text = line.lstrip("#").strip()
            in_refs = text in {"References", "Supplementary References"}
            if si:
                h = para(doc, text, size=13 if level <= 3 else 11.5, bold=True)
            else:
                h = para(doc, text, size=13 if level <= 2 else 11.5, bold=True, italic=level >= 3)
            h.paragraph_format.keep_with_next = True
            i += 1
            continue
        if line == "Supplementary References":
            in_refs = True
            para(doc, line, size=13, bold=True)
            i += 1
            continue
        m = CAPTION.match(line)
        if m:
            if "Fig." in m.group("label"):
                figure(doc, FIGURES / figures[int(m.group("num"))], fig_width)
                caption(doc, line, 9.5)
            else:
                caption(doc, line, 10)
            i += 1
            continue
        if front:
            center = WD_ALIGN_PARAGRAPH.CENTER
            if si and line == "Supplementary Information":
                para(doc, line, align=center, size=15, bold=True)
            elif line == TITLE:
                para(
                    doc,
                    line,
                    align=center if not si else WD_ALIGN_PARAGRAPH.JUSTIFY,
                    size=14 if si else None,
                    bold=not si,
                )
            elif line.startswith(("¹", "²", "*Corresponding")):
                para(doc, line, align=center, size=10)
            elif line.startswith("Md. Rifat Khandaker"):
                para(doc, line, align=center)
            elif si and line == "Contents":
                para(doc, line, size=11.5, bold=True)
            else:
                para(doc, line)
            i += 1
            continue
        p = para(doc, line, size=10 if in_refs else None)
        following = next((ln.strip() for ln in lines[i + 1 :] if ln.strip()), "")
        if not in_refs and (following.startswith("|") or len(line) <= SHORT_LABEL_MAX_CHARS):
            # Table lead-ins and one-line labels stay with what they introduce.
            p.paragraph_format.keep_with_next = not following.startswith("#")
        if in_refs:
            p.paragraph_format.left_indent = Emu(288290)
            p.paragraph_format.first_line_indent = Emu(-288290)
        i += 1
    doc.save(str(out))


def build_plain(src: Path, template: Path, out: Path) -> None:
    doc = docx.Document(str(template))
    signature = False
    for raw in src.read_text(encoding="utf-8").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if signature:
            # Compact signature block: one line per paragraph, no gap within a signatory.
            sp = para(doc, line, align=WD_ALIGN_PARAGRAPH.LEFT)
            sp.paragraph_format.space_after = Pt(0)
            sp.paragraph_format.keep_with_next = True
            if line.startswith("Mohammad"):
                sp.paragraph_format.space_before = Pt(10)
            continue
        if line == "Sincerely,":
            signature = True
            para(doc, line, align=WD_ALIGN_PARAGRAPH.LEFT).paragraph_format.keep_with_next = True
            continue
        if line.startswith("# "):
            para(doc, line[2:], align=WD_ALIGN_PARAGRAPH.CENTER, size=14, bold=True)
        elif line.startswith("## "):
            para(doc, line[3:], size=12, bold=True)
        else:
            para(doc, line, align=WD_ALIGN_PARAGRAPH.LEFT)
    doc.save(str(out))


# ------------------------------------------------------------------ fail-closed checks


def document_text(path: Path) -> str:
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs]
    parts += [c.text for t in d.tables for row in t.rows for c in row.cells]
    return "\n".join(parts)


def check(outputs: dict[str, Path]) -> list[str]:
    errors: list[str] = []
    for name, path in outputs.items():
        with zipfile.ZipFile(path) as zf:
            bad = zf.testzip()
            if bad:
                errors.append(f"{name}: corrupt member {bad}")
        text = document_text(path)
        for phrase in STALE_PHRASES:
            if phrase in text:
                errors.append(f"{name}: stale phrase {phrase!r}")
        for needed in REQUIRED[name]:
            if needed not in text:
                errors.append(f"{name}: required text missing {needed!r}")
        long_scripts = [
            r.text
            for para_ in docx.Document(str(path)).paragraphs
            for r in para_.runs
            if (r.font.superscript or r.font.subscript) and len(r.text) > 12
        ]
        if long_scripts:
            errors.append(f"{name}: oversized superscript/subscript run {long_scripts[0][:40]!r}")
        if re.search(r"</?(?:b|i|sup|sub)>|\*\*|`|(?<![\w¹*\n])\*(?!Corresponding)[A-Za-z]", text):
            errors.append(f"{name}: unrendered markup")
    for name, figs in (
        ("Main_Manuscript.docx", MAIN_FIGURES),
        ("Supporting_Information.docx", SI_FIGURES),
    ):
        d = docx.Document(str(outputs[name]))
        embedded = [
            hashlib.sha256(rel.target_part.blob).hexdigest()
            for rel in d.part.rels.values()
            if "image" in rel.reltype
        ]
        canonical = [hashlib.sha256((FIGURES / f).read_bytes()).hexdigest() for f in figs.values()]
        if sorted(embedded) != sorted(canonical):
            errors.append(f"{name}: embedded figure bytes differ from manuscript/figures")
    sys.path.insert(0, str(ROOT / "scripts" / "manuscript"))
    import build_tables  # noqa: PLC0415

    built = {n: b() for n, b in build_tables.BUILDERS.items()}
    errors += [f"table provenance: {e}" for e in build_tables.verify(built)]
    errors += [f"table provenance: {e}" for e in build_tables.verify_si_table1()]
    return errors


def main() -> None:
    OUT.mkdir(exist_ok=True)
    outputs = {
        "Main_Manuscript.docx": OUT / "Main_Manuscript.docx",
        "Supporting_Information.docx": OUT / "Supporting_Information.docx",
        "Cover_Letter.docx": OUT / "Cover_Letter.docx",
        "Data_and_Code_Availability.docx": OUT / "Data_and_Code_Availability.docx",
    }
    build_article(
        SRC / "00_main.md",
        TEMPLATES / "main_template.docx",
        outputs["Main_Manuscript.docx"],
        si=False,
    )
    build_article(
        SRC / "01_si.md",
        TEMPLATES / "si_template.docx",
        outputs["Supporting_Information.docx"],
        si=True,
    )
    build_plain(
        SRC / "02_cover_letter.md", TEMPLATES / "letter_template.docx", outputs["Cover_Letter.docx"]
    )
    build_plain(
        SRC / "03_data_and_code_availability.md",
        TEMPLATES / "statement_template.docx",
        outputs["Data_and_Code_Availability.docx"],
    )
    errors = check(outputs)
    for e in errors:
        print("FAIL", e)
    for path in outputs.values():
        print(f"built {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
    print(f"{len(errors)} check failures")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
