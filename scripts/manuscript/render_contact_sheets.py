#!/usr/bin/env python3
"""Convert the submission DOCX files to PDF (headless LibreOffice) and render
contact-sheet PNG composites (PyMuPDF) for visual page-by-page inspection.
Also reports any page without text or images (blank page check).

Run: python scripts/manuscript/render_contact_sheets.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = ROOT / "submission"
CONTACT_SHEET_DIR = ROOT / "submission" / "contact_sheets"

DOCX_FILES = [
    "Main_Manuscript.docx",
    "Supporting_Information.docx",
    "Cover_Letter.docx",
    "Data_and_Code_Availability.docx",
]

PAGES_PER_SHEET = 6  # 2x3 grid
COLS = 2
ZOOM = 1.5


def convert_to_pdf(docx_path: Path) -> Path:
    subprocess.run(
        [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(SUBMISSION_DIR),
            str(docx_path),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    return docx_path.with_suffix(".pdf")


def render_contact_sheets(pdf_path: Path, out_prefix: str) -> list[Path]:
    doc = fitz.open(str(pdf_path))
    n_pages = doc.page_count
    page_images: list[Image.Image] = []
    for pno in range(n_pages):
        pix = doc[pno].get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
        page_images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
        if not doc[pno].get_text().strip() and not doc[pno].get_images():
            print(f"BLANK PAGE: {pdf_path.name} page {pno + 1}")
    doc.close()

    out_paths: list[Path] = []
    sheet_idx = 0
    for start in range(0, n_pages, PAGES_PER_SHEET):
        chunk = page_images[start : start + PAGES_PER_SHEET]
        rows = (len(chunk) + COLS - 1) // COLS
        cell_w = max(im.width for im in chunk)
        cell_h = max(im.height for im in chunk)
        sheet = Image.new("RGB", (cell_w * COLS, cell_h * rows), (255, 255, 255))
        for i, im in enumerate(chunk):
            row, col = divmod(i, COLS)
            sheet.paste(im, (col * cell_w, row * cell_h))
        end_page = start + len(chunk)
        out_path = (
            CONTACT_SHEET_DIR / f"{out_prefix}_sheet{sheet_idx:02d}_pages{start + 1}-{end_page}.png"
        )
        sheet.save(str(out_path))
        out_paths.append(out_path)
        sheet_idx += 1
    return out_paths


def main() -> int:
    CONTACT_SHEET_DIR.mkdir(parents=True, exist_ok=True)
    all_sheets: list[Path] = []
    for name in DOCX_FILES:
        docx_path = SUBMISSION_DIR / name
        if not docx_path.exists():
            print(f"ERROR: {docx_path} does not exist", file=sys.stderr)
            return 1
        for stale in CONTACT_SHEET_DIR.glob(f"{docx_path.stem}_sheet*.png"):
            stale.unlink()
        pdf_path = convert_to_pdf(docx_path)
        n_pages = fitz.open(str(pdf_path)).page_count
        print(f"{name}: {n_pages} pages -> {pdf_path.name}")
        sheets = render_contact_sheets(pdf_path, docx_path.stem)
        all_sheets.extend(sheets)
        print(f"  {len(sheets)} contact sheet(s) written")
    print("\nAll contact sheets:")
    for s in all_sheets:
        print(" ", s.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
