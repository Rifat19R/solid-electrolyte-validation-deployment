"""Non-destructive readers for the OBELiX raw dataset (final experimental task).

Reads only from data/raw/obelix/; never writes there. See DATA_CONTRACT.md
dataset role D. CIF coordinates in `extracted/all_randomized_cifs/` have
randomization noise applied by the upstream authors while preserving
symmetry (original ICSD/COD-sourced CIFs are not redistributed publicly).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "obelix"
CIF_DIR = RAW / "extracted" / "all_randomized_cifs"

CENSOR_PATTERN = r"^\s*[<>]"

#: DATA_CONTRACT.md "Censoring": machine-readable label-status categories.
#: LEFT_CENSORED = true value known only to be below the reported bound
#: (e.g. "<1E-10": below detection limit). RIGHT_CENSORED = true value known
#: only to be above the reported bound (e.g. ">5e-3"). EXACT = a parseable
#: point value. MISSING = null/unparseable.
LABEL_STATUS_EXACT = "EXACT"
LABEL_STATUS_LEFT_CENSORED = "LEFT_CENSORED"
LABEL_STATUS_RIGHT_CENSORED = "RIGHT_CENSORED"
LABEL_STATUS_MISSING = "MISSING"

_NUMERIC_RE = re.compile(r"^\s*[-+]?[0-9][0-9.eE+-]*\s*$")


def load_obelix(split: str = "all") -> pd.DataFrame:
    """split: 'all', 'train', or 'test' (official split provided by the authors)."""
    if split not in {"all", "train", "test"}:
        raise ValueError(f"Unknown split: {split}")
    return pd.read_csv(RAW / f"{split}.csv")


def is_censored(conductivity_str: pd.Series) -> pd.Series:
    """True where the reported conductivity is an inequality, not an exact value."""
    return conductivity_str.astype(str).str.match(CENSOR_PATTERN)


def _label_status_one(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return LABEL_STATUS_MISSING
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return LABEL_STATUS_MISSING
    if text.startswith("<"):
        return LABEL_STATUS_LEFT_CENSORED
    if text.startswith(">"):
        return LABEL_STATUS_RIGHT_CENSORED
    if _NUMERIC_RE.match(text):
        return LABEL_STATUS_EXACT
    return LABEL_STATUS_MISSING


def label_status(conductivity_str: pd.Series) -> pd.Series:
    """Classify each raw conductivity value into EXACT / LEFT_CENSORED /
    RIGHT_CENSORED / MISSING. Never converts a censored value into an exact one.
    """
    return conductivity_str.apply(_label_status_one)


def censoring_bound(conductivity_str: pd.Series) -> pd.Series:
    """Numeric bound extracted from the raw string (the point value for EXACT,
    the detection-limit bound for LEFT_/RIGHT_CENSORED, NaN for MISSING).
    """

    def _bound(raw: object) -> float:
        status = _label_status_one(raw)
        if status == LABEL_STATUS_MISSING:
            return float("nan")
        text = str(raw).strip().lstrip("<>").strip()
        return float(text)

    return conductivity_str.apply(_bound)


def available_cif_ids() -> set[str]:
    return {p.stem for p in CIF_DIR.glob("*.cif")}


def load_structure(row_id: str) -> Structure:
    path = CIF_DIR / f"{row_id}.cif"
    if not path.exists():
        raise FileNotFoundError(f"No local CIF for OBELiX row id {row_id!r}")
    return Structure.from_file(path)


def iter_structures() -> Iterator[tuple[str, Structure]]:
    """Yield (row_id, pymatgen Structure) for every OBELiX row with a local CIF."""
    for cif_id in sorted(available_cif_ids()):
        yield cif_id, load_structure(cif_id)
