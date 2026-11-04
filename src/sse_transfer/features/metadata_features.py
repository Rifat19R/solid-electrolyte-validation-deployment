"""Deterministic global structural/metadata features (non-composition).

DATA_CONTRACT.md: features must never include row IDs or target-derived
columns. Only crystallographic metadata available independently of the
conductivity measurement is used: formula units (Z), space group number,
lattice parameters, a derived unit-cell volume, and the dataset curators'
structural "Family" label.
"""

from __future__ import annotations

import math

import pandas as pd

NUMERIC_METADATA_COLUMNS: tuple[str, ...] = (
    "Z",
    "Space group #",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
)
FAMILY_COLUMN = "Family"


def cell_volume_angstrom3(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> float:
    """Standard triclinic unit-cell volume from lattice parameters (degrees).

    Purely deterministic function of raw inputs; not fit to any data.
    """
    ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
    factor = (
        1.0
        - math.cos(ar) ** 2
        - math.cos(br) ** 2
        - math.cos(gr) ** 2
        + 2.0 * math.cos(ar) * math.cos(br) * math.cos(gr)
    )
    return a * b * c * math.sqrt(max(factor, 0.0))


def metadata_features(raw: pd.DataFrame) -> pd.DataFrame:
    """`raw` must contain NUMERIC_METADATA_COLUMNS and FAMILY_COLUMN.

    Returns a frame aligned to `raw`'s index: numeric lattice/space-group
    columns plus a derived cell volume, and the raw (unencoded) Family
    string for a caller-supplied, train-fitted categorical encoder.
    """
    out = raw[list(NUMERIC_METADATA_COLUMNS)].astype(float).copy()
    out["cell_volume_A3"] = [
        cell_volume_angstrom3(*row)
        for row in out[["a", "b", "c", "alpha", "beta", "gamma"]].itertuples(index=False)
    ]
    out[FAMILY_COLUMN] = raw[FAMILY_COLUMN].astype("string")
    return out
