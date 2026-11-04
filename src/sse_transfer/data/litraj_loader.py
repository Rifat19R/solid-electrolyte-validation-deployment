"""Non-destructive readers for the LiTraj raw datasets (BVEL13k, nebBVSE122k, nebDFT2k).

Reads only from data/raw/<dataset>/extracted/; never writes there. See
DATA_CONTRACT.md dataset roles A (bvel_percolation), B (bvse_hops), and
C (dft_neb).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from ase.io import read as ase_read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"

_ADAPTOR = AseAtomsAdaptor()


def load_bvel13k_index() -> pd.DataFrame:
    return pd.read_csv(RAW / "bvel_percolation" / "extracted" / "BVEL13k" / "BVEL13k_index.csv")


def load_nebbvse122k_index() -> pd.DataFrame:
    return pd.read_csv(RAW / "bvse_hops" / "extracted" / "nebBVSE122k" / "nebBVSE122k_index.csv")


def load_nebdft2k_index() -> pd.DataFrame:
    return pd.read_csv(RAW / "dft_neb" / "extracted" / "nebDFT2k" / "nebDFT2k_index.csv")


def iter_bvel13k_structures(split: str | None = None) -> Iterator[tuple[str, Structure]]:
    """Yield (material_id, pymatgen Structure) for BVEL13k whole-crystal structures.

    `material_id` is read from each frame's `atoms.info` (embedded by the
    source dataset), not inferred from row position.
    """
    splits = [split] if split else ["train", "val", "test"]
    base = RAW / "bvel_percolation" / "extracted" / "BVEL13k"
    for s in splits:
        atoms_list = ase_read(base / f"BVEL13k_{s}.xyz", index=":")
        for atoms in atoms_list:
            structure = _ADAPTOR.get_structure(atoms)
            yield atoms.info["material_id"], structure
