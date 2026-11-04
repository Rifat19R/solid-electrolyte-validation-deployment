"""OBELiX parser row-count, target/unit, and disorder-preservation checks against
the locally downloaded raw dataset. Skips (does not fail) when the raw data has
not been fetched via scripts/fetch_raw_data.py, since data/raw/ is gitignored
and not part of the repository.

The LiTraj (BVEL/BVSE/DFT-NEB) parser checks of the closed earlier project are
in tests/closed_project/test_pretraining_parsers.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

pytestmark = pytest.mark.skipif(
    not (RAW / "obelix" / "all.csv").exists(),
    reason="raw datasets not present locally; run scripts/fetch_raw_data.py",
)


def test_obelix_row_counts_and_official_split() -> None:
    from sse_transfer.data.obelix_loader import load_obelix

    assert len(load_obelix("all")) == 599
    assert len(load_obelix("train")) == 478
    assert len(load_obelix("test")) == 121


def test_obelix_censored_target_count() -> None:
    from sse_transfer.data.obelix_loader import is_censored, load_obelix

    df = load_obelix("all")
    assert int(is_censored(df["Ionic conductivity (S cm-1)"]).sum()) == 37


def test_obelix_fractional_occupancy_is_preserved() -> None:
    """Fractional occupancies must survive CIF parsing untouched (DATA_CONTRACT.md
    'Disorder': do not round occupancies to a single species)."""
    from sse_transfer.data.obelix_loader import iter_structures

    n_total = 0
    n_disordered = 0
    fractional_occupancies_seen = []
    for _cif_id, structure in iter_structures():
        n_total += 1
        if not structure.is_ordered:
            n_disordered += 1
            for site in structure:
                fractional_occupancies_seen.extend(
                    occ for occ in site.species.values() if 0 < occ < 1
                )
    assert n_total == 321
    assert n_disordered == 249
    # occupancies must be preserved as real fractions, not rounded to 0/1
    assert any(0 < occ < 1 for occ in fractional_occupancies_seen)
