"""Integrity checks on the frozen Phase 0C artifacts under data/processed/v1/.

Skips (does not fail) when the raw datasets or the processed-v1 artifacts are
not present locally, since both data/raw/ and data/processed/ are gitignored.
Run `python scripts/fetch_raw_data.py` then `python scripts/build_processed_v1.py`
to produce them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "v1"
LEAKAGE = ROOT / "reports" / "results" / "leakage"

pytestmark = pytest.mark.skipif(
    not (RAW / "obelix" / "all.csv").exists() or not (PROCESSED / "obelix_index.json").exists(),
    reason="raw data and/or processed v1 artifacts not present; run "
    "scripts/fetch_raw_data.py then scripts/build_processed_v1.py",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_obelix_split_is_disjoint_partition_of_all_rows() -> None:
    index = json.loads((PROCESSED / "obelix_index.json").read_text())
    train = set(index["splits"]["train"])
    val = set(index["splits"]["validation"])
    test = set(index["splits"]["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    all_ids = set(pd.read_csv(RAW / "obelix" / "all.csv")["ID"].astype(str))
    assert train | val | test == all_ids


def test_official_test_split_is_untouched() -> None:
    index = json.loads((PROCESSED / "obelix_index.json").read_text())
    raw_test_ids = set(pd.read_csv(RAW / "obelix" / "test.csv")["ID"].astype(str))
    assert set(index["splits"]["test"]) == raw_test_ids


def test_validation_split_never_breaks_a_composition_group() -> None:
    index = json.loads((PROCESSED / "obelix_index.json").read_text())
    train_df = pd.read_csv(RAW / "obelix" / "train.csv")
    comp_of = dict(zip(train_df["ID"].astype(str), train_df["Reduced Composition"], strict=True))

    train_comps = {comp_of[i] for i in index["splits"]["train"]}
    val_comps = {comp_of[i] for i in index["splits"]["validation"]}
    assert train_comps.isdisjoint(val_comps)


def test_obelix_split_freeze_sidecar_matches_file_on_disk() -> None:
    split_path = PROCESSED / "obelix_index.json"
    sidecar = json.loads(Path(f"{split_path}.sha256.json").read_text())
    assert sidecar["sha256"] == _sha256(split_path)


def test_exclusion_ledger_freeze_sidecar_matches_file_on_disk() -> None:
    ledger_path = LEAKAGE / "exclusion_ledger.json"
    sidecar = json.loads(Path(f"{ledger_path}.sha256.json").read_text())
    assert sidecar["sha256"] == _sha256(ledger_path)


def test_excluded_records_index_matches_included_flags_in_dataset_indices() -> None:
    excluded = json.loads((PROCESSED / "excluded_records_index.json").read_text())
    transport = json.loads((PROCESSED / "transport_pretraining_index.json").read_text())
    dft = json.loads((PROCESSED / "dft_neb_index.json").read_text())

    bvel_excluded_flagged = {
        row["material_id"] for row in transport["bvel_percolation"] if not row["included"]
    }
    bvse_excluded_flagged = {
        row["edge_id"] for row in transport["bvse_hops"] if not row["included"]
    }
    dft_excluded_flagged = {row["edge_id"] for row in dft["dft_neb"] if not row["included"]}

    assert set(excluded["bvel_percolation"]) == bvel_excluded_flagged
    assert set(excluded["bvse_hops"]) == bvse_excluded_flagged
    assert set(excluded["dft_neb"]) == dft_excluded_flagged


def test_no_official_test_or_validation_structure_match_survives_in_pretraining() -> None:
    """Final-test/validation protection: every EXACT_STRUCTURE_MATCH pretraining
    material implicated by an OBELiX test or validation row must be excluded."""
    index = json.loads((PROCESSED / "obelix_index.json").read_text())
    protected_ids = set(index["splits"]["test"]) | set(index["splits"]["validation"])

    struct_overlap = json.loads(
        (LEAKAGE / "obelix_vs_pretraining_structure_overlap.json").read_text()
    )
    excluded_materials = set(
        json.loads((PROCESSED / "excluded_records_index.json").read_text())["bvel_percolation"]
    )

    for match in struct_overlap["matches"]:
        if match["classification"] != "EXACT_STRUCTURE_MATCH":
            continue
        if match["query_id"] in protected_ids:
            assert match["matched_id"] in excluded_materials, (
                f"{match['matched_id']} matches protected OBELiX id "
                f"{match['query_id']} but was not excluded"
            )


def test_disorder_is_preserved_for_every_locally_available_obelix_structure() -> None:
    """Regression test: preprocessing must not silently round fractional
    occupancy for any structure referenced by the frozen train/validation/test
    split (DATA_CONTRACT.md 'Disorder')."""
    from sse_transfer.data.obelix_loader import available_cif_ids, load_structure

    index = json.loads((PROCESSED / "obelix_index.json").read_text())
    all_split_ids = (
        set(index["splits"]["train"])
        | set(index["splits"]["validation"])
        | set(index["splits"]["test"])
    )
    cif_backed_ids = all_split_ids & available_cif_ids()
    assert len(cif_backed_ids) > 0

    n_disordered = 0
    for row_id in cif_backed_ids:
        structure = load_structure(row_id)
        if not structure.is_ordered:
            n_disordered += 1
            fractional = [occ for site in structure for occ in site.species.values() if 0 < occ < 1]
            assert fractional, (
                f"{row_id} is disordered but no fractional (0,1) occupancies survived loading"
            )

    # Matches the Phase 0B baseline fraction (249/321 ~ 0.7757) within a small
    # tolerance driven by which CIF-backed ids happen to fall in each split.
    assert n_disordered / len(cif_backed_ids) > 0.5
