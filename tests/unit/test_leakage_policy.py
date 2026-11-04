import pytest

from sse_transfer.data.leakage_policy import build_ledger, excluded_material_ids


def test_exact_match_against_test_split_is_excluded() -> None:
    matches = [{"query_id": "t1", "matched_id": "mp-1", "classification": "EXACT_STRUCTURE_MATCH"}]
    protected = {"t1": "final_test_protection"}
    result = excluded_material_ids(matches, protected_obelix_ids=protected)
    assert "mp-1" in result
    assert result["mp-1"] == [("t1", "EXACT_STRUCTURE_MATCH", "final_test_protection")]


def test_exact_match_against_unprotected_train_row_is_not_excluded() -> None:
    matches = [
        {"query_id": "trn1", "matched_id": "mp-2", "classification": "EXACT_STRUCTURE_MATCH"}
    ]
    result = excluded_material_ids(matches, protected_obelix_ids={})
    assert result == {}


def test_composition_only_match_is_never_excluded_even_if_protected() -> None:
    matches = [{"query_id": "t1", "matched_id": "mp-3", "classification": "COMPOSITION_ONLY_MATCH"}]
    protected = {"t1": "final_test_protection"}
    result = excluded_material_ids(matches, protected_obelix_ids=protected)
    assert result == {}


def test_probable_match_requires_a_recorded_review_decision() -> None:
    matches = [
        {
            "query_id": "unreviewed",
            "matched_id": "mp-999",
            "classification": "PROBABLE_STRUCTURE_MATCH",
        }
    ]
    protected = {"unreviewed": "final_test_protection"}
    with pytest.raises(ValueError, match="no recorded review decision"):
        excluded_material_ids(matches, protected_obelix_ids=protected)


def test_probable_match_reviewed_retain_is_not_excluded() -> None:
    # x2i / mp-1540429 is a real REVIEWED_PROBABLE_MATCHES RETAIN case.
    matches = [
        {
            "query_id": "x2i",
            "matched_id": "mp-1540429",
            "classification": "PROBABLE_STRUCTURE_MATCH",
        }
    ]
    protected = {"x2i": "final_test_protection"}
    result = excluded_material_ids(matches, protected_obelix_ids=protected)
    assert result == {}


def test_probable_match_reviewed_exclude_is_excluded_when_protected() -> None:
    # mdp / mp-641703 is a real REVIEWED_PROBABLE_MATCHES EXCLUDE case.
    matches = [
        {"query_id": "mdp", "matched_id": "mp-641703", "classification": "PROBABLE_STRUCTURE_MATCH"}
    ]
    protected = {"mdp": "final_test_protection"}
    result = excluded_material_ids(matches, protected_obelix_ids=protected)
    assert "mp-641703" in result


def test_build_ledger_cascades_material_exclusion_to_edge_datasets() -> None:
    excluded = {"mp-1": [("t1", "EXACT_STRUCTURE_MATCH", "final_test_protection")]}
    entries = build_ledger(
        excluded,
        bvel_material_ids={"mp-1", "mp-2"},
        bvse_edge_index={"mp-1": ["mp-1_0_1_0_0_0", "mp-1_0_2_0_0_0"]},
        dft_edge_index={"mp-1": ["mp-1_0_1_0_0_0"]},
    )
    datasets = {(e.original_dataset, e.record_id) for e in entries}
    assert ("bvel_percolation", "mp-1") in datasets
    assert ("bvse_hops", "mp-1_0_1_0_0_0") in datasets
    assert ("bvse_hops", "mp-1_0_2_0_0_0") in datasets
    assert ("dft_neb", "mp-1_0_1_0_0_0") in datasets
    assert len(entries) == 4


def test_build_ledger_skips_bvel_entry_when_material_not_in_bvel() -> None:
    excluded = {"mp-9": [("t1", "EXACT_STRUCTURE_MATCH", "final_test_protection")]}
    entries = build_ledger(excluded, bvel_material_ids=set(), bvse_edge_index={}, dft_edge_index={})
    assert entries == []
