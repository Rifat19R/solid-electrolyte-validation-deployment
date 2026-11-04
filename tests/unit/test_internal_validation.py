from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sse_transfer.evaluation.internal_validation import (
    MISSING_FAMILY,
    canonicalize_and_fill_family,
    family_group_kfold_splits,
    inner_early_stop_split,
    leave_one_major_family_out_splits,
    major_family_counts,
    random_kfold_splits,
    size_matched_random_splits,
)


def test_canonicalize_and_fill_family_applies_alias_and_fills_missing() -> None:
    raw = pd.Series(["argyrodite", "argyrodites", "Halide", None, "garnets"])
    out = canonicalize_and_fill_family(raw)
    assert out.tolist() == ["argyrodites", "argyrodites", "halides", MISSING_FAMILY, "garnets"]
    assert out.isna().sum() == 0


def test_major_family_counts_respects_threshold() -> None:
    families = pd.Series(["a"] * 20 + ["b"] * 14 + ["c"] * 15)
    counts = major_family_counts(families, min_n=15)
    assert counts == {"a": 20, "c": 15}


def test_random_kfold_splits_partition_all_rows_exactly_once_in_test() -> None:
    n = 37
    folds = random_kfold_splits(n, n_splits=5, seed=123)
    assert len(folds) == 5
    seen: list[int] = []
    for train_idx, test_idx in folds:
        assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(n))


def test_random_kfold_splits_deterministic_for_fixed_seed() -> None:
    a = random_kfold_splits(40, n_splits=5, seed=7)
    b = random_kfold_splits(40, n_splits=5, seed=7)
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=True):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(te_a, te_b)


def test_family_group_kfold_keeps_each_family_in_a_single_fold() -> None:
    families = pd.Series(["fam1"] * 10 + ["fam2"] * 8 + ["fam3"] * 6 + ["fam4"] * 6 + ["fam5"] * 5)
    folds = family_group_kfold_splits(families, n_splits=5, seed=11)
    fam_arr = families.to_numpy()
    for _, test_idx in folds:
        test_families = set(fam_arr[test_idx].tolist())
        # every family appearing in this fold's test set must not appear in
        # any other fold's test set (whole-group assignment)
        for _other_train_idx, other_test_idx in folds:
            if np.array_equal(other_test_idx, test_idx):
                continue
            assert test_families.isdisjoint(set(fam_arr[other_test_idx].tolist()))


def test_leave_one_major_family_out_splits_cover_and_partition() -> None:
    families = pd.Series(["fam1"] * 5 + ["fam2"] * 5 + ["small"] * 2)
    splits = leave_one_major_family_out_splits(families, ["fam1", "fam2"])
    assert set(splits.keys()) == {"fam1", "fam2"}
    for name, (train_idx, test_idx) in splits.items():
        assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))
        assert len(train_idx) + len(test_idx) == len(families)
        assert (families.to_numpy()[test_idx] == name).all()
        assert (families.to_numpy()[train_idx] != name).all()


def test_leave_one_major_family_out_splits_rejects_empty_family() -> None:
    families = pd.Series(["fam1"] * 5)
    with pytest.raises(ValueError):
        leave_one_major_family_out_splits(families, ["nonexistent"])


def test_inner_early_stop_split_disjoint_and_covers_all_rows() -> None:
    inner_train, inner_stop = inner_early_stop_split(50, stop_fraction=0.2, seed=5)
    assert set(inner_train.tolist()).isdisjoint(set(inner_stop.tolist()))
    assert sorted(inner_train.tolist() + inner_stop.tolist()) == list(range(50))
    assert len(inner_stop) == 10


def test_inner_early_stop_split_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        inner_early_stop_split(10, stop_fraction=1.0, seed=1)


def test_size_matched_random_splits_matches_reference_sizes_and_partitions() -> None:
    families = pd.Series(["fam1"] * 10 + ["fam2"] * 8 + ["fam3"] * 6 + ["fam4"] * 6 + ["fam5"] * 5)
    n = len(families)
    reference = family_group_kfold_splits(families, n_splits=5, seed=11)
    matched = size_matched_random_splits(n, reference, seed=99)
    assert len(matched) == len(reference)
    seen: list[int] = []
    for (_, ref_test), (train_idx, test_idx) in zip(reference, matched, strict=True):
        assert len(test_idx) == len(ref_test)
        assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))
        assert len(train_idx) + len(test_idx) == n
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(n))


def test_size_matched_random_splits_deterministic_and_seed_varies_membership() -> None:
    families = pd.Series(["fam1"] * 10 + ["fam2"] * 8 + ["fam3"] * 6 + ["fam4"] * 6 + ["fam5"] * 5)
    n = len(families)
    reference = family_group_kfold_splits(families, n_splits=5, seed=11)
    a = size_matched_random_splits(n, reference, seed=1)
    b = size_matched_random_splits(n, reference, seed=1)
    c = size_matched_random_splits(n, reference, seed=2)
    for (_, te_a), (_, te_b) in zip(a, b, strict=True):
        assert np.array_equal(te_a, te_b)
    assert any(not np.array_equal(te_a, te_c) for (_, te_a), (_, te_c) in zip(a, c, strict=True))


def test_size_matched_random_splits_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError):
        size_matched_random_splits(10, [(np.arange(5), np.arange(5, 8))], seed=1)
