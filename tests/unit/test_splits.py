import pandas as pd

from sse_transfer.data.splits import build_composition_group_validation_split


def _fixture_df() -> pd.DataFrame:
    rows = []
    # 5 singleton-composition rows
    for i in range(5):
        rows.append({"ID": f"single_{i}", "Reduced Composition": f"CompoundS{i}"})
    # 3 groups of 5 rows sharing a composition each (polymorphs/duplicate measurements)
    for g in range(3):
        for j in range(5):
            rows.append({"ID": f"grp{g}_{j}", "Reduced Composition": f"CompoundG{g}"})
    return pd.DataFrame(rows)


def test_composition_groups_never_split_across_train_and_validation() -> None:
    df = _fixture_df()
    result = build_composition_group_validation_split(df, seed=1, target_fraction=0.3)

    comp_of = dict(zip(df["ID"], df["Reduced Composition"], strict=True))
    train_comps = {comp_of[i] for i in result.train_ids}
    val_comps = {comp_of[i] for i in result.validation_ids}
    assert train_comps.isdisjoint(val_comps)


def test_split_is_a_disjoint_partition_of_all_rows() -> None:
    df = _fixture_df()
    result = build_composition_group_validation_split(df, seed=7, target_fraction=0.25)

    all_ids = set(df["ID"])
    assert set(result.train_ids).isdisjoint(result.validation_ids)
    assert set(result.train_ids) | set(result.validation_ids) == all_ids


def test_same_seed_is_deterministic() -> None:
    df = _fixture_df()
    a = build_composition_group_validation_split(df, seed=42, target_fraction=0.3)
    b = build_composition_group_validation_split(df, seed=42, target_fraction=0.3)
    assert a.train_ids == b.train_ids
    assert a.validation_ids == b.validation_ids


def test_different_seeds_can_produce_different_splits() -> None:
    df = _fixture_df()
    a = build_composition_group_validation_split(df, seed=1, target_fraction=0.3)
    b = build_composition_group_validation_split(df, seed=2, target_fraction=0.3)
    assert (a.validation_ids, a.validation_groups) != (b.validation_ids, b.validation_groups)


def test_target_fraction_out_of_range_rejected() -> None:
    df = _fixture_df()
    try:
        build_composition_group_validation_split(df, seed=1, target_fraction=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for out-of-range target_fraction")
