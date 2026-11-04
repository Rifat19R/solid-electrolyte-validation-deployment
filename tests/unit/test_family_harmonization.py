import pandas as pd

from sse_transfer.data.family_harmonization import (
    FAMILY_ALIAS_MAP,
    canonical_family,
    detect_alias_clusters,
)


def test_canonical_family_maps_known_aliases() -> None:
    raw = pd.Series(["Halide", "argyrodite", "cholorides", "nitride"])
    result = canonical_family(raw)
    assert result.tolist() == ["halides", "argyrodites", "chlorides", "nitrides"]


def test_canonical_family_leaves_unmapped_values_unchanged() -> None:
    raw = pd.Series(["halides", "argyrodites", "NASICON", "LISICON", "garnet"])
    result = canonical_family(raw)
    assert result.tolist() == raw.tolist()


def test_canonical_family_passes_nan_through() -> None:
    raw = pd.Series(["Halide", None, "garnet"])
    result = canonical_family(raw)
    assert result.iloc[0] == "halides"
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == "garnet"


def test_canonical_family_output_has_no_remaining_alias_clusters() -> None:
    raw = pd.Series(list(FAMILY_ALIAS_MAP.keys()) + list(FAMILY_ALIAS_MAP.values()))
    harmonized = canonical_family(raw)
    clusters = detect_alias_clusters(harmonized)
    assert clusters["n_clusters"] == 0


def test_detect_alias_clusters_finds_known_true_positives() -> None:
    raw = pd.Series(
        [
            "Halide",
            "halides",
            "argyrodite",
            "argyrodites",
            "chlorides",
            "cholorides",
            "nitride",
            "nitrides",
            "garnet",
            "oxides",
        ]
    )
    result = detect_alias_clusters(raw)
    cluster_sets = [frozenset(c) for c in result["suspected_same_family_clusters"]]
    assert frozenset({"Halide", "halides"}) in cluster_sets
    assert frozenset({"argyrodite", "argyrodites"}) in cluster_sets
    assert frozenset({"chlorides", "cholorides"}) in cluster_sets
    assert frozenset({"nitride", "nitrides"}) in cluster_sets
    assert result["n_clusters"] == 4


def test_detect_alias_clusters_excludes_lisicon_nasicon_false_positive() -> None:
    raw = pd.Series(["LISICON", "NASICON", "garnet"])
    result = detect_alias_clusters(raw)
    assert result["n_clusters"] == 0
