import math

import pandas as pd

from sse_transfer.features.metadata_features import (
    NUMERIC_METADATA_COLUMNS,
    cell_volume_angstrom3,
    metadata_features,
)


def test_cell_volume_orthorhombic_is_simple_product() -> None:
    volume = cell_volume_angstrom3(2.0, 3.0, 4.0, 90.0, 90.0, 90.0)
    assert math.isclose(volume, 24.0, rel_tol=1e-9)


def test_cell_volume_triclinic_matches_hand_computation() -> None:
    a, b, c, alpha, beta, gamma = 5.0, 6.0, 7.0, 80.0, 85.0, 75.0
    ar, br, gr = math.radians(alpha), math.radians(beta), math.radians(gamma)
    expected = (
        a
        * b
        * c
        * math.sqrt(
            1
            - math.cos(ar) ** 2
            - math.cos(br) ** 2
            - math.cos(gr) ** 2
            + 2 * math.cos(ar) * math.cos(br) * math.cos(gr)
        )
    )
    assert math.isclose(cell_volume_angstrom3(a, b, c, alpha, beta, gamma), expected, rel_tol=1e-12)


def test_metadata_features_returns_expected_columns_and_no_ids() -> None:
    raw = pd.DataFrame(
        {
            "Z": [4],
            "Space group #": [225],
            "a": [10.0],
            "b": [10.0],
            "c": [10.0],
            "alpha": [90.0],
            "beta": [90.0],
            "gamma": [90.0],
            "Family": ["garnet"],
            "ID": ["abc"],
            "Ionic conductivity (S cm-1)": ["1e-4"],
        }
    )
    out = metadata_features(raw)
    for col in NUMERIC_METADATA_COLUMNS:
        assert col in out.columns
    assert "cell_volume_A3" in out.columns
    assert "Family" in out.columns
    assert "ID" not in out.columns
    assert "Ionic conductivity (S cm-1)" not in out.columns
    assert math.isclose(out.loc[0, "cell_volume_A3"], 1000.0, rel_tol=1e-9)


def test_metadata_features_preserves_missing_family_as_na() -> None:
    raw = pd.DataFrame(
        {
            "Z": [4],
            "Space group #": [225],
            "a": [10.0],
            "b": [10.0],
            "c": [10.0],
            "alpha": [90.0],
            "beta": [90.0],
            "gamma": [90.0],
            "Family": [None],
        }
    )
    out = metadata_features(raw)
    assert out.loc[0, "Family"] is pd.NA
