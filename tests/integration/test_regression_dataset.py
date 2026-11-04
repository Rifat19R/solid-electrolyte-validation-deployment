"""Phase 1 integration checks for the frozen OBELiX regression dataset.

Skips (does not fail) when raw data / processed-v1 artifacts are absent
locally, matching tests/integration/test_processed_v1.py's convention,
since both data/raw/ and data/processed/ are gitignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sse_transfer.data.regression_dataset import (
    SPLIT_PATH,
    SplitIntegrityError,
    load_censored_train_frame,
    load_regression_dataset,
    verify_split_hash,
)
from sse_transfer.features.build_features import RAW_INPUT_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

pytestmark = pytest.mark.skipif(
    not (RAW / "obelix" / "all.csv").exists() or not SPLIT_PATH.exists(),
    reason="raw data and/or processed v1 artifacts not present; run "
    "scripts/fetch_raw_data.py then scripts/build_processed_v1.py",
)

LEAKY_COLUMNS = {
    "ID",
    "Ionic conductivity (S cm-1)",
    "IC (Total)",
    "IC (Bulk)",
    "DOI",
    "Checked",
    "Ref",
    "Cif ID",
    "Cif ref_1",
    "Cif ref_2",
    "note",
    "close match",
    "close match DOI",
    "ICSD ID",
    "Laskowski ID",
    "Liion ID",
}


def test_verify_split_hash_passes_on_untouched_split() -> None:
    verify_split_hash()  # must not raise


def test_verify_split_hash_fails_on_tampered_split(tmp_path) -> None:
    tampered = tmp_path / "obelix_index.json"
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    payload["splits"]["train"] = payload["splits"]["train"][:-1]
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    real_sidecar = SPLIT_PATH.with_suffix(SPLIT_PATH.suffix + ".sha256.json")
    sidecar = tmp_path / "obelix_index.json.sha256.json"
    sidecar.write_text(real_sidecar.read_text(encoding="utf-8"))

    with pytest.raises(SplitIntegrityError):
        verify_split_hash(tampered)


def test_load_regression_dataset_excludes_censored_rows() -> None:
    dataset = load_regression_dataset()
    n_total = len(dataset.train) + len(dataset.validation) + len(dataset.test)
    assert n_total == 562  # frozen EXACT-only row count (Phase 0C evidence summary)
    assert dataset.n_censored_excluded == 37


def test_load_regression_dataset_splits_are_disjoint_and_match_frozen_membership() -> None:
    dataset = load_regression_dataset()
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))

    ids_train, ids_val, ids_test = (
        set(dataset.train["ID"]),
        set(dataset.validation["ID"]),
        set(dataset.test["ID"]),
    )
    assert ids_train.isdisjoint(ids_val)
    assert ids_train.isdisjoint(ids_test)
    assert ids_val.isdisjoint(ids_test)

    assert ids_train <= set(split["splits"]["train"])
    assert ids_val <= set(split["splits"]["validation"])
    assert ids_test <= set(split["splits"]["test"])


def test_load_regression_dataset_target_has_no_nan_or_inf() -> None:
    dataset = load_regression_dataset()
    import numpy as np

    for frame in (dataset.train, dataset.validation, dataset.test):
        assert np.isfinite(frame["y"].to_numpy()).all()


def test_load_regression_dataset_features_contain_no_leaky_columns() -> None:
    dataset = load_regression_dataset()
    assert LEAKY_COLUMNS.isdisjoint(set(dataset.feature_columns))


def test_load_regression_dataset_is_deterministic() -> None:
    first = load_regression_dataset()
    second = load_regression_dataset()
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)


def test_regression_dataset_module_never_imports_pretraining_loaders() -> None:
    """Static guard: Phase 1 classical baselines must never touch
    pretraining raw data, so no pretraining-leakage-excluded record can
    reappear through this code path (CLAUDE.md Phase 1 'Validate inputs').
    `litraj_loader` is the only module that reads bvel_percolation/
    bvse_hops/dft_neb raw data; the regression-dataset code path must not
    import it. Checked via AST rather than sys.modules, which other tests
    in the same session may have already populated."""
    import ast

    import sse_transfer.data.regression_dataset as module
    import sse_transfer.features.build_features as features_module

    for mod in (module, features_module):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
            elif isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
        assert not any("litraj_loader" in name for name in imported_names)


def test_raw_input_columns_exclude_target_and_ids() -> None:
    assert LEAKY_COLUMNS.isdisjoint(set(RAW_INPUT_COLUMNS))


# ---------------------------------------------------------------------------
# Phase 2.6A: load_censored_train_frame
# ---------------------------------------------------------------------------


def test_load_censored_train_frame_row_count_matches_known_split_audit() -> None:
    # Directly audited (23 train / 6 validation / 8 test of the 37 total
    # LEFT_CENSORED rows; 0 RIGHT_CENSORED, 0 MISSING in this dataset).
    frame = load_censored_train_frame()
    assert len(frame) == 23


def test_load_censored_train_frame_is_train_split_only() -> None:
    frame = load_censored_train_frame()
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    assert set(frame["ID"]) <= set(split["splits"]["train"])
    assert set(frame["ID"]).isdisjoint(set(split["splits"]["validation"]))
    assert set(frame["ID"]).isdisjoint(set(split["splits"]["test"]))
    assert (frame["obelix_split"] == "train").all()


def test_load_censored_train_frame_disjoint_from_exact_train_ids() -> None:
    dataset = load_regression_dataset()
    frame = load_censored_train_frame()
    assert set(frame["ID"]).isdisjoint(set(dataset.train["ID"]))


def test_load_censored_train_frame_y_is_always_nan() -> None:
    import numpy as np

    frame = load_censored_train_frame()
    assert frame["y"].isna().all()
    assert np.isfinite(frame["censor_bound_log10"].to_numpy()).all()


def test_load_censored_train_frame_direction_is_left_or_right_only() -> None:
    frame = load_censored_train_frame()
    assert set(frame["censor_direction"].unique()) <= {"LEFT_CENSORED", "RIGHT_CENSORED"}


def test_load_censored_train_frame_no_leaky_columns() -> None:
    frame = load_censored_train_frame()
    non_feature_columns = {"ID", "y", "obelix_split", "censor_direction", "censor_bound_log10"}
    feature_columns = set(frame.columns) - non_feature_columns
    assert LEAKY_COLUMNS.isdisjoint(feature_columns)


def test_load_censored_train_frame_is_deterministic() -> None:
    first = load_censored_train_frame()
    second = load_censored_train_frame()
    pd.testing.assert_frame_equal(first, second)
