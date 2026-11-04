"""Unit tests for N6's pure functions (run-authorization guard, cohort
scoring) -- no real model fitting, no Hargreaves target ever opened here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "transport_analysis"))

from n6_external_transfer import check_run_authorization, score_cohort  # noqa: E402


def test_check_run_authorization_first_run(tmp_path: Path) -> None:
    out = tmp_path / "n6.json"
    run_count, authorized = check_run_authorization(out, force_unauthorized_rerun=False)
    assert run_count == 1
    assert authorized is True


def test_check_run_authorization_refuses_second_primary_run(tmp_path: Path) -> None:
    out = tmp_path / "n6.json"
    out.write_text(json.dumps({"run_count": 1, "authorized": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="one-shot"):
        check_run_authorization(out, force_unauthorized_rerun=False)


def test_check_run_authorization_forced_rerun_is_flagged_non_primary(tmp_path: Path) -> None:
    out = tmp_path / "n6.json"
    out.write_text(json.dumps({"run_count": 1, "authorized": True}), encoding="utf-8")
    run_count, authorized = check_run_authorization(out, force_unauthorized_rerun=True)
    assert run_count == 2
    assert authorized is False


def test_check_run_authorization_allows_rerun_after_prior_nonprimary(tmp_path: Path) -> None:
    out = tmp_path / "n6.json"
    out.write_text(json.dumps({"run_count": 2, "authorized": False}), encoding="utf-8")
    run_count, authorized = check_run_authorization(out, force_unauthorized_rerun=False)
    assert run_count == 3
    assert authorized is False


def test_score_cohort_perfect_prediction_zero_mae() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    preds = {"model_a": np.array([1.0, 2.0, 3.0])}
    out = score_cohort(preds, y_true)
    assert out["model_a"]["mae"] == pytest.approx(0.0)


def test_score_cohort_multiple_models_independent() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    preds = {
        "good": np.array([0.0, 1.0, 2.0, 3.0]),
        "bad": np.array([10.0, 10.0, 10.0, 10.0]),
    }
    out = score_cohort(preds, y_true)
    assert out["good"]["mae"] < out["bad"]["mae"]
    assert set(out.keys()) == {"good", "bad"}


# --- Merge routing: the primary one-shot record can never be rewritten -------


def _fake_fragments(directory: Path) -> None:
    import n6_external_transfer as n6

    directory.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(n6.MODEL_NAMES):
        fragment = {
            "obelix_dev_population_n": 449,
            "obelix_test_untouched_row_count": 113,
            "runtime_seconds": 1.0,
            "strict": {"mae": 1.0 + i / 10},
            "relaxed": {"mae": 1.1 + i / 10},
        }
        (directory / f"{name}.json").write_text(json.dumps(fragment), encoding="utf-8")


def _primary(tmp_path: Path) -> Path:
    primary = tmp_path / "primary.json"
    primary.write_text(json.dumps({"run_count": 1, "authorized": True}), encoding="utf-8")
    return primary


def test_merge_refuses_without_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import n6_external_transfer as n6

    monkeypatch.setattr(n6, "_write_report", lambda manifest: None)
    _fake_fragments(tmp_path / "frags")
    primary = _primary(tmp_path)
    before = primary.read_bytes()
    with pytest.raises(RuntimeError, match="one-shot"):
        n6.merge_fragments(False, tmp_path / "frags", primary, tmp_path / "out.json")
    assert primary.read_bytes() == before
    assert not (tmp_path / "out.json").exists()


def test_forced_merge_is_nonprimary_and_leaves_primary_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import n6_external_transfer as n6

    monkeypatch.setattr(n6, "_write_report", lambda manifest: None)
    _fake_fragments(tmp_path / "frags")
    primary = _primary(tmp_path)
    before = primary.read_bytes()
    out = n6.merge_fragments(True, tmp_path / "frags", primary, tmp_path / "out.json")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["authorized"] is False
    assert written["run_count"] == 2
    assert written["status"].startswith("NON-PRIMARY")
    assert primary.read_bytes() == before


def test_merge_requires_existing_primary_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import n6_external_transfer as n6

    monkeypatch.setattr(n6, "_write_report", lambda manifest: None)
    _fake_fragments(tmp_path / "frags")
    with pytest.raises(RuntimeError, match="never creates a new primary"):
        n6.merge_fragments(True, tmp_path / "frags", tmp_path / "missing.json", tmp_path / "o.json")


def test_merge_cannot_target_protected_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import n6_external_transfer as n6

    from sse_transfer.utils.output_guard import ProtectedOutputError

    monkeypatch.setattr(n6, "_write_report", lambda manifest: None)
    _fake_fragments(tmp_path / "frags")
    primary = _primary(tmp_path)
    for target in (n6.PRIMARY_RECORD, ROOT / "artifacts" / "authoritative" / "x.json"):
        with pytest.raises(ProtectedOutputError):
            n6.merge_fragments(True, tmp_path / "frags", primary, target)


def test_default_routes_never_point_at_recorded_results() -> None:
    import n6_external_transfer as n6

    from sse_transfer.utils.output_guard import is_protected

    assert is_protected(n6.PRIMARY_RECORD)
    assert is_protected(n6.FRAGMENTS_DIR)
    assert not is_protected(n6.RERUN_JSON)
    assert not is_protected(n6.FRAGMENT_OUT_DIR)
    assert not is_protected(n6.REPORT_MD)
