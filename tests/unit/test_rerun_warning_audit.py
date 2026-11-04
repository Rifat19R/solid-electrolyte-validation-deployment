"""Rerun logs: only the exact audited upstream warnings are accepted."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "release"))

import rerun_pipeline  # noqa: E402

NESTED = (
    "/x/.venv/lib/python3.12/site-packages/torch/nn/modules/transformer.py:533: UserWarning: "
    "The PyTorch API of nested tensors is in prototype stage and will change in the near future."
)
OTHER = "/x/scripts/foo.py:10: UserWarning: something else"


def _log(tmp_path: Path, stderr: str) -> Path:
    log = tmp_path / "rerun_log.jsonl"
    log.write_text(json.dumps({"command": ["python", "a.py"], "stderr_tail": stderr}) + "\n")
    return log


@pytest.mark.parametrize(
    ("stderr", "audited", "unexpected"), [(NESTED, 1, 0), (OTHER, 0, 1), ("", 0, 0)]
)
def test_warning_audit_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stderr: str, audited: int, unexpected: int
) -> None:
    monkeypatch.setattr(rerun_pipeline, "LOG_PATH", _log(tmp_path, stderr))
    result = rerun_pipeline.audit_logged_warnings()
    assert result["audited_upstream"] == audited
    assert len(result["unexpected"]) == unexpected


def test_subprocess_filter_is_the_exact_audited_warning() -> None:
    action, message, category, module = rerun_pipeline.AUDITED_SUBPROCESS_WARNINGS.split(":")
    assert (action, category, module) == ("ignore", "UserWarning", "torch.nn.modules.transformer")
    assert message.startswith("The PyTorch API of nested tensors")
