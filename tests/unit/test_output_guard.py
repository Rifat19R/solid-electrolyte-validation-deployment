"""Write-path guard: recorded results are immutable; reruns go to rerun/."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sse_transfer.utils.output_guard import (
    PROTECTED_DIRS,
    REPO_ROOT,
    ProtectedOutputError,
    guard_output,
    is_protected,
)

PIPELINE_DIRS = [REPO_ROOT / "scripts" / "transport_analysis", REPO_ROOT / "scripts" / "analysis"]
WRITE_METHODS = {"write_text", "write_bytes", "to_csv", "to_json", "savefig"}
# Reconstructs the frozen cohorts only when absent, after verifying the lock
# hash, and never overwrites (covered by test_build_external_cohort.py).
HASH_VERIFIED_WRITERS = {"build_external_cohort.py"}


@pytest.mark.parametrize(
    "protected", PROTECTED_DIRS, ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_every_protected_location_is_refused(protected: Path) -> None:
    with pytest.raises(ProtectedOutputError):
        guard_output(protected / "nested" / "file.json")


def test_rerun_location_is_allowed(tmp_path: Path) -> None:
    assert not is_protected(REPO_ROOT / "artifacts" / "rerun" / "x.json")
    target = tmp_path / "a" / "b.json"
    assert guard_output(target) == target
    assert target.parent.is_dir()


def test_relative_traversal_cannot_escape_into_protected() -> None:
    sneaky = REPO_ROOT / "artifacts" / "rerun" / ".." / "authoritative" / "x.json"
    with pytest.raises(ProtectedOutputError):
        guard_output(sneaky)


def _const_path(node: ast.expr, consts: dict[str, str]) -> str | None:
    """Evaluate `ROOT / "a" / "b"`-style module constants to a repo-relative path."""
    if isinstance(node, ast.Name):
        return {"ROOT": "", "REPO_ROOT": ""}.get(node.id, consts.get(node.id))
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = _const_path(node.left, consts), _const_path(node.right, consts)
        if left is None or right is None:
            return None
        return f"{left}/{right}".strip("/")
    return None


def _pipeline_scripts() -> list[Path]:
    return sorted(p for d in PIPELINE_DIRS for p in d.glob("*.py"))


@pytest.mark.parametrize("script", _pipeline_scripts(), ids=lambda p: p.name)
def test_every_pipeline_write_is_guarded(script: Path) -> None:
    if script.name in HASH_VERIFIED_WRITERS:
        return
    tree = ast.parse(script.read_text(encoding="utf-8"))
    unguarded = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr in WRITE_METHODS:
            target = node.func.value if node.func.attr != "to_csv" else None
            if node.func.attr in {"to_csv", "to_json"}:
                target = node.args[0] if node.args else None
            guarded = (
                isinstance(target, ast.Call)
                and isinstance(target.func, ast.Name)
                and target.func.id == "guard_output"
            )
            if not guarded:
                unguarded.append(node.lineno)
        if node.func.attr == "dump" and isinstance(node.func.value, ast.Name):
            unguarded.append(node.lineno)  # json.dump to an open handle bypasses the guard
    assert unguarded == [], f"{script.name}: unguarded writes at lines {unguarded}"


@pytest.mark.parametrize("script", _pipeline_scripts(), ids=lambda p: p.name)
def test_default_output_constants_are_not_protected(script: Path) -> None:
    tree = ast.parse(script.read_text(encoding="utf-8"))
    consts: dict[str, str] = {}
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                value = _const_path(node.value, consts)
                if value is not None:
                    consts[target.id] = value
                    name = target.id
                    is_output = name.startswith(("OUTPUT", "OUT_", "REPORT", "RERUN"))
                    is_output |= name in {"ARTIFACT_JSON", "FRAGMENT_OUT_DIR", "TABLES_DIR"}
                    is_output |= name.startswith("ARTIFACT_") and len(name) == len("ARTIFACT_A")
                    if is_output and is_protected(REPO_ROOT / value):
                        offenders.append(f"{name}={value}")
    assert offenders == []
