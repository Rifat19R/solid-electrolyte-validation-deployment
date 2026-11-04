"""The package test runner's exclusions must be complete, current and documented,
and the package must not depend on anything it deliberately does not ship."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "release" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exceptions_doc() -> str:
    # release/ in the repository; the package root in the extracted package.
    for path in (
        ROOT / "release" / "TEST_COVERAGE_EXCEPTIONS.md",
        ROOT / "TEST_COVERAGE_EXCEPTIONS.md",
    ):
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise AssertionError("TEST_COVERAGE_EXCEPTIONS.md not found")


def test_runner_has_no_module_ignores_or_expected_skips() -> None:
    runner = _load("run_package_tests")
    assert not hasattr(runner, "IGNORED_FILES")
    assert not hasattr(runner, "EXPECTED_SKIPS")
    source = (ROOT / "scripts" / "release" / "run_package_tests.py").read_text("utf-8")
    assert '"--ignore"' not in source and "'--ignore'" not in source


def test_every_exclusion_target_exists() -> None:
    assert _load("run_package_tests").stale_targets() == []


def test_every_exclusion_is_restricted_data_with_reason_and_documented() -> None:
    runner = _load("run_package_tests")
    doc = _exceptions_doc()
    assert runner.RESTRICTED, "expected the documented restricted-data tests"
    for nodeid, reason in runner.RESTRICTED.items():
        assert "Hargreaves" in reason, nodeid
        assert f"`{nodeid}`" in doc, nodeid


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text("utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_no_shipped_python_file_imports_an_unshipped_closed_project_module(
    current_files: list[str],
) -> None:
    builder = _load("build_package")
    closed = {
        rel.removeprefix("src/").removesuffix(".py").replace("/", ".")
        for rel in builder.CLOSED_PROJECT_SRC
    }
    offenders = []
    for rel in current_files:
        if rel.endswith(".py") and builder.in_package_scope(rel) and (ROOT / rel).is_file():
            hit = _imported_modules(ROOT / rel) & closed
            if hit:
                offenders.append((rel, sorted(hit)))
    assert offenders == []


def test_unshipped_test_directories_hold_only_their_declared_kind() -> None:
    builder = _load("build_package")
    assert builder.EXCLUDE_PREFIXES == (
        "tests/closed_project/",
        "tests/dev_tooling/",
        "tests/repository/",
    )
    for rel in builder.CLOSED_PROJECT_SRC:
        assert not builder.in_package_scope(rel), rel
