"""Write-path guard for analysis reruns.

Recorded results are immutable: the frozen protocol inputs and primary
one-shot external-transfer record (``artifacts/frozen``), the current
results (``artifacts/authoritative``), the superseded results kept for the
record (``artifacts/historical``), the per-model fragments those results were
merged from (``artifacts/fragments``), the frozen external cohorts, and the
phase reports. Every rerun writes under ``artifacts/rerun`` or
``reports/rerun`` instead, where it can be compared against the record.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RERUN_ARTIFACTS = REPO_ROOT / "artifacts" / "rerun"
RERUN_REPORTS = REPO_ROOT / "reports" / "rerun"

PROTECTED_DIRS: tuple[Path, ...] = tuple(
    REPO_ROOT / rel
    for rel in (
        "artifacts/frozen",
        "artifacts/authoritative",
        "artifacts/historical",
        "artifacts/fragments",
        "data/raw",
        "data/processed/external_cohort",
        "data/processed/v1",
        "data/manifests",
        "manuscript/reference_submission",
        "reports/cross_dataset_study",
        "reports/remediation",
        "reports/history",
    )
)


class ProtectedOutputError(RuntimeError):
    """Raised when code tries to write into a protected record location."""


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def is_protected(path: Path) -> bool:
    resolved = Path(path).resolve()
    return any(_is_within(resolved, protected.resolve()) for protected in PROTECTED_DIRS)


def guard_output(path: Path) -> Path:
    """Return ``path`` (parent directory created) if it is safe to write, else raise.

    Call this on every artifact/report path immediately before writing it.
    """
    if is_protected(path):
        raise ProtectedOutputError(
            f"refusing to write {path}: it is inside a protected record location. "
            "Reruns must write under artifacts/rerun/ or reports/rerun/."
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return Path(path)
