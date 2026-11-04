#!/usr/bin/env python3
"""Build the curated, deterministic reproducibility ZIP.

Stages an explicit allowlist of repository paths into
`release/build/SSE_Reproducibility/`, writes a clean pinned environment
(`environment/requirements-lock.txt`, no machine paths), `VERSION.txt` and
`MANIFEST_SHA256.txt`, enforces the exclusion rules (fail-closed), and writes
`release/SSE_Reproducibility_Submission.zip` (sorted entries, fixed timestamps)
plus its `.sha256` file.

Archival builds are tied to one commit and fail unless the working tree is
clean (`git status --porcelain` empty). Every staged file must be either
tracked at HEAD or one of the git-ignored inputs (raw OBELiX, processed
OBELiX split, per-model fragments) pinned by SHA-256 in the tracked record
`provenance/untracked_inputs.json`. Hargreaves-derived files are withheld or
redacted as recorded in `provenance/restricted_data.json`
(scripts/release/restricted_data.py).

Run with: python scripts/release/build_package.py
          python scripts/release/build_package.py --allow-dirty   (development
              build into release/build-dev/; never an archival deposit)
          python scripts/release/build_package.py --write-input-record
"""

from __future__ import annotations

import getpass
import hashlib
import json
import re
import shutil
import socket
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import restricted_data  # noqa: E402

RELEASE = ROOT / "release"
STAGE = RELEASE / "build" / "SSE_Reproducibility"
ZIP_PATH = RELEASE / "SSE_Reproducibility_Submission.zip"
INPUT_RECORD = ROOT / "provenance" / "untracked_inputs.json"
# Git-ignored directories whose files are shipped; each file is hash-pinned
# in INPUT_RECORD (data policy keeps them out of git).
UNTRACKED_INPUT_DIRS = ("data/raw/obelix", "data/processed/v1", "artifacts/fragments")

# Directories copied recursively (repo-relative) and single files.
INCLUDE_DIRS = [
    "src",
    "scripts/transport_analysis",
    "scripts/analysis",
    "scripts/figures",
    "scripts/manuscript",
    "scripts/release",
    "tests",
    "configs",
    "data/manifests",
    "data/raw/obelix",
    "data/processed/v1",
    "artifacts/frozen",
    "artifacts/authoritative",
    "artifacts/fragments",
    "manuscript/source",
    "manuscript/templates",
    "manuscript/figures",
    "manuscript/tables",
    "submission",
    "provenance",
    "reports/remediation/specs",
    "reports/results/leakage",
]
INCLUDE_FILES = {
    "release/README.md": "README.md",
    "release/TEST_COVERAGE_EXCEPTIONS.md": "TEST_COVERAGE_EXCEPTIONS.md",
    "LICENSE": "LICENSE",
    "CITATION.cff": "CITATION.cff",
    "RIGHTS_AND_LICENSING.md": "RIGHTS_AND_LICENSING.md",
    "RESEARCH_CONTRACT.md": "RESEARCH_CONTRACT.md",
    "DATA_CONTRACT.md": "DATA_CONTRACT.md",
    "EXPERIMENT_PROTOCOL.md": "EXPERIMENT_PROTOCOL.md",
    "DEFINITION_OF_DONE.md": "DEFINITION_OF_DONE.md",
    "CHANGELOG.md": "CHANGELOG.md",
    "pyproject.toml": "pyproject.toml",
    "Makefile": "Makefile",
    "data/RECONSTRUCTION.md": "data/RECONSTRUCTION.md",
    "scripts/preflight.py": "scripts/preflight.py",
    "scripts/fetch_raw_data.py": "scripts/fetch_raw_data.py",
    "scripts/build_processed_v1.py": "scripts/build_processed_v1.py",
    "scripts/freeze_split.py": "scripts/freeze_split.py",
    "scripts/hash_file.py": "scripts/hash_file.py",
    "scripts/manuscript_figures/palette.py": "scripts/manuscript_figures/palette.py",
    "reports/rerun/rerun_comparison.md": "reports/rerun/rerun_comparison.md",
    "reports/rerun/rerun_comparison.json": "reports/rerun/rerun_comparison.json",
}
# Tests that are not shipped: the closed earlier project's tests, the
# development editor-hook tests, and repository-only record checks.
EXCLUDE_PREFIXES = ("tests/closed_project/", "tests/dev_tooling/", "tests/repository/")
# Modules of the closed earlier project that no shipped script, module or test
# imports (checked by tests/unit/test_package_test_policy.py).
CLOSED_PROJECT_SRC = tuple(
    f"src/sse_transfer/{m}.py"
    for m in (
        "audit/cif_rounding",
        "audit/leakage",
        "constants",
        "data/canonical_id",
        "data/inspect_datasets",
        "data/structure_flag_reconciliation",
        "data/structure_recovery",
        "evaluation/ensembling",
        "features/preprocessing",
        "features/pretrained_structure_embeddings",
        "features/structure_features",
        "features/transport_descriptors",
        "training/censored_loss",
        "uncertainty/dispersion",
    )
)
# Files that define the machine-path detector pattern itself.
PATTERN_DEFINITIONS = {"scripts/release/build_package.py", "tests/unit/test_portable_provenance.py"}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "contact_sheets"}
SKIP_SUFFIXES = {".pyc", ".log", ".out", ".tmp"}

# Fail-closed exclusion rules applied to the staged tree.
FORBIDDEN_PATH = re.compile(
    r"npj|acs|jcim|turnitin|quarantine|historical|pre_fix|LiIonDatabase|overlap_ledger|\.env$|\.venv|\.git/",
    re.I,
)
MACHINE_PATH = re.compile(r"/mnt/[a-z]/|/home/[^/\s\"']+/|[A-Za-z]:\\\\(?:Users|Rifat)")
TEXT_SUFFIXES = {".py", ".md", ".json", ".csv", ".txt", ".toml", ".cff", ".yaml", ".yml", ".jsonl"}
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def in_package_scope(rel: str) -> bool:
    """Whether a repo-relative path is one `stage()` copies into the package
    (for INCLUDE_FILES, the destination path) and `check_exclusions()` allows.
    Redacted files are in scope (shipped with redacted fields)."""
    parts = set(Path(rel).parts)
    if FORBIDDEN_PATH.search(rel) or rel in restricted_data.WITHHELD:
        return False
    if rel.startswith(EXCLUDE_PREFIXES) or rel in CLOSED_PROJECT_SRC:
        return False
    if any(part.endswith(".egg-info") for part in parts):
        return False
    if rel in INCLUDE_FILES.values() or rel in {"VERSION.txt", "MANIFEST_SHA256.txt"}:
        return True
    if rel.startswith("environment/"):
        return rel in {"environment/requirements-lock.txt", "environment/python-version.txt"}
    return (
        any(rel.startswith(f"{d}/") for d in INCLUDE_DIRS)
        and not parts & SKIP_PARTS
        and Path(rel).suffix not in SKIP_SUFFIXES
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout


def selected_sources() -> dict[str, str]:
    """Repo-relative source path -> package-relative destination path."""
    selected: dict[str, str] = {}
    for rel in INCLUDE_DIRS:
        src = ROOT / rel
        if not src.is_dir():
            raise SystemExit(f"missing include dir {rel}")
        for f in sorted(src.rglob("*")):
            r = f.relative_to(ROOT).as_posix()
            if f.is_file() and in_package_scope(r):
                selected[r] = r
    for src_rel, dst_rel in INCLUDE_FILES.items():
        if not (ROOT / src_rel).is_file():
            raise SystemExit(f"missing include file {src_rel}")
        selected[src_rel] = dst_rel
    return selected


def untracked_inputs(selected: dict[str, str]) -> dict[str, str]:
    tracked = set(git("ls-files", "-z").split("\0"))
    return {
        src: sha256(ROOT / src)
        for src in sorted(selected)
        if src not in tracked and src.startswith(UNTRACKED_INPUT_DIRS)
    }


def check_provenance(selected: dict[str, str]) -> list[str]:
    """Every staged source is tracked at HEAD (clean tree) or hash-pinned."""
    tracked = set(git("ls-files", "-z").split("\0"))
    pinned = json.loads(INPUT_RECORD.read_text("utf-8"))["files"]
    errors = []
    for src in sorted(selected):
        if src in tracked:
            continue
        if src not in pinned:
            errors.append(f"untracked and not hash-pinned: {src}")
        elif sha256(ROOT / src) != pinned[src]:
            errors.append(f"hash differs from {INPUT_RECORD.name}: {src}")
    stale = sorted(set(pinned) - set(selected))
    errors += [f"pinned input no longer shipped: {src}" for src in stale]
    return errors


def stage(selected: dict[str, str]) -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    record = restricted_data.load_record(ROOT)
    for src_rel, dst_rel in selected.items():
        dst = STAGE / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src_rel in record["redacted"]:
            payload = restricted_data.redact((ROOT / src_rel).read_bytes())
            expected = record["redacted"][src_rel]["redacted_sha256"]
            if restricted_data.sha256_bytes(payload) != expected:
                raise SystemExit(
                    f"redacted {src_rel} does not match provenance/restricted_data.json"
                )
            dst.write_bytes(payload)
        else:
            shutil.copy2(ROOT / src_rel, dst)
    missing = set(record["redacted"]) - set(selected)
    if missing:
        raise SystemExit(f"redacted files not staged: {sorted(missing)}")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    lock = [
        ln
        for ln in freeze.splitlines()
        if ln and not ln.startswith(("-e ", "#")) and "sse-transport-transfer" not in ln
    ]
    env = STAGE / "environment"
    env.mkdir(parents=True, exist_ok=True)
    (env / "requirements-lock.txt").write_text(
        "\n".join(sorted(lock, key=str.lower)) + "\n", encoding="utf-8"
    )
    (env / "python-version.txt").write_text(
        f"{sys.version_info.major}.{sys.version_info.minor}\n", encoding="utf-8"
    )
    head = git("rev-parse", "HEAD").strip()
    dirty = bool(git("status", "--porcelain").strip())
    (STAGE / "VERSION.txt").write_text(
        "SSE reproducibility package\n"
        f"source commit: {head}"
        f"{' (DEVELOPMENT BUILD from an uncommitted tree; not archival)' if dirty else ''}\n"
        "Hargreaves-derived files: withheld or redacted (provenance/restricted_data.json)\n"
        "public repository URL and archival DOI: added on deposit (not yet created)\n",
        encoding="utf-8",
    )


def local_identity_pattern() -> re.Pattern[str]:
    """The local user name and host name (they must not leak into the package)."""
    names = {n for n in (getpass.getuser(), socket.gethostname().split(".")[0]) if len(n) >= 4}
    return re.compile("|".join(rf"\b{re.escape(n)}\b" for n in sorted(names)) or r"(?!)", re.I)


def check_exclusions() -> list[str]:
    errors = []
    identity = local_identity_pattern()
    for f in sorted(STAGE.rglob("*")):
        rel = f.relative_to(STAGE).as_posix()
        if FORBIDDEN_PATH.search(rel) or rel in restricted_data.WITHHELD:
            errors.append(f"forbidden path in package: {rel}")
        if rel.startswith(EXCLUDE_PREFIXES) or rel in CLOSED_PROJECT_SRC:
            errors.append(f"excluded path in package: {rel}")
        if f.is_file() and f.suffix in TEXT_SUFFIXES and rel not in PATTERN_DEFINITIONS:
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if MACHINE_PATH.search(text):
                errors.append(f"machine path in {rel}")
            if identity.search(text):
                errors.append(f"local user/host name in {rel}")
    return errors


def write_manifest() -> None:
    lines = [
        f"{sha256(f)}  {f.relative_to(STAGE).as_posix()}"
        for f in sorted(STAGE.rglob("*"))
        if f.is_file() and f.name != "MANIFEST_SHA256.txt"
    ]
    (STAGE / "MANIFEST_SHA256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_zip() -> str:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file():
                info = zipfile.ZipInfo(
                    f"SSE_Reproducibility/{f.relative_to(STAGE).as_posix()}", FIXED_TIME
                )
                info.external_attr = 0o644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, f.read_bytes())
    digest = sha256(ZIP_PATH)
    (ZIP_PATH.parent / f"{ZIP_PATH.name}.sha256").write_text(
        f"{digest}  {ZIP_PATH.name}\n", encoding="utf-8"
    )
    return digest


def main(argv: list[str]) -> None:
    global STAGE, ZIP_PATH
    selected = selected_sources()
    if "--write-input-record" in argv:
        INPUT_RECORD.write_text(
            json.dumps(
                {
                    "description": (
                        "SHA-256 of every git-ignored input file the reproducibility "
                        "package ships (scripts/release/build_package.py)."
                    ),
                    "files": untracked_inputs(selected),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {INPUT_RECORD.relative_to(ROOT)}")
        return
    dirty = git("status", "--porcelain").strip()
    if dirty and "--allow-dirty" not in argv:
        print(dirty)
        raise SystemExit(
            "FAIL working tree is not clean; an archival build must come from a "
            "committed HEAD (use --allow-dirty only for a development build)"
        )
    if "--allow-dirty" in argv:
        STAGE = RELEASE / "build-dev" / "SSE_Reproducibility"
        ZIP_PATH = RELEASE / "build-dev" / "SSE_Reproducibility_DEV.zip"
        ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    errors = check_provenance(selected)
    if dirty:
        # Development build: report, do not enforce (the tree is not a commit).
        print(f"development build: {len(errors)} provenance findings not enforced")
    else:
        for e in errors:
            print("FAIL", e)
        if errors:
            raise SystemExit(1)
    stage(selected)
    errors = check_exclusions()
    for e in errors:
        print("FAIL", e)
    if errors:
        raise SystemExit(1)
    write_manifest()
    digest = write_zip()
    n = sum(1 for f in STAGE.rglob("*") if f.is_file())
    size = ZIP_PATH.stat().st_size
    print(f"staged {n} files; {ZIP_PATH.relative_to(ROOT)} sha256 {digest} ({size:,} bytes)")


if __name__ == "__main__":
    main(sys.argv[1:])
