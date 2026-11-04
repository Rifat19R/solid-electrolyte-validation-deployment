#!/usr/bin/env python3
"""Rerun the load-bearing analysis programs and compare against the record.

Every program writes only under ``artifacts/rerun/`` and ``reports/rerun/``
(enforced by ``sse_transfer.utils.output_guard``); the authoritative record
is never touched. Stages:

  aggregate  statistics, merges of the recorded per-model fragments, external
             cohort verification and diagnostics (minutes)
  refit      every per-model fit that produced a recorded fragment (hours)
  compare    compare rerun outputs with ``artifacts/authoritative/`` and the
             recorded fragments, separating scientific payload from run
             metadata, and write ``reports/rerun/rerun_comparison.{json,md}``

Run with: python scripts/release/rerun_pipeline.py --stage aggregate|refit|compare
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.models.ladder import MODEL_NAMES  # noqa: E402
from sse_transfer.utils.output_guard import guard_output  # noqa: E402

PY = sys.executable
T = "scripts/transport_analysis/"
A = "scripts/analysis/"
AUTH = ROOT / "artifacts" / "authoritative"
RERUN = ROOT / "artifacts" / "rerun"
FRAGMENTS = ROOT / "artifacts" / "fragments"
LOG_PATH = ROOT / "reports" / "rerun" / "rerun_log.jsonl"
COMPARISON_JSON = ROOT / "reports" / "rerun" / "rerun_comparison.json"
COMPARISON_MD = ROOT / "reports" / "rerun" / "rerun_comparison.md"

AGGREGATE: list[list[str]] = [
    [A + "build_external_cohort.py"],
    [T + "n4b_nested_internal_validation.py", "--merge"],
    [T + "n20b_formula_group_kfold.py", "--merge"],
    [T + "n21_repeated_cv_sensitivity.py", "--merge"],
    [A + "issue04_size_matched_control.py", "--merge"],
    [A + "issue08b_chemical_system_groupkfold.py", "--merge"],
    [A + "issue09b_missing_singleton_groupkfold.py", "--merge"],
    [A + "corrective01_merge_n6.py"],
    [T + "n7b_regime_utility_nested.py"],
    [T + "n8_applicability_domain.py"],
    [T + "n9b_relaxed_sensitivity_nested.py"],
    [T + "n21c_primary_external_context.py"],
    [T + "n21d_ad_benchmark_closure.py"],
    [T + "n15_reviewer_remediation.py"],
    [T + "n20_reviewer_remediation.py"],
    [T + "phase1_0_1_predeclare_and_glass_split.py"],
    [T + "phase1_2_factual_fixes.py"],
    [T + "phase2_0_glass_subset_regime_cis.py"],
    [T + "phase2_1_label_compat_dev_only_full.py"],
    [A + "issue03_common_support_lomfo.py"],
    [A + "issue06_doi_cluster_bootstrap.py"],
    [A + "issue08b_regime_utility.py"],
    [A + "issue09a_missing_family_excluded.py"],
    [A + "issue09b_regime_utility.py"],
    [A + "corrective01_issue10_formula_balanced_sensitivity.py"],
    [T + "n18_nn_distance_forensic.py"],
    [T + "n21e_robust_shift_distance.py"],
    [A + "issue05_fold_local_shift_distance.py"],
]

N6_SKLEARN_AND_FROZEN_TORCH = [m for m in MODEL_NAMES if m != "pretrained_element_embedding"]
REFIT: list[list[str]] = (
    [[T + "n6_external_transfer.py", "--model", m] for m in N6_SKLEARN_AND_FROZEN_TORCH]
    + [[A + "corrective01_run_n6_pretrained_element_embedding.py"]]
    + [[T + "n4b_nested_internal_validation.py", "--model", m] for m in MODEL_NAMES]
    + [[T + "n20b_formula_group_kfold.py", "--model", m] for m in MODEL_NAMES]
    + [[A + "issue08b_chemical_system_groupkfold.py", "--model", m] for m in MODEL_NAMES]
    + [[A + "issue09b_missing_singleton_groupkfold.py", "--model", m] for m in MODEL_NAMES]
    + [
        [A + "issue04_size_matched_control.py", "--model", m, "--seed", str(s)]
        for s in (20260930, 20260931, 20260932)
        for m in MODEL_NAMES
    ]
    + [
        [T + "n21_repeated_cv_sensitivity.py", "--seed", str(s), "--model", m]
        for s in range(20260921, 20260931)
        for m in MODEL_NAMES
    ]
    + [[A + "corrective01_retune_pretrained_element_embedding.py"]]
)

# Run metadata that legitimately differs between executions or was rewritten
# by the path migration; reported separately, never silently dropped.
VOLATILE_KEYS = {
    "git_commit",
    "git_branch",
    "runtime_seconds",
    "total_runtime_seconds",
    "runtime_seconds_total",
    "elapsed_seconds",
    "created_at_utc",
    "generated_at_utc",
    "timestamp_utc",
    # annotation and bootstrap bookkeeping fields, never statistics
    "corrective_note",
    "n_finite",
}
PATH_LABEL_SUFFIXES = ("_source", "_path", "_file", "source", "spec_file")


def _env() -> dict[str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    return {"python": platform.python_version(), "platform": platform.platform(), "git_head": head}


# Exact audited upstream warning (same message/category/module as the pytest
# policy in pyproject.toml): torch.nn.TransformerEncoder's nested-tensor fast
# path notice, emitted by the composition-attention model. Nothing broader.
AUDITED_SUBPROCESS_WARNINGS = (
    "ignore:The PyTorch API of nested tensors is in prototype stage:UserWarning:"
    "torch.nn.modules.transformer"
)


def run_commands(commands: list[list[str]]) -> int:
    failures = 0
    env = _env()
    child_env = dict(os.environ, PYTHONWARNINGS=AUDITED_SUBPROCESS_WARNINGS)
    for cmd in commands:
        t0 = time.time()
        proc = subprocess.run(
            [PY, *cmd], cwd=ROOT, capture_output=True, text=True, check=False, env=child_env
        )
        record = {
            "command": ["python", *cmd],
            "returncode": proc.returncode,
            "seconds": round(time.time() - t0, 1),
            "stdout_tail": proc.stdout[-1500:],
            "stderr_tail": proc.stderr[-3000:],
            **env,
        }
        with guard_output(LOG_PATH).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        print(f"[{status}] {' '.join(cmd)} ({record['seconds']}s)", flush=True)
        failures += proc.returncode != 0
    return failures


WARNING_LINE = re.compile(r"(?P<file>\S+\.py):\d+: (?P<category>\w+Warning): (?P<message>.*)")


def _audited_warning_filters() -> list[tuple[re.Pattern[str], str, re.Pattern[str]]]:
    """The exact `ignore:` entries of the pytest fail-closed warning policy."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    filters = []
    for entry in config["tool"]["pytest"]["ini_options"]["filterwarnings"]:
        if entry.startswith("ignore:"):
            _, message, category, module = entry.split(":", 3)
            filters.append((re.compile(message), category, re.compile(module)))
    return filters


def _module_name(path: str) -> str:
    rel = path.split("site-packages/")[-1]
    return rel.removesuffix(".py").replace("/", ".")


def audit_logged_warnings() -> dict[str, Any]:
    """Classify every warning in the latest log record of each command."""
    latest: dict[str, dict[str, Any]] = {}
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        latest[" ".join(record["command"])] = record
    filters = _audited_warning_filters()
    audited, unexpected = 0, []
    for command, record in latest.items():
        for line in record["stderr_tail"].splitlines():
            m = WARNING_LINE.search(line)
            if not m:
                continue
            module = _module_name(m.group("file"))
            known = any(
                cat == m.group("category") and mod.match(module) and msg.match(m.group("message"))
                for msg, cat, mod in filters
            )
            if known:
                audited += 1
            else:
                unexpected.append(f"{command}: {line.strip()[:160]}")
    return {"commands": len(latest), "audited_upstream": audited, "unexpected": unexpected}


def _is_path_label(key: str) -> bool:
    # *_sha256 fields hash input files whose bytes also carry run metadata.
    return (
        key.endswith((*PATH_LABEL_SUFFIXES, "_sha256"))
        or key.startswith("git_commit")
        or key in {"lock_file", "spec_path", "phase"}
    )


def compare_payload(
    rec: Any, new: Any, path: str = "", out: dict[str, list[Any]] | None = None
) -> dict[str, list[Any]]:
    out = out if out is not None else {"scientific": [], "metadata": [], "max_abs_diff": [0.0]}
    if isinstance(rec, dict) and isinstance(new, dict):
        for key in sorted(set(rec) | set(new)):
            sub = f"{path}/{key}"
            if key in VOLATILE_KEYS or _is_path_label(key):
                if rec.get(key) != new.get(key):
                    out["metadata"].append(sub)
                continue
            if key not in rec or key not in new:
                out["scientific"].append(
                    f"{sub}: present only in {'rerun' if key in new else 'record'}"
                )
                continue
            compare_payload(rec[key], new[key], sub, out)
    elif isinstance(rec, list) and isinstance(new, list):
        if len(rec) != len(new):
            out["scientific"].append(f"{path}: length {len(rec)} vs {len(new)}")
        for i, (a, b) in enumerate(zip(rec, new, strict=False)):
            compare_payload(a, b, f"{path}[{i}]", out)
    elif isinstance(rec, bool) or isinstance(new, bool):
        if rec != new:
            out["scientific"].append(f"{path}: {rec!r} vs {new!r}")
    elif isinstance(rec, (int, float)) and isinstance(new, (int, float)):
        if not (math.isnan(rec) and math.isnan(new)) and rec != new:
            diff = abs(float(rec) - float(new))
            out["max_abs_diff"][0] = max(out["max_abs_diff"][0], diff)
            if diff > 1e-9 * max(1.0, abs(float(rec))):
                out["scientific"].append(f"{path}: {rec!r} vs {new!r}")
    elif rec != new:
        if isinstance(rec, str) and isinstance(new, str) and ("/" in rec or "\\" in rec):
            out["metadata"].append(path)
        else:
            out["scientific"].append(f"{path}: {str(rec)[:80]!r} vs {str(new)[:80]!r}")
    return out


def _pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for rec in sorted(AUTH.rglob("*.json")):
        new = RERUN / rec.relative_to(AUTH)
        pairs.append((rec, new))
    for new in sorted((RERUN / "fragments").rglob("*.json")):
        pairs.append((FRAGMENTS / new.relative_to(RERUN / "fragments"), new))
    return pairs


def compare() -> int:
    rows = []
    for rec, new in _pairs():
        row: dict[str, Any] = {
            "record": rec.relative_to(ROOT).as_posix(),
            "rerun": new.relative_to(ROOT).as_posix(),
        }
        if not new.exists():
            row["status"] = "NOT_RERUN"
        elif not rec.exists():
            row["status"] = "NO_RECORD"
        else:
            diff = compare_payload(
                json.loads(rec.read_text(encoding="utf-8")),
                json.loads(new.read_text(encoding="utf-8")),
            )
            row["scientific_differences"] = diff["scientific"][:25]
            row["n_scientific_differences"] = len(diff["scientific"])
            row["metadata_differences"] = diff["metadata"][:25]
            row["max_abs_numeric_diff"] = diff["max_abs_diff"][0]
            if diff["scientific"]:
                row["status"] = "DRIFT"
            elif diff["max_abs_diff"][0] > 0:
                row["status"] = "EQUAL_WITHIN_1e-9"
            else:
                row["status"] = "EXACT"
        rows.append(row)
    guard_output(COMPARISON_JSON).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines = [
        "# Rerun comparison\n",
        f"Status counts: {counts}\n",
        "| record | status | max abs diff | metadata-only differences |",
        "|---|---|---:|---|",
    ]
    for r in rows:
        meta = ", ".join(r.get("metadata_differences", [])[:4])
        lines.append(
            f"| `{r['record']}` | {r['status']} | {r.get('max_abs_numeric_diff', '')} | {meta} |"
        )
    drift = [r for r in rows if r["status"] == "DRIFT"]
    for r in drift:
        lines.append(f"\n## DRIFT: {r['record']}\n")
        lines += [f"- {d}" for d in r["scientific_differences"]]
    guard_output(COMPARISON_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(counts)
    warnings_audit = audit_logged_warnings()
    lines = [
        "\n## Warnings in rerun logs\n",
        f"Latest records of {warnings_audit['commands']} commands; "
        f"{warnings_audit['audited_upstream']} warnings match the exact audited upstream "
        f"policy (pyproject.toml); {len(warnings_audit['unexpected'])} unexpected.",
        *[f"- UNEXPECTED: {w}" for w in warnings_audit["unexpected"]],
    ]
    with guard_output(COMPARISON_MD).open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    n_unexpected = len(warnings_audit["unexpected"])
    print(f"warnings: {warnings_audit['audited_upstream']} audited, {n_unexpected} unexpected")
    return len(drift) + n_unexpected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["aggregate", "refit", "compare"], required=True)
    args = parser.parse_args()
    if args.stage == "aggregate":
        raise SystemExit(run_commands(AGGREGATE))
    if args.stage == "refit":
        raise SystemExit(run_commands(REFIT))
    raise SystemExit(compare())


if __name__ == "__main__":
    main()
