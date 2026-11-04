#!/usr/bin/env python3
"""Build the main-text tables from authoritative artifacts and verify the manuscript.

Writes `manuscript/tables/table{1..4}_*.csv` (full precision) and then checks
that every number in the corresponding markdown table of
`manuscript/source/00_main.md` equals the CSV value at the manuscript's
displayed precision. Any mismatch, missing row, or missing table fails the
run (exit 1), so a stale or hand-edited table cannot reach the DOCX build.

Sources (all under artifacts/authoritative/ unless stated):
  Table 1  data/manifests/external_cohort.json (stage counts),
           data/manifests/hargreaves_conductivity.json (820 rows, 214 sources),
           data/processed/v1/obelix_index.json + obelix_target_primary_exact.json
           (449 development / 113 test exact-label rows),
           n15_artifacts/n15_source_disjoint.json (153 / 51)
  Table 2  n7b_regime_utility_nested_corrected.json (A-C, eight models),
           n15_artifacts/n15_stable_model_sensitivity.json (A-C, six models),
           phase1_factual_fixes_corrected.json:item3_formula_gkf_utility (D)
  Table 3  n7b_regime_utility_nested_corrected.json:pairwise_bootstrap_ci (B, C),
           phase1_factual_fixes_corrected.json:item4_pairwise_diffs.formulaGKF_minus_A (D)
  Table 4  phase1_glass_split_corrected.json:per_model_subset_metrics

Run with: python scripts/manuscript/build_tables.py [--check-only]
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from sse_transfer.utils.manuscript_text import plain_text  # noqa: E402

AUTH = ROOT / "artifacts" / "authoritative"
TABLES = ROOT / "manuscript" / "tables"
MAIN_MD = ROOT / "manuscript" / "source" / "00_main.md"
SI_MD = ROOT / "manuscript" / "source" / "01_si.md"
SI_TABLE1_CSV = TABLES / "si_table1_robustness_summary.csv"

MODEL_LABELS = {
    "extra_trees": "Extremely randomized trees",
    "catboost": "CatBoost",
    "random_forest": "Random forest",
    "composition_attention": "Composition attention",
    "xgboost": "XGBoost",
    "shallow_mlp": "Shallow MLP",
    "ridge": "Ridge",
    "pretrained_element_embedding": "Pooled elemental-property model",
}
STATS = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]
STAT_LABELS = {
    "median_abs_mae_gap": "Median |gap|",
    "spearman_rho": "Spearman ρ",
    "kendall_tau": "Kendall τ",
    "top_model_selection_regret": "Regret",
}
REGIMES = {
    "A_random_5fold_cv": "A: random five-fold CV",
    "B_family_group_kfold": "B: Family-GroupKFold",
    "C_lomfo": "C: LOMFO",
}
D_LABEL = "D: formula-GroupKFold (post hoc control)"


def _load(rel: str, base: Path = AUTH) -> Any:
    return json.loads((base / rel).read_text(encoding="utf-8"))


def table1() -> list[dict[str, Any]]:
    stage = _load("data/manifests/external_cohort.json", ROOT)["stage_counts"]
    harg = _load("data/manifests/hargreaves_conductivity.json", ROOT)
    index = _load("data/processed/v1/obelix_index.json", ROOT)
    exact = _load("data/processed/v1/obelix_target_primary_exact.json", ROOT)
    split = {i: name for name, ids in index["splits"].items() for i in ids}
    counts = collections.Counter(split[row["id"]] for row in exact["rows"])
    sd = _load("n15_artifacts/n15_source_disjoint.json")
    n_sources = int(re.search(r"from (\d+) sources", harg["reported_count_per_paper"]).group(1))
    return [
        {"cohort": "OBELiX development population", "n": counts["train"] + counts["validation"]},
        {"cohort": "OBELiX official test split", "n": counts["test"]},
        {"cohort": "Hargreaves et al. database", "n": harg["parsed_count"], "sources": n_sources},
        {"cohort": "Room-temperature window (15–35 °C)", "n": stage["rt_window_rows"]},
        {"cohort": "Strict external cohort (primary)", "n": stage["strict_independent_rows"]},
        {
            "cohort": "DOI-source-disjoint subset of strict cohort",
            "n": sd["n_source_disjoint"],
            "strict_rows_removed": sd["n_excluded_for_source_overlap"],
        },
        {"cohort": "Relaxed external cohort", "n": stage["relaxed_independent_rows"]},
    ]


def table2() -> list[dict[str, Any]]:
    n7b = _load("n7b_regime_utility_nested_corrected.json")["per_regime_summary"]
    six = _load("n15_artifacts/n15_stable_model_sensitivity.json")["per_regime_summary"]
    d = _load("phase1_factual_fixes_corrected.json")["item3_formula_gkf_utility"]
    rows: list[dict[str, Any]] = []
    for model_set, source, d_key in (
        ("Eight models", n7b, "primary_8_model"),
        ("Six models", six, "six_model_stable_sensitivity"),
    ):
        for key, label in REGIMES.items():
            rows.append(
                {"regime": label, "model_set": model_set, **{s: source[key][s] for s in STATS}}
            )
        rows.append({"regime": D_LABEL, "model_set": model_set, **{s: d[d_key][s] for s in STATS}})
    return rows


def table3() -> list[dict[str, Any]]:
    pairs = _load("n7b_regime_utility_nested_corrected.json")["pairwise_bootstrap_ci"]
    d = _load("phase1_factual_fixes_corrected.json")["item4_pairwise_diffs"]["formulaGKF_minus_A"]
    rows = []
    for label, source in (
        ("B − A", pairs["B_minus_A"]),
        ("C − A", pairs["C_minus_A"]),
        ("D − A", d),
    ):
        for stat in STATS:
            ci = source[stat]
            excludes = ci["ci_low"] > 0 or ci["ci_high"] < 0
            rows.append(
                {
                    "comparison": label,
                    "statistic": STAT_LABELS[stat],
                    "point_delta": ci["point_delta"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "ci_excludes_zero": "Yes" if excludes else "No",
                }
            )
    return rows


def table4() -> list[dict[str, Any]]:
    metrics = _load("phase1_glass_split_corrected.json")["per_model_subset_metrics"]
    rows = []
    for model, label in MODEL_LABELS.items():
        seen, glass = metrics["SEEN"][model], metrics["UNSEEN_GLASS"][model]
        crys = metrics["UNSEEN_CRYSTALLINE"][model]
        rows.append(
            {
                "model": label,
                "seen_n": seen["n"],
                "seen_mae": seen["mae"],
                "seen_ci_low": seen["mae_bootstrap_ci"]["ci_low"],
                "seen_ci_high": seen["mae_bootstrap_ci"]["ci_high"],
                "glass_n": glass["n"],
                "glass_mae": glass["mae"],
                "glass_ci_low": glass["mae_bootstrap_ci"]["ci_low"],
                "glass_ci_high": glass["mae_bootstrap_ci"]["ci_high"],
                "crystalline_n": crys["n"],
                "crystalline_mae_descriptive": crys["mae"],
            }
        )
    return sorted(rows, key=lambda r: r["seen_mae"])


BUILDERS = {
    "table1_cohorts.csv": table1,
    "table2_regime_utility.csv": table2,
    "table3_pairwise_deltas.csv": table3,
    "table4_deployment_subset_mae.csv": table4,
}


def write_tables() -> dict[str, list[dict[str, Any]]]:
    built = {}
    TABLES.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        rows = builder()
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with open(TABLES / name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        built[name] = rows
    return built


# ---------------------------------------------------------------- verification


def _md_table(caption_prefix: str, text: str) -> list[list[str]]:
    """Rows (cells) of the first markdown table after the caption line."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(caption_prefix)), None)
    if start is None:
        raise SystemExit(f"FAIL: caption {caption_prefix!r} not found in {MAIN_MD.name}")
    rows: list[list[str]] = []
    for ln in lines[start + 1 :]:
        if ln.startswith("|"):
            cells = [
                c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", ln.strip())[1:-1]
            ]
            if not set("".join(cells)) <= set("-: "):
                rows.append(cells)
        elif rows:
            break
    return rows[1:]  # drop header


def _num(cell: str) -> list[float]:
    return [float(x.replace("−", "-")) for x in re.findall(r"[−-]?\d+(?:\.\d+)?", cell)]


def _fmt_close(shown: float, value: float, decimals: int) -> bool:
    return abs(shown - round(value, decimals)) < 0.5 * 10**-decimals + 1e-12


def verify(built: dict[str, list[dict[str, Any]]]) -> list[str]:
    text = plain_text(MAIN_MD.read_text(encoding="utf-8"))
    errors: list[str] = []

    t1 = {r[0]: r for r in _md_table("Table 1 |", text)}
    for row in built["table1_cohorts.csv"]:
        md = t1.get(row["cohort"])
        if md is None or _num(md[1]) != [row["n"]]:
            errors.append(f"Table 1 {row['cohort']}: manuscript {md} vs n={row['n']}")
        elif (
            "strict_rows_removed" in row
            and f"{row['strict_rows_removed']} strict rows" not in md[3]
        ):
            errors.append(
                f"Table 1 {row['cohort']}: removed-row count {row['strict_rows_removed']}"
            )
        elif "sources" in row and f"{row['sources']} literature sources" not in md[2]:
            errors.append(f"Table 1 {row['cohort']}: source count {row['sources']}")

    md2 = _md_table("Table 2 |", text)
    if len(md2) != len(built["table2_regime_utility.csv"]):
        errors.append(
            f"Table 2: {len(md2)} manuscript rows vs {len(built['table2_regime_utility.csv'])}"
        )
    for md, row in zip(md2, built["table2_regime_utility.csv"], strict=False):
        if md[0] != row["regime"] or not md[1].startswith(row["model_set"]):
            errors.append(f"Table 2 row order: {md[:2]} vs {row['regime']}/{row['model_set']}")
        for cell, stat in zip(md[2:6], STATS, strict=True):
            (shown,) = _num(cell)
            if not _fmt_close(shown, row[stat], 4):
                errors.append(
                    f"Table 2 {row['regime']} {row['model_set']} {stat}: {shown} vs {row[stat]:.6f}"
                )

    md3 = _md_table("Table 3 |", text)
    for md, row in zip(md3, built["table3_pairwise_deltas.csv"], strict=False):
        point, low, high = _num(md[2])[0], *_num(md[3])
        for shown, value, what in (
            (point, row["point_delta"], "Δ"),
            (low, row["ci_low"], "CI low"),
            (high, row["ci_high"], "CI high"),
        ):
            if not _fmt_close(shown, value, 4):
                errors.append(
                    f"Table 3 {row['comparison']} {row['statistic']} {what}: {shown} vs {value:.6f}"
                )
        if md[4] != row["ci_excludes_zero"]:
            errors.append(f"Table 3 {row['comparison']} {row['statistic']}: excludes-zero {md[4]}")
    if len(md3) != len(built["table3_pairwise_deltas.csv"]):
        errors.append(
            f"Table 3: {len(md3)} manuscript rows vs {len(built['table3_pairwise_deltas.csv'])}"
        )

    md4 = _md_table("Table 4 |", text)
    for md, row in zip(md4, built["table4_deployment_subset_mae.csv"], strict=False):
        if md[0] != row["model"]:
            errors.append(f"Table 4 row order: {md[0]} vs {row['model']}")
        expected = [
            [row["seen_mae"], row["seen_ci_low"], row["seen_ci_high"]],
            [row["glass_mae"], row["glass_ci_low"], row["glass_ci_high"]],
            [row["crystalline_mae_descriptive"]],
        ]
        for cell, values in zip(md[1:4], expected, strict=True):
            shown = _num(cell)
            if len(shown) != len(values) or not all(
                _fmt_close(s, v, 2) for s, v in zip(shown, values, strict=True)
            ):
                errors.append(f"Table 4 {row['model']}: {cell} vs {[round(v, 4) for v in values]}")
    if len(md4) != 8:
        errors.append(f"Table 4: {len(md4)} manuscript rows, expected 8")
    return errors


def verify_si_table1() -> list[str]:
    """Verify Supplementary Table 1 (scripts/figures/build_robustness_table.py)."""
    sys.path.insert(0, str(ROOT / "scripts" / "figures"))
    import build_robustness_table as figure_tables  # noqa: PLC0415

    errors = []
    # Rebuild into a temporary directory and compare; never rewrite the record.
    with tempfile.TemporaryDirectory() as tmp:
        figure_tables.TABLES_DIR = Path(tmp)
        figure_tables.build_table4()
        rebuilt = (Path(tmp) / SI_TABLE1_CSV.name).read_bytes()
    if rebuilt != SI_TABLE1_CSV.read_bytes():
        errors.append(f"{SI_TABLE1_CSV.name} differs from its rebuild from the artifacts")
    with open(SI_TABLE1_CSV, newline="", encoding="utf-8") as fh:
        rows = {r["model"]: r for r in csv.DictReader(fh)}
    md_rows = _md_table("Supplementary Table 1 |", plain_text(SI_MD.read_text(encoding="utf-8")))
    by_label = {label: key for key, label in MODEL_LABELS.items()}
    cols = [
        "mae_strict_n204",
        "mae_source_disjoint_n153",
        "mae_delta_source_disjoint",
        "mae_window_20_30_n197",
        "mae_delta_window_only_same_formula_subset",
        "mae_delta_window_20_30_vs_full_strict",
    ]
    if len(md_rows) != len(rows):
        errors.append(f"Supplementary Table 1: {len(md_rows)} rows vs {len(rows)}")
    for md in md_rows:
        row = rows.get(by_label.get(md[0], ""))
        if row is None:
            errors.append(f"Supplementary Table 1: unknown model {md[0]}")
            continue
        for cell, col in zip(md[1:7], cols, strict=True):
            (shown,) = _num(cell)
            if not _fmt_close(shown, float(row[col]), 4):
                errors.append(f"Supplementary Table 1 {md[0]} {col}: {shown} vs {row[col]}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="verify without rewriting CSVs")
    args = parser.parse_args()
    built = {n: b() for n, b in BUILDERS.items()} if args.check_only else write_tables()
    errors = verify(built) + verify_si_table1()
    for err in errors:
        print("FAIL", err)
    print(f"{len(BUILDERS)} tables built; {len(errors)} manuscript mismatches")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
