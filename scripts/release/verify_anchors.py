#!/usr/bin/env python3
"""Recompute the protected scientific anchors from data and authoritative artifacts.

Exits 1 if any dataset count, external MAE, primary regime-utility statistic
or the H1/H2 verdict differs from the values reported in the manuscript.

Run with: python scripts/release/verify_anchors.py [REPO_ROOT]
"""

import collections
import json
import sys
from pathlib import Path

import pandas as pd

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
C = R / "artifacts/authoritative"
# The cohort CSVs are withheld from the public package (Hargreaves terms) and
# exist only after scripts/release/restricted_data.py restore.
EXT = R / "data/processed/external_cohort"
checks = []


def chk(name, got, exp, tol=5e-5):
    ok = (abs(got - exp) <= tol) if isinstance(exp, float) else got == exp
    checks.append((name, got, exp, ok))


idx = json.loads((R / "data/processed/v1/obelix_index.json").read_text())
ex = json.loads((R / "data/processed/v1/obelix_target_primary_exact.json").read_text())
ca = json.loads((R / "data/processed/v1/obelix_target_censoring_aware.json").read_text())
split = {x: k for k, v in idx["splits"].items() for x in v}
cnt = collections.Counter(split[r["id"]] for r in ex["rows"])
chk("obelix_processed_total", idx["counts"]["total"], 599)
chk("exact_label_population", len(ex["rows"]), 562)
chk("development_population", cnt["train"] + cnt["validation"], 449)
chk("exact_label_test", cnt["test"], 113)
chk("censored_excluded", ca["n_left_censored"] + ca["n_right_censored"], 37)
n6 = json.loads((C / "n6_external_transfer_corrected.json").read_text())["results"]
for cohort, n in (("strict", 204), ("relaxed", 239)):
    rows = {len(n6[m][cohort]["predictions"]["reduced_formula"]) for m in n6}
    if (EXT / f"hargreaves_{cohort}.csv").exists():
        rows.add(len(pd.read_csv(EXT / f"hargreaves_{cohort}.csv")))
    chk(f"{cohort}_cohort", rows.pop() if len(rows) == 1 else sorted(rows), n)
sd = next((C / "n15_artifacts").glob("*source_disjoint.json"))
chk("doi_source_disjoint", json.loads(sd.read_text())["n_source_disjoint"], 153)
for m, v in {
    "extra_trees": 1.2022,
    "composition_attention": 1.2824,
    "random_forest": 1.2861,
    "catboost": 1.2916,
    "xgboost": 1.3492,
    "ridge": 1.3820,
    "shallow_mlp": 1.4831,
    "pretrained_element_embedding": 1.7163,
}.items():
    chk(f"external_mae[{m}]", round(n6[m]["strict"]["mae"], 4), v)
u = json.loads((C / "n7b_regime_utility_nested_corrected.json").read_text())
keys = ["median_abs_mae_gap", "spearman_rho", "kendall_tau", "top_model_selection_regret"]
for reg, vals in {
    "A_random_5fold_cv": (0.4531, 0.6429, 0.5000, 0.0000),
    "B_family_group_kfold": (0.6810, 0.5476, 0.3571, 0.1471),
    "C_lomfo": (0.5149, 0.5714, 0.4286, 0.0802),
}.items():
    for k, v in zip(keys, vals, strict=True):
        chk(f"utility[{reg}].{k}", round(u["per_regime_summary"][reg][k], 4), v)
h = list(u["h1_family_aware_smaller_mae_gap_than_random_cv"].values()) + list(
    u["h2_family_aware_better_rank_agreement_than_random_cv"].values()
)
chk("H1_H2_any_supported", any(h), False)
for n, g, e, ok in checks:
    print(f"{'PASS' if ok else 'FAIL'}\t{n}\tgot={g}\texpected={e}")
nf = sum(not c[3] for c in checks)
print(f"{len(checks) - nf}/{len(checks)} PASS")
sys.exit(1 if nf else 0)
