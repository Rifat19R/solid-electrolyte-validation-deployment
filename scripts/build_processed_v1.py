#!/usr/bin/env python3
"""Phase 0C: build frozen, leakage-safe derived datasets under data/processed/v1/.

Reads only from data/raw/ (via the existing loaders) and from the Phase 0B
leakage-audit reports; never modifies raw files. Writes:

- reports/results/leakage/exclusion_ledger.{json,csv}
- data/processed/v1/obelix_index.json                  (frozen train/val/test split)
- data/processed/v1/transport_pretraining_index.json   (bvel_percolation + bvse_hops)
- data/processed/v1/dft_neb_index.json
- data/processed/v1/excluded_records_index.json
- data/processed/v1/obelix_target_primary_exact.json
- data/processed/v1/obelix_target_censoring_aware.json

Then freezes the split by invoking scripts/freeze_split.py on obelix_index.json
and exclusion_ledger.json, producing SHA-256 sidecars.

Run with: python scripts/build_processed_v1.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sse_transfer.data.leakage_policy import (  # noqa: E402
    build_ledger,
    excluded_material_ids,
)
from sse_transfer.data.litraj_loader import (  # noqa: E402
    load_bvel13k_index,
    load_nebbvse122k_index,
    load_nebdft2k_index,
)
from sse_transfer.data.obelix_loader import (  # noqa: E402
    censoring_bound,
    label_status,
    load_obelix,
)
from sse_transfer.data.splits import build_composition_group_validation_split  # noqa: E402

LEAKAGE_DIR = ROOT / "reports" / "results" / "leakage"
PROCESSED_DIR = ROOT / "data" / "processed" / "v1"
STRUCTURE_OVERLAP_REPORT = LEAKAGE_DIR / "obelix_vs_pretraining_structure_overlap.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def _freeze(path: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "freeze_split.py"), str(path)],
        check=True,
        cwd=ROOT,
    )


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    vs_cfg = config["validation_split"]
    seed = int(vs_cfg["seed"])
    target_fraction = float(vs_cfg["target_fraction"])

    if not STRUCTURE_OVERLAP_REPORT.exists():
        raise SystemExit(
            f"Missing {STRUCTURE_OVERLAP_REPORT.relative_to(ROOT)}. "
            "Run scripts/run_leakage_audit.py first."
        )
    structure_overlap = json.loads(STRUCTURE_OVERLAP_REPORT.read_text(encoding="utf-8"))

    # --- 1. Freeze the OBELiX train/validation/test split ------------------
    obelix_all = load_obelix("all")
    obelix_train = load_obelix("train")
    obelix_test = load_obelix("test")
    test_ids = sorted(obelix_test["ID"].astype(str))

    holdout = build_composition_group_validation_split(
        obelix_train, seed=seed, target_fraction=target_fraction
    )

    obelix_index = {
        "dataset": "obelix",
        "processed_version": "v1",
        "official_test_preserved": True,
        "seed": seed,
        "target_fraction": target_fraction,
        "grouping_rule": holdout.grouping_rule,
        "validation_groups": holdout.validation_groups,
        "train_groups": holdout.train_groups,
        "splits": {
            "train": holdout.train_ids,
            "validation": holdout.validation_ids,
            "test": test_ids,
        },
        "counts": {
            "train": len(holdout.train_ids),
            "validation": len(holdout.validation_ids),
            "test": len(test_ids),
            "total": len(holdout.train_ids) + len(holdout.validation_ids) + len(test_ids),
        },
        "notes": (
            "test is the untouched official OBELiX test.csv split; validation is "
            "carved out of the official train.csv by whole composition group and "
            "must never be altered because of model performance."
        ),
    }
    assert obelix_index["counts"]["total"] == len(obelix_all)
    _write_json(PROCESSED_DIR / "obelix_index.json", obelix_index)

    # --- 2. Apply the leakage-exclusion policy ------------------------------
    protected: dict[str, str] = {qid: "final_test_protection" for qid in test_ids}
    for qid in holdout.validation_ids:
        protected.setdefault(qid, "validation_protection")

    excluded = excluded_material_ids(structure_overlap["matches"], protected_obelix_ids=protected)

    bvel_df = load_bvel13k_index()
    bvse_df = load_nebbvse122k_index()
    dft_df = load_nebdft2k_index()

    bvse_edge_index: dict[str, list[str]] = (
        bvse_df.groupby("material_id")["edge_id"].apply(list).to_dict()
    )
    dft_edge_index: dict[str, list[str]] = (
        dft_df.groupby("material_id")["edge_id"].apply(list).to_dict()
    )

    ledger_entries = build_ledger(
        excluded,
        bvel_material_ids=set(bvel_df["material_id"]),
        bvse_edge_index=bvse_edge_index,
        dft_edge_index=dft_edge_index,
    )
    ledger_payload = {
        "policy_reference": "CLAUDE.md Phase 0C leakage-exclusion hierarchy",
        "n_excluded_materials": len(excluded),
        "n_excluded_records": len(ledger_entries),
        "entries": [e.as_dict() for e in ledger_entries],
    }
    _write_json(LEAKAGE_DIR / "exclusion_ledger.json", ledger_payload)

    ledger_csv = LEAKAGE_DIR / "exclusion_ledger.csv"
    with ledger_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_dataset",
                "record_id",
                "material_id",
                "match_type",
                "protection_tier",
                "matched_obelix_ids",
                "exclusion_reason",
            ],
        )
        writer.writeheader()
        for e in ledger_entries:
            row = e.as_dict()
            row["matched_obelix_ids"] = ";".join(row["matched_obelix_ids"])
            writer.writerow(row)
    print(f"wrote {ledger_csv.relative_to(ROOT)}")

    excluded_bvel = {mid for mid in excluded if mid in set(bvel_df["material_id"])}
    excluded_bvse_edges = {edge_id for mid in excluded for edge_id in bvse_edge_index.get(mid, [])}
    excluded_dft_edges = {edge_id for mid in excluded for edge_id in dft_edge_index.get(mid, [])}

    # --- 3. Transport pretraining index (roles A + B) -----------------------
    transport_index = {
        "bvel_percolation": [
            {
                "material_id": row["material_id"],
                "litraj_split": row["_split"],
                "included": row["material_id"] not in excluded_bvel,
            }
            for row in bvel_df.to_dict("records")
        ],
        "bvse_hops": [
            {
                "edge_id": row["edge_id"],
                "material_id": row["material_id"],
                "litraj_split": row["_split"],
                "included": row["edge_id"] not in excluded_bvse_edges,
            }
            for row in bvse_df.to_dict("records")
        ],
        "counts": {
            "bvel_percolation_total": len(bvel_df),
            "bvel_percolation_excluded": len(excluded_bvel),
            "bvse_hops_total": len(bvse_df),
            "bvse_hops_excluded": len(excluded_bvse_edges),
        },
    }
    _write_json(PROCESSED_DIR / "transport_pretraining_index.json", transport_index)

    # --- 4. DFT-NEB index (role C) ------------------------------------------
    dft_index = {
        "dft_neb": [
            {
                "edge_id": row["edge_id"],
                "material_id": row["material_id"],
                "litraj_split": row["_split"],
                "included": row["edge_id"] not in excluded_dft_edges,
            }
            for row in dft_df.to_dict("records")
        ],
        "counts": {
            "total": len(dft_df),
            "excluded": len(excluded_dft_edges),
        },
    }
    _write_json(PROCESSED_DIR / "dft_neb_index.json", dft_index)

    # --- 5. Excluded-record index (compact cross-reference) -----------------
    excluded_index = {
        "ledger_reference": "reports/results/leakage/exclusion_ledger.json",
        "bvel_percolation": sorted(excluded_bvel),
        "bvse_hops": sorted(excluded_bvse_edges),
        "dft_neb": sorted(excluded_dft_edges),
    }
    _write_json(PROCESSED_DIR / "excluded_records_index.json", excluded_index)

    # --- 6. Censoring-policy target indices ---------------------------------
    conductivity_col = obelix_all["Ionic conductivity (S cm-1)"]
    status = label_status(conductivity_col)
    bound = censoring_bound(conductivity_col)

    split_of: dict[str, str] = {}
    for rid in holdout.train_ids:
        split_of[rid] = "train"
    for rid in holdout.validation_ids:
        split_of[rid] = "validation"
    for rid in test_ids:
        split_of[rid] = "test"

    censoring_rows = []
    exact_rows = []
    import math

    for row_id, stat, b in zip(obelix_all["ID"].astype(str), status, bound, strict=True):
        split = split_of.get(row_id, "unknown")
        log10_bound = math.log10(b) if not math.isnan(b) else None
        entry = {
            "id": row_id,
            "obelix_split": split,
            "label_status": stat,
            "log10_bound": log10_bound,
            "bound_direction": (
                "upper"
                if stat == "LEFT_CENSORED"
                else "lower"
                if stat == "RIGHT_CENSORED"
                else None
            ),
            "y_exact": log10_bound if stat == "EXACT" else None,
        }
        censoring_rows.append(entry)
        if stat == "EXACT":
            exact_rows.append({"id": row_id, "obelix_split": split, "y": log10_bound})

    primary_exact = {
        "target": "log10_sigma_rt",
        "unit": "log10(S/cm)",
        "policy": "primary regression index: EXACT-labeled rows only",
        "temperature_caveat": (
            "OBELiX's raw conductivity column has no per-row temperature field. "
            "Dataset-level room-temperature scope is VERIFIED_BY_PRIMARY_SOURCE "
            "against DOI 10.1039/D5DD00441A (see data/manifests/obelix.json "
            "'temperature_scope_verification'); individual-row measurement "
            "temperature is not independently confirmable from the local CSVs."
        ),
        "n_rows": len(exact_rows),
        "rows": exact_rows,
    }
    _write_json(PROCESSED_DIR / "obelix_target_primary_exact.json", primary_exact)

    censoring_aware = {
        "target": "log10_sigma_rt",
        "unit": "log10(S/cm)",
        "policy": (
            "censoring-aware / sensitivity-analysis index: all rows, with an explicit "
            "label_status; censored rows carry a bound and direction instead of a "
            "point value and must not be treated as exact regression targets"
        ),
        "label_status_values": ["EXACT", "LEFT_CENSORED", "RIGHT_CENSORED", "MISSING"],
        "n_rows": len(censoring_rows),
        "n_exact": int((status == "EXACT").sum()),
        "n_left_censored": int((status == "LEFT_CENSORED").sum()),
        "n_right_censored": int((status == "RIGHT_CENSORED").sum()),
        "n_missing": int((status == "MISSING").sum()),
        "rows": censoring_rows,
    }
    _write_json(PROCESSED_DIR / "obelix_target_censoring_aware.json", censoring_aware)

    # --- 7. Freeze splits ----------------------------------------------------
    _freeze(PROCESSED_DIR / "obelix_index.json")
    _freeze(LEAKAGE_DIR / "exclusion_ledger.json")

    print("\nSummary:")
    print(f"  obelix train/validation/test = {obelix_index['counts']}")
    print(f"  excluded materials = {len(excluded)}")
    print(f"  excluded bvel_percolation records = {len(excluded_bvel)}")
    print(f"  excluded bvse_hops records = {len(excluded_bvse_edges)}")
    print(f"  excluded dft_neb records = {len(excluded_dft_edges)}")
    print(
        f"  censoring: {censoring_aware['n_exact']} EXACT, "
        f"{censoring_aware['n_left_censored']} LEFT_CENSORED, "
        f"{censoring_aware['n_right_censored']} RIGHT_CENSORED, "
        f"{censoring_aware['n_missing']} MISSING"
    )


if __name__ == "__main__":
    main()
