"""Shared sys.path/import glue for the manuscript figure and table builders.

Reuses scripts/manuscript_figures/palette.py unchanged (no new colors
invented) for CVD-safety and typographic consistency with the rest of the
project's figures, then applies two N16-specific overrides on top of
`apply_style()`:
  - pure white figure/axes/savefig background (submission figure spec), not the
    shared palette's off-white SURFACE tone used elsewhere in the project;
  - a slightly smaller base font so the effective *final* (post-shrink,
    two-column) font size lands near 8 pt, per the N16 figure-quality spec.

Importing this module only applies the style; it creates no directories.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "manuscript_figures"))

import matplotlib as mpl  # noqa: E402
from palette import (  # noqa: E402,F401
    CATEGORICAL,
    CATEGORICAL_ORDER,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    apply_style,
)

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
AUTHORITATIVE_DIR = ARTIFACTS_DIR / "authoritative"
TABLES_DIR = REPO_ROOT / "manuscript" / "tables"
OUT_DIR = REPO_ROOT / "manuscript" / "figures"

PNG_DPI = 300


def apply_n16_style() -> None:
    apply_style()
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "svg.fonttype": "none",
        }
    )


def save_all(fig, stem: str, *, bbox_inches: str | None = "tight") -> None:
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches=bbox_inches)
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches=bbox_inches)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches=bbox_inches, dpi=PNG_DPI)
    print(f"Wrote {OUT_DIR / stem}.{{pdf,svg,png}}")


# Fixed per-model color/label assignment, stable across every N16 figure that
# shows more than one model -- identity, not rank (unchanged convention from
# the earlier figure builder). Display label for
# `pretrained_element_embedding` renamed to "pooled elemental-property model"
# per N16 spec; the frozen internal artifact key is unchanged.
MODEL_ORDER = [
    "ridge",
    "random_forest",
    "extra_trees",
    "xgboost",
    "catboost",
    "shallow_mlp",
    "pretrained_element_embedding",
    "composition_attention",
]
MODEL_LABELS = {
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "shallow_mlp": "Shallow MLP",
    "pretrained_element_embedding": "Pooled elemental-property model",
    "composition_attention": "Composition attention",
}
STABLE_MODELS = [
    "random_forest",
    "extra_trees",
    "xgboost",
    "catboost",
    "pretrained_element_embedding",
    "composition_attention",
]
MODEL_COLORS = {
    model: CATEGORICAL[CATEGORICAL_ORDER[i % len(CATEGORICAL_ORDER)]]
    for i, model in enumerate(MODEL_ORDER)
}

REGIME_ORDER = ["A_random_5fold_cv", "B_family_group_kfold", "C_lomfo"]
REGIME_LABELS = {
    "A_random_5fold_cv": "A: Random 5-fold CV",
    "B_family_group_kfold": "B: Family-GroupKFold",
    "C_lomfo": "C: LOMFO",
}
REGIME_COLORS = {
    "A_random_5fold_cv": CATEGORICAL["blue"],
    "B_family_group_kfold": CATEGORICAL["aqua"],
    "C_lomfo": CATEGORICAL["yellow"],
}
