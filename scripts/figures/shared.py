"""Shared style, palette and output glue for the manuscript and SI figures.

Every figure is drawn at its final printed width (the submission build embeds
main-text figures at 6.45 in and SI figures at 6.35 in), so the font sizes
set here are the sizes that appear on the page.

Colour system (one mapping, used identically in every figure):
  - models: Paul Tol's "muted" colour-blind-safe qualitative set, fixed per
    model (identity, not rank);
  - internal-validation regimes: a separate scheme that shares no hue with
    the model set -- random CV (A) in neutral gray, the two family-aware
    regimes (B, C) in two blues of increasing exclusion severity, the
    formula-grouped leakage control (D) in ochre;
  - external cohorts: brick tones; risk-score comparators: grays, with
    random rejection (the baseline) lightest.

Figures built with this module must not contain headline suptitles, section
symbols, artifact or script file paths, or in-image footnote text; all
descriptive text lives in the manuscript figure caption.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
# Figures read the authoritative (corrective-01 corrected) results directly;
# artifacts/historical/ holds the superseded pre-correction values.
AUTHORITATIVE_DIR = ARTIFACTS_DIR / "authoritative"
FROZEN_DIR = ARTIFACTS_DIR / "frozen"
OUT_DIR = REPO_ROOT / "manuscript" / "figures"
SI_OUT_DIR = OUT_DIR / "si"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SI_OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_DPI = 600
MAIN_WIDTH_IN = 6.45
SI_WIDTH_IN = 6.35

# Ink and chrome.
INK_PRIMARY = "#1a1a1a"
INK_SECONDARY = "#4d4d4d"
INK_MUTED = "#7a7a7a"
GRID = "#e6e6e6"
SPINE = "#9a9a9a"
NEUTRAL_FILL = "#f2f2f2"

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
# Display label for `pretrained_element_embedding` is "Pooled elemental-property
# model"; the frozen internal artifact key is unchanged.
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
    "ridge": "#CC6677",  # rose
    "random_forest": "#999933",  # olive
    "extra_trees": "#117733",  # green
    "xgboost": "#DDCC77",  # sand
    "catboost": "#44AA99",  # teal
    "shallow_mlp": "#882255",  # wine
    "pretrained_element_embedding": "#88CCEE",  # cyan
    "composition_attention": "#AA4499",  # purple
}

REGIME_ORDER = ["A_random_5fold_cv", "B_family_group_kfold", "C_lomfo"]
REGIME_ORDER_WITH_D = [*REGIME_ORDER, "D_formula_group_kfold"]
REGIME_LABELS_WITH_D = {
    "A_random_5fold_cv": "A: Random 5-fold CV",
    "B_family_group_kfold": "B: Family-GroupKFold",
    "C_lomfo": "C: LOMFO",
    "D_formula_group_kfold": "D: Formula-GroupKFold",
}
REGIME_LABELS = {k: REGIME_LABELS_WITH_D[k] for k in REGIME_ORDER}
REGIME_COLORS_WITH_D = {
    "A_random_5fold_cv": "#8c8c8c",  # neutral gray (random)
    "B_family_group_kfold": "#5b8fc7",  # steel blue
    "C_lomfo": "#1f3d6b",  # navy
    "D_formula_group_kfold": "#c08a1e",  # ochre (leakage control)
}
REGIME_COLORS = {k: REGIME_COLORS_WITH_D[k] for k in REGIME_ORDER}

EXTERNAL_COLORS = {
    "hargreaves_strict": "#a8432f",  # brick
    "hargreaves_source_disjoint": "#d9876b",  # light brick
}

# Risk-score comparators (Fig. 6): grays, the tested AD score darkest and the
# random-rejection baseline lightest; no hue is shared with models/regimes.
RISK_COLORS = {
    "ad_nn_distance": "#2f2f2f",
    "family_novelty": "#6e6e6e",
    "cross_model_dispersion": "#6e6e6e",
    "kde_novelty": "#6e6e6e",
    "random_rejection": "#c4c4c4",
}


def lighten(hex_color: str, amount: float = 0.55) -> str:
    """Blend a colour toward white by `amount` (0 = unchanged, 1 = white)."""
    r, g, b = mpl.colors.to_rgb(hex_color)
    return mpl.colors.to_hex((r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount))


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Sans",
            "mathtext.it": "Liberation Sans:italic",
            "mathtext.bf": "Liberation Sans:bold",
            "mathtext.cal": "Liberation Sans",
            "mathtext.sf": "Liberation Sans",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "normal",
            "axes.labelsize": 8,
            "axes.labelcolor": INK_PRIMARY,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handleheight": 0.9,
            "hatch.linewidth": 0.6,
            "text.color": INK_PRIMARY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_letter(ax, letter: str, *, x: float = -0.02, y: float = 1.02) -> None:
    """Bold lower-case panel letter at the top-left corner, outside the axes."""
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="bottom",
        color=INK_PRIMARY,
    )


def save_all(fig, stem: str, *, si: bool = False) -> None:
    out_dir = SI_OUT_DIR if si else OUT_DIR
    # No CreationDate: identical inputs give byte-identical PDFs.
    kw = {"bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(out_dir / f"{stem}.pdf", metadata={"CreationDate": None}, **kw)
    fig.savefig(out_dir / f"{stem}.png", dpi=PNG_DPI, **kw)
    print(f"Wrote {out_dir / stem}.{{pdf,png}}")
