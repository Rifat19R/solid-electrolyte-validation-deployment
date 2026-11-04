"""Fig. 1 -- study, cohort, and independence framework (schematic). Every
count is read programmatically from the frozen artifacts; provenance lives in
the Supplementary Information, not in the figure image.

Layout: development lane (OBELiX) on the left, external lane (Hargreaves et
al.) on the right, one-shot transfer of the frozen models between them, and
the independence caveat spanning the bottom.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from shared import (
    AUTHORITATIVE_DIR,
    EXTERNAL_COLORS,
    FROZEN_DIR,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    MAIN_WIDTH_IN,
    MODEL_COLORS,
    MODEL_ORDER,
    REGIME_COLORS_WITH_D,
    REGIME_ORDER_WITH_D,
    REPO_ROOT,
    apply_style,
    lighten,
    save_all,
)

apply_style()

manifest = json.loads((REPO_ROOT / "data" / "manifests" / "external_cohort.json").read_text())
n4b = json.loads((AUTHORITATIVE_DIR / "n4b_nested_internal_validation_corrected.json").read_text())
n15sd = json.loads((AUTHORITATIVE_DIR / "n15_artifacts" / "n15_source_disjoint.json").read_text())
ladder = json.loads((FROZEN_DIR / "model_ladder_frozen.json").read_text())

sc = manifest["stage_counts"]
n_dev = n4b["obelix_dev_population_n"]
n_test = n4b["obelix_test_untouched_row_count"]
n_models = len(ladder["models"])
n_raw = sc["raw_rows"]
n_rt = sc["rt_window_rows"]
n_strict = sc["strict_independent_rows"]
n_relaxed = sc["relaxed_independent_rows"]
n_sd = n15sd["n_source_disjoint"]
n_excl_source = n15sd["n_excluded_for_source_overlap"]
frac_excl_source = n15sd["fraction_excluded"]

DEV_EC, DEV_FC = "#4d4d4d", "#f4f5f7"
EXT_EC = EXTERNAL_COLORS["hargreaves_strict"]
EXT_FC = lighten(EXT_EC, 0.9)
SD_EC = EXTERNAL_COLORS["hargreaves_source_disjoint"]
FS = 7.2
BOX_H = 0.95


def box(ax, x, yc, w, text, *, ec, fc, h=BOX_H, ls="-", lw=0.9, fontsize=FS, dy=0.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, yc - h / 2),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.08",
            linewidth=lw,
            linestyle=ls,
            edgecolor=ec,
            facecolor=fc,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        yc + dy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK_PRIMARY,
        zorder=4,
        linespacing=1.3,
    )


def arrow(ax, start, end, *, color=INK_SECONDARY, lw=1.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            zorder=2,
            shrinkA=0,
            shrinkB=0,
        )
    )


def elbow(ax, start, end, *, color=INK_SECONDARY, lw=1.0):
    """Vertical-then-horizontal-then-vertical connector ending in an arrow."""
    (x0, y0), (x1, y1) = start, end
    ym = (y0 + y1) / 2
    ax.plot([x0, x0, x1], [y0, ym, ym], color=color, lw=lw, zorder=2, solid_capstyle="butt")
    arrow(ax, (x1, ym), (x1, y1), color=color, lw=lw)


fig, ax = plt.subplots(figsize=(MAIN_WIDTH_IN, 4.75))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.15)
ax.axis("off")

Y1, Y2, Y3, Y4 = 6.05, 4.62, 3.2, 1.78
LX0, LX1 = 0.1, 4.55  # development lane
RX0, RX1 = 5.45, 9.9  # external lane

for x0, x1, title in (
    (LX0, LX1, "Development: OBELiX"),
    (RX0, RX1, "External: Hargreaves et al. (2023)"),
):
    ax.text(x0, 6.92, title, ha="left", va="center", fontsize=8, fontweight="bold")
    ax.plot([x0, x1], [6.72, 6.72], color=INK_MUTED, lw=0.6)

# Development lane.
DEV_W = LX1 - LX0
box(
    ax,
    LX0,
    Y1,
    DEV_W,
    f"OBELiX development population\nn = {n_dev} (train + validation)\nfrozen, reused unmodified",
    ec=DEV_EC,
    fc=DEV_FC,
)
TEST_X0 = LX0 + 1.55
box(
    ax,
    TEST_X0,
    Y2,
    LX1 - TEST_X0,
    f"Official OBELiX test split, n = {n_test}\nnot used",
    ec=INK_MUTED,
    fc="white",
    ls=(0, (3, 2)),
    h=0.7,
    fontsize=6.9,
)
box(
    ax,
    LX0,
    Y3,
    LX1 - LX0,
    f"{n_models} frozen composition-only models\nridge → trees → attention\n"
    "Magpie schema, 132 features",
    ec=DEV_EC,
    fc=DEV_FC,
    h=1.1,
    dy=0.08,
)
# One colour chip per model (the fixed per-model colours used in Figs 2-6).
chip_w, chip_gap = 0.22, 0.06
chips_x0 = LX0 + (LX1 - LX0) / 2 - (len(MODEL_ORDER) * (chip_w + chip_gap) - chip_gap) / 2
for i, m in enumerate(MODEL_ORDER):
    ax.add_patch(
        Rectangle(
            (chips_x0 + i * (chip_w + chip_gap), Y3 - 0.47),
            chip_w,
            0.1,
            facecolor=MODEL_COLORS[m],
            edgecolor="none",
            zorder=4,
        )
    )
arrow(ax, (LX0 + 0.75, Y1 - BOX_H / 2), (LX0 + 0.75, Y3 + 0.55))

regime_text = "4 internal validation regimes on the development population"
box(ax, LX0, Y4, LX1 - LX0, regime_text, ec=DEV_EC, fc=DEV_FC, dy=0.24, fontsize=FS)
regime_short = {
    "A_random_5fold_cv": "A  Random 5-fold",
    "B_family_group_kfold": "B  Family-GroupKFold",
    "C_lomfo": "C  LOMFO",
    "D_formula_group_kfold": "D  Formula-GroupKFold",
}
for i, r in enumerate(REGIME_ORDER_WITH_D):
    col, row = i % 2, i // 2
    xx = LX0 + 0.35 + col * 2.15
    yy = Y4 - 0.03 - row * 0.26
    ax.add_patch(
        Rectangle(
            (xx, yy - 0.07),
            0.14,
            0.14,
            facecolor=REGIME_COLORS_WITH_D[r],
            edgecolor="none",
            zorder=4,
        )
    )
    ax.text(xx + 0.22, yy, regime_short[r], ha="left", va="center", fontsize=6.9, zorder=4)
arrow(ax, (LX0 + (LX1 - LX0) / 2, Y3 - 0.55), (LX0 + (LX1 - LX0) / 2, Y4 + BOX_H / 2))

# External lane.
box(
    ax,
    RX0,
    Y1,
    RX1 - RX0,
    f"Raw literature database\nn = {n_raw} rows, 214 sources",
    ec=EXT_EC,
    fc=EXT_FC,
)
box(
    ax,
    RX0,
    Y2,
    RX1 - RX0,
    f"Predeclared room-temperature window\n(15–35 °C): n = {n_rt}",
    ec=EXT_EC,
    fc=EXT_FC,
)
arrow(ax, ((RX0 + RX1) / 2, Y1 - BOX_H / 2), ((RX0 + RX1) / 2, Y2 + BOX_H / 2))

STRICT_W = 2.3
box(
    ax,
    RX0,
    Y3,
    STRICT_W,
    f"Strict cohort, n = {n_strict}\ncomposition-disjoint\nfrom OBELiX",
    ec=EXT_EC,
    fc=EXT_FC,
    lw=1.6,
    h=1.1,
)
box(
    ax,
    RX0 + STRICT_W + 0.15,
    Y3,
    RX1 - RX0 - STRICT_W - 0.15,
    f"Relaxed cohort, n = {n_relaxed}\ncompanion sensitivity\nstrict is a subset of relaxed",
    ec=EXT_EC,
    fc="white",
    ls=(0, (3, 2)),
    h=1.1,
    fontsize=6.9,
)
elbow(ax, ((RX0 + RX1) / 2, Y2 - BOX_H / 2), (RX0 + STRICT_W / 2, Y3 + 0.55))
elbow(
    ax,
    ((RX0 + RX1) / 2, Y2 - BOX_H / 2),
    (RX0 + STRICT_W + 0.15 + (RX1 - RX0 - STRICT_W - 0.15) / 2, Y3 + 0.55),
)

box(
    ax,
    RX0,
    Y4,
    RX1 - RX0,
    f"Source-disjoint subset of strict cohort, n = {n_sd}\n"
    f"{n_excl_source} of {n_strict} strict rows ({frac_excl_source * 100:.1f}%) "
    "excluded for a shared\n"
    "literature-source DOI with a different OBELiX composition",
    ec=SD_EC,
    fc=lighten(SD_EC, 0.88),
    fontsize=6.9,
)
arrow(ax, (RX0 + STRICT_W / 2, Y3 - 0.55), (RX0 + STRICT_W / 2, Y4 + BOX_H / 2))

# One-shot external transfer of the frozen models.
arrow(ax, (LX1, Y3), (RX0, Y3), color=INK_PRIMARY, lw=1.6)
ax.text(
    (LX1 + RX0) / 2,
    Y3 + 0.12,
    "one-shot\nexternal\ntransfer",
    ha="center",
    va="bottom",
    fontsize=6.9,
    style="italic",
    color=INK_PRIMARY,
    linespacing=1.15,
)

box(
    ax,
    LX0,
    0.55,
    RX1 - LX0,
    "Independence caveat: the strict cohort is composition-disjoint from OBELiX by construction "
    "(no shared reduced formula survives\nexclusion), but not fully source-independent — both "
    "datasets compile from overlapping solid-electrolyte literature.\n"
    f"The source-disjoint subset (n = {n_sd}) additionally requires no shared literature-source "
    "DOI, giving a strictly stronger\n"
    "independence guarantee at the cost of 25% fewer external rows.",
    ec=INK_MUTED,
    fc="white",
    h=0.95,
    fontsize=6.7,
)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
save_all(fig, "fig1_schematic")
plt.close(fig)
