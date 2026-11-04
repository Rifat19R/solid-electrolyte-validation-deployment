"""SI Fig. 1 -- deployment-shift alignment (target-free, composition-space),
using the robust (post hoc-fixed) distance/Wasserstein numbers as primary
(artifacts/authoritative/n21e_robust_shift_distance.json); the original
standardization-scale-artifact numbers are retained only in the SI text,
labelled superseded.

Panel a: interquartile range (bar), median (tick), median-to-90th-percentile
whisker and cohort maximum (diamond). Panel b: robust 1-D Wasserstein
distance of each internal regime vs. the strict and source-disjoint cohorts.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from shared import (
    AUTHORITATIVE_DIR,
    EXTERNAL_COLORS,
    INK_PRIMARY,
    REGIME_COLORS,
    SI_WIDTH_IN,
    apply_style,
    lighten,
    panel_letter,
    save_all,
)

apply_style()

n21e = json.loads((AUTHORITATIVE_DIR / "n21e_robust_shift_distance.json").read_text())
dist = n21e["distance_distribution_summary_robust"]
wass = n21e["wasserstein_distance_internal_vs_external_robust"]

CATS = [
    ("A_random_5fold_cv", "A: Random 5-fold CV", REGIME_COLORS["A_random_5fold_cv"]),
    ("B_family_group_kfold", "B: Family-GroupKFold", REGIME_COLORS["B_family_group_kfold"]),
    ("C_lomfo", "C: LOMFO", REGIME_COLORS["C_lomfo"]),
    ("hargreaves_strict", "Strict cohort", EXTERNAL_COLORS["hargreaves_strict"]),
    (
        "hargreaves_source_disjoint",
        "Source-disjoint cohort",
        EXTERNAL_COLORS["hargreaves_source_disjoint"],
    ),
]

fig, (ax, ax_b) = plt.subplots(1, 2, figsize=(SI_WIDTH_IN, 3.0), width_ratios=[1.45, 1.0])

for i, (key, _label, color) in enumerate(CATS):
    dd = dist[key]
    y = len(CATS) - 1 - i
    q25, q75, med, q90, mx = dd["q25"], dd["q75"], dd["median"], dd["q90"], dd["max"]
    ax.plot([med, q90], [y, y], color=color, lw=1.2, zorder=2)
    ax.plot([q90, q90], [y - 0.12, y + 0.12], color=color, lw=1.2, zorder=2)
    ax.plot([q25, q75], [y, y], color=lighten(color, 0.45), lw=7, solid_capstyle="butt", zorder=3)
    ax.plot([med], [y], marker="|", color=INK_PRIMARY, markersize=11, markeredgewidth=1.5, zorder=4)
    ax.plot(
        [mx],
        [y],
        marker="D",
        color=color,
        markeredgecolor=INK_PRIMARY,
        markeredgewidth=0.6,
        markersize=5,
        zorder=5,
    )

ax.set_yticks([len(CATS) - 1 - i for i in range(len(CATS))])
ax.set_yticklabels([c[1] for c in CATS])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.axhline(1.5, color="#d0d0d0", lw=0.6, ls=":")
for label, yy, va in (("internal ↑", 1.58, "bottom"), ("external ↓", 1.42, "top")):
    ax.text(27.8, yy, label, ha="right", va=va, fontsize=7, style="italic", color="#7a7a7a")
ax.set_ylim(-0.6, len(CATS) - 0.4)
ax.set_xlim(0, 28)
ax.set_xlabel("Robust standardized composition-space\nNN distance to development population")
ax.grid(axis="x")
ax.legend(
    handles=[
        Patch(facecolor="#bdbdbd", label="IQR"),
        Line2D([], [], marker="|", color=INK_PRIMARY, ls="none", markersize=9, label="median"),
        Line2D([], [], color="#7a7a7a", lw=1.2, label="to 90th pct."),
        Line2D(
            [],
            [],
            marker="D",
            color="#bdbdbd",
            markeredgecolor=INK_PRIMARY,
            ls="none",
            markersize=5,
            label="maximum",
        ),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=4,
    fontsize=7,
    handlelength=1.2,
    columnspacing=1.0,
    borderaxespad=0.2,
)
panel_letter(ax, "a", x=-0.36, y=1.02)

regimes = ["A_random_5fold_cv", "B_family_group_kfold", "C_lomfo"]
x = np.arange(3)
width = 0.38
vs_strict = [wass[r]["vs_hargreaves_strict"] for r in regimes]
vs_sd = [wass[r]["vs_hargreaves_source_disjoint"] for r in regimes]
colors = [REGIME_COLORS[r] for r in regimes]
ax_b.bar(x - width / 2, vs_strict, width, color=colors, edgecolor="white", linewidth=0.4)
ax_b.bar(
    x + width / 2,
    vs_sd,
    width,
    color=[lighten(c, 0.6) for c in colors],
    edgecolor=colors,
    linewidth=0.6,
    hatch="////",
)
ax_b.set_yscale("log")
ax_b.set_ylim(0.3, 10)
ax_b.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2, 5, 10]))
ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax_b.yaxis.set_minor_locator(NullLocator())
ax_b.set_xticks(x)
ax_b.set_xticklabels(["A\nRandom", "B\nFamily-GKF", "C\nLOMFO"])
ax_b.tick_params(axis="x", length=0)
ax_b.set_ylabel("Robust 1-D Wasserstein distance\n(log axis)")
ax_b.grid(axis="y")
ax_b.legend(
    handles=[
        Patch(facecolor="#8c8c8c", edgecolor="white", label="vs. strict"),
        Patch(
            facecolor=lighten("#8c8c8c", 0.6),
            edgecolor="#8c8c8c",
            hatch="////",
            label="vs. source-disjoint",
        ),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=2,
    fontsize=7,
    handlelength=1.2,
    columnspacing=1.0,
    borderaxespad=0.2,
)
panel_letter(ax_b, "b", x=-0.3, y=1.02)

fig.tight_layout(w_pad=1.2)
save_all(fig, "si_fig1_shift_alignment", si=True)
plt.close(fig)
