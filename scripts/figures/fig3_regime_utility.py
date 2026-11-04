"""Fig. 3 -- validation-regime utility: primary 8-model result (filled bars)
and stable 6-model sensitivity (hatched bars) for regimes A-D, one panel per
comparison statistic.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from shared import (
    AUTHORITATIVE_DIR,
    INK_MUTED,
    INK_PRIMARY,
    MAIN_WIDTH_IN,
    REGIME_COLORS_WITH_D,
    REGIME_ORDER_WITH_D,
    apply_style,
    lighten,
    panel_letter,
    save_all,
)

apply_style()

n7b = json.loads((AUTHORITATIVE_DIR / "n7b_regime_utility_nested_corrected.json").read_text())
n15 = json.loads(
    (AUTHORITATIVE_DIR / "n15_artifacts" / "n15_stable_model_sensitivity.json").read_text()
)
fixes = json.loads((AUTHORITATIVE_DIR / "phase1_factual_fixes_corrected.json").read_text())

primary = dict(n7b["per_regime_summary"])
stable = dict(n15["per_regime_summary"])
primary["D_formula_group_kfold"] = fixes["item3_formula_gkf_utility"]["primary_8_model"]
stable["D_formula_group_kfold"] = fixes["item3_formula_gkf_utility"]["six_model_stable_sensitivity"]

STATS = [
    ("median_abs_mae_gap", "Median |internal − external| MAE gap", "lower is better", False),
    ("spearman_rho", "Spearman ρ with external ranking", "higher is better", True),
    ("kendall_tau", "Kendall τ with external ranking", "higher is better", True),
    ("top_model_selection_regret", "Top-model selection regret", "lower is better", False),
]
TICK_LABELS = ["A\nRandom", "B\nFamily-GKF", "C\nLOMFO", "D\nFormula-GKF"]

fig, axes = plt.subplots(2, 2, figsize=(MAIN_WIDTH_IN, 4.9))
x = np.arange(len(REGIME_ORDER_WITH_D))
width = 0.38

for ax, letter, (key, title, direction, higher_better) in zip(
    axes.flat, "abcd", STATS, strict=True
):
    primary_vals = [primary[r][key] for r in REGIME_ORDER_WITH_D]
    stable_vals = [stable[r][key] for r in REGIME_ORDER_WITH_D]
    colors = [REGIME_COLORS_WITH_D[r] for r in REGIME_ORDER_WITH_D]
    bars_primary = ax.bar(
        x - width / 2, primary_vals, width, color=colors, edgecolor="white", linewidth=0.4
    )
    ax.bar(
        x + width / 2,
        stable_vals,
        width,
        color=[lighten(c, 0.6) for c in colors],
        edgecolor=colors,
        linewidth=0.6,
        hatch="////",
    )
    best_p = max(primary_vals) if higher_better else min(primary_vals)
    for i, v in enumerate(primary_vals):
        if np.isclose(v, best_p, rtol=0, atol=1e-12):  # outline every tied best regime
            bars_primary[i].set_edgecolor(INK_PRIMARY)
            bars_primary[i].set_linewidth(1.3)
    ymax = max(max(primary_vals), max(stable_vals)) * 1.12
    bar_x = [*(x - width / 2), *(x + width / 2)]
    for xi, v in zip(bar_x, [*primary_vals, *stable_vals], strict=True):
        if v == 0:  # zero-height bars are otherwise invisible
            ax.text(xi, ymax * 0.015, "0", ha="center", va="bottom", fontsize=7, color=INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(TICK_LABELS, fontsize=7)
    ax.tick_params(axis="x", length=0)
    ax.set_title(f"{title}\n", loc="left", fontsize=8)
    ax.text(
        0.0,
        1.035,
        direction,
        transform=ax.transAxes,
        fontsize=7,
        color=INK_MUTED,
        style="italic",
        va="bottom",
    )
    ax.grid(axis="y")
    ax.set_ylim(0, ymax)
    ax.axhline(0, color=INK_MUTED, lw=0.7)
    panel_letter(ax, letter, x=-0.12, y=1.06)

legend_handles = [
    Patch(facecolor="#8c8c8c", edgecolor="white", label="8 models (pre-specified primary)"),
    Patch(
        facecolor=lighten("#8c8c8c", 0.6),
        edgecolor="#8c8c8c",
        hatch="////",
        label="6 stable models (post hoc sensitivity)",
    ),
    Patch(
        facecolor="white",
        edgecolor=INK_PRIMARY,
        linewidth=1.3,
        label="best 8-model regime (ties outlined)",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    fontsize=7.2,
    columnspacing=1.6,
)

fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.6, w_pad=2.0)
save_all(fig, "fig3_regime_utility")
plt.close(fig)
