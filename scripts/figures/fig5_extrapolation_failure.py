"""Fig. 5 -- family-holdout extrapolation failure: per-family Family-GroupKFold
MAE for ridge and the shallow MLP with Extra Trees as a well-behaved
reference (a), and each model's Family-GroupKFold pooled-MAE blowup relative
to its best other-regime pooled MAE (b).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullLocator
from shared import (
    AUTHORITATIVE_DIR,
    INK_MUTED,
    INK_SECONDARY,
    MAIN_WIDTH_IN,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    apply_style,
    panel_letter,
    save_all,
)

apply_style()

n4b = json.loads((AUTHORITATIVE_DIR / "n4b_nested_internal_validation_corrected.json").read_text())
major_families = n4b["major_families"]
results = n4b["results"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(MAIN_WIDTH_IN, 3.2), width_ratios=[1.35, 1.0])

PANEL_A_MODELS = [
    ("ridge", "Ridge"),
    ("shallow_mlp", "Shallow MLP"),
    ("extra_trees", "Extra Trees (reference)"),
]
per_family_by_model = {
    m: results[m]["B_family_group_kfold"]["per_family_mae"] for m, _ in PANEL_A_MODELS
}
families = [
    f for f in major_families if all(f in per_family_by_model[m] for m, _ in PANEL_A_MODELS)
]
families.sort(key=lambda f: per_family_by_model["ridge"][f]["mae"], reverse=True)

xf = np.arange(len(families))
width = 0.27
for i, (model, label) in enumerate(PANEL_A_MODELS):
    values = [per_family_by_model[model][f]["mae"] for f in families]
    ax.bar(
        xf + (i - 1) * width,
        values,
        width,
        label=label,
        color=MODEL_COLORS[model],
        edgecolor="white",
        linewidth=0.3,
    )
ax.set_yscale("log")
ax.set_ylim(0.3, 1e4)
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xticks(xf)
ax.set_xticklabels(families, rotation=40, ha="right", rotation_mode="anchor", fontsize=7.2)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Per-family Family-GroupKFold MAE\n(log$_{10}$ units; log axis)")
ax.grid(axis="y", which="major")
ax.legend(loc="upper right", fontsize=7.2)
panel_letter(ax, "a", x=-0.17, y=1.02)

blowup = {}
for m in MODEL_ORDER:
    r = results[m]
    a = r["A_random_5fold_cv"]["pooled_mae"]
    b = r["B_family_group_kfold"]["pooled_mae"]
    c = r["C_lomfo"]["pooled_mae_over_major_family_rows"]
    best_other = min(a, c)
    blowup[m] = b / best_other

models_sorted_b = sorted(MODEL_ORDER, key=lambda m: blowup[m])
yb = np.arange(len(models_sorted_b))
ax2.barh(
    yb,
    [blowup[m] for m in models_sorted_b],
    color=[MODEL_COLORS[m] for m in models_sorted_b],
    height=0.7,
)
ax2.set_xscale("log")
ax2.set_xlim(0.8, 1e4)
ax2.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax2.xaxis.set_minor_locator(NullLocator())
ax2.set_yticks(yb)
ax2.set_yticklabels([MODEL_LABELS[m] for m in models_sorted_b])
ax2.tick_params(axis="y", length=0)
ax2.spines["left"].set_visible(False)
ax2.axvline(1.0, color=INK_MUTED, lw=0.8, linestyle="--", zorder=1)
ax2.set_xlabel("Family-GroupKFold pooled MAE /\nbest other-regime pooled MAE")
ax2.grid(axis="x", which="major")
for yi, m in zip(yb, models_sorted_b, strict=True):
    ax2.text(
        blowup[m] * 1.25, yi, f"{blowup[m]:.1f}×", va="center", fontsize=7, color=INK_SECONDARY
    )
panel_letter(ax2, "b", x=-1.0, y=1.02)

fig.tight_layout(w_pad=1.0)
save_all(fig, "fig5_extrapolation_failure")
plt.close(fig)
