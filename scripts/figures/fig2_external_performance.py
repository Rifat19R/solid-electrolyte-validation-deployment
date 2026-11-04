"""Fig. 2 -- external performance with paired bootstrap confidence intervals
(strict cohort, primary), with the development-population mean/median
constant-predictor baselines shown as a shaded band.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from shared import (
    AUTHORITATIVE_DIR,
    INK_PRIMARY,
    INK_SECONDARY,
    MAIN_WIDTH_IN,
    MODEL_COLORS,
    MODEL_LABELS,
    apply_style,
    save_all,
)

apply_style()

n15 = json.loads(
    (AUTHORITATIVE_DIR / "n15_artifacts" / "n15_external_uncertainty.json").read_text()
)
ctx = json.loads((AUTHORITATIVE_DIR / "n21c_primary_external_context_corrected.json").read_text())
strict = n15["hargreaves_strict"]
per_model = strict["per_model"]
best_model = strict["best_model_point_estimate"]
baseline_lo = min(ctx["mean_baseline_mae"], ctx["median_baseline_mae"])
baseline_hi = max(ctx["mean_baseline_mae"], ctx["median_baseline_mae"])

models_sorted = sorted(per_model.keys(), key=lambda m: per_model[m]["point_mae"])

fig, ax = plt.subplots(figsize=(MAIN_WIDTH_IN, 3.0))
y = list(range(len(models_sorted)))[::-1]

ax.axvspan(baseline_lo, baseline_hi, color="#e3e3e3", lw=0, zorder=0)
ax.text(
    (baseline_lo + baseline_hi) / 2,
    len(models_sorted) - 0.35,
    "constant-predictor\nbaselines",
    ha="center",
    va="bottom",
    fontsize=7,
    color=INK_SECONDARY,
    linespacing=1.1,
)

for yi, model in zip(y, models_sorted, strict=True):
    d = per_model[model]
    color = MODEL_COLORS[model]
    ax.plot(
        [d["ci_low"], d["ci_high"]],
        [yi, yi],
        color=color,
        lw=2.4,
        solid_capstyle="round",
        zorder=2,
    )
    ax.plot(
        [d["point_mae"]],
        [yi],
        marker="o",
        color=color,
        markeredgecolor=INK_PRIMARY,
        markeredgewidth=0.8,
        markersize=7 if model == best_model else 5.5,
        zorder=3,
    )
    # P(rank 1) as an aligned column just outside the plotting area.
    ax.text(
        1.02,
        yi,
        f"{d['probability_rank_1']:.2f}",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        fontsize=7.5,
        color=INK_SECONDARY,
    )

ax.text(
    1.02,
    len(models_sorted) - 0.35,
    "P(rank 1)",
    transform=ax.get_yaxis_transform(),
    va="bottom",
    ha="left",
    fontsize=7.5,
    color=INK_PRIMARY,
)

ax.set_yticks(y)
ax.set_yticklabels([MODEL_LABELS[m] + (" *" if m == best_model else "") for m in models_sorted])
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, len(models_sorted) - 0.35 + 0.9)
ax.set_xlabel(r"External MAE, log$_{10}$($\sigma_{\mathrm{RT}}$ / S cm$^{-1}$), strict cohort")
ax.set_xlim(0.95, 2.0)
ax.grid(axis="x")
ax.spines["left"].set_visible(False)

fig.tight_layout()
save_all(fig, "fig2_external_performance")
plt.close(fig)
