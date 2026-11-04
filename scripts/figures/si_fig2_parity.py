"""SI Fig. 2 -- observed vs. predicted parity, all eight models, strict
external cohort, on shared axes.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from shared import (
    AUTHORITATIVE_DIR,
    INK_MUTED,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    SI_WIDTH_IN,
    apply_style,
    save_all,
)

apply_style()

n6 = json.loads((AUTHORITATIVE_DIR / "n6_external_transfer_corrected.json").read_text())
ctx = json.loads((AUTHORITATIVE_DIR / "n21c_primary_external_context_corrected.json").read_text())

all_vals = []
per_model_xy: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for model in MODEL_ORDER:
    strict = n6["results"][model]["strict"]
    y_true = np.array(strict["predictions"]["y_true"], dtype=float)
    y_pred = np.array(strict["predictions"]["y_pred"], dtype=float)
    per_model_xy[model] = (y_true, y_pred)
    all_vals.append(y_true)
    all_vals.append(y_pred)
all_vals = np.concatenate(all_vals)
lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
margin = 0.05 * (hi - lo)
lims = (lo - margin, hi + margin)

fig, axes = plt.subplots(2, 4, figsize=(SI_WIDTH_IN, 3.9), sharex=True, sharey=True)

for ax, model in zip(axes.flat, MODEL_ORDER, strict=True):
    y_true, y_pred = per_model_xy[model]
    color = MODEL_COLORS[model]
    d = ctx["per_model"][model]
    ax.plot(lims, lims, color=INK_MUTED, lw=0.8, ls="--", zorder=1)
    ax.scatter(
        y_true,
        y_pred,
        s=7,
        color=color,
        alpha=0.75,
        edgecolors="#333333",
        linewidths=0.15,
        zorder=2,
    )
    label = MODEL_LABELS[model].replace("elemental-property model", "elemental-\nproperty model")
    ax.set_title(label, fontsize=7.5, pad=10)
    ax.text(
        0.5,
        1.0,
        f"MAE = {d['mae']:.2f}, $R^2$ = {d['r2']:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=INK_MUTED,
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(lw=0.5)
    ax.set_box_aspect(1)
    ax.tick_params(labelsize=7)

fig.supxlabel(r"Observed log$_{10}$($\sigma_{\mathrm{RT}}$ / S cm$^{-1}$)", fontsize=8, y=0.015)
fig.supylabel(r"Predicted log$_{10}$($\sigma_{\mathrm{RT}}$ / S cm$^{-1}$)", fontsize=8, x=0.01)

fig.tight_layout(rect=(0.02, 0.035, 1, 1), h_pad=2.2, w_pad=0.6)
save_all(fig, "si_fig2_parity", si=True)
plt.close(fig)
