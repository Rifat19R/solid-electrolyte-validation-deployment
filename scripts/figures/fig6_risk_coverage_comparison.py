"""Fig. 6 -- prediction-time risk stratification on the strict external
cohort: mean AURC across the 8 models for five risk scores (a; four-point
coverage grid 0.25-1.0, lower is better) and the per-model Spearman
association between composition-space AD distance and absolute external
error (b). Built directly from the frozen AD-benchmark-closure artifact; no
new fitting or external inference.

Source: artifacts/authoritative/n21d_ad_benchmark_closure_corrected.json
  summary.<comparator>.mean_aurc_across_8_models
  per_model.<model>.ad_nn_distance.spearman_r
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from shared import (
    AUTHORITATIVE_DIR,
    INK_MUTED,
    INK_SECONDARY,
    MAIN_WIDTH_IN,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    RISK_COLORS,
    apply_style,
    panel_letter,
    save_all,
)

apply_style()

d = json.loads((AUTHORITATIVE_DIR / "n21d_ad_benchmark_closure_corrected.json").read_text())
per_model = d["per_model"]
summary = d["summary"]

COMPARATORS = {
    "ad_nn_distance": "Composition-space AD\n(NN distance)",
    "family_novelty": "Family-seen/unseen flag",
    "cross_model_dispersion": "Cross-model prediction\ndispersion",
    "kde_novelty": "KDE source-density score",
    "random_rejection": "Random rejection\n(baseline)",
}

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(MAIN_WIDTH_IN, 3.0), width_ratios=[1.0, 1.0])

aurc_vals = {k: summary[k]["mean_aurc_across_8_models"] for k in COMPARATORS}
sorted_keys = sorted(COMPARATORS, key=lambda k: aurc_vals[k])
ya = np.arange(len(sorted_keys))
ax_a.barh(
    ya,
    [aurc_vals[k] for k in sorted_keys],
    color=[RISK_COLORS[k] for k in sorted_keys],
    height=0.65,
)
ax_a.set_yticks(ya)
ax_a.set_yticklabels([COMPARATORS[k] for k in sorted_keys], linespacing=1.05)
ax_a.tick_params(axis="y", length=0)
ax_a.invert_yaxis()
ax_a.spines["left"].set_visible(False)
ax_a.set_xlim(0, 1.6)
ax_a.set_xlabel("Mean AURC across 8 models\n(coverage 0.25–1.0; lower is better)")
ax_a.grid(axis="x")
for yi, k in zip(ya, sorted_keys, strict=True):
    ax_a.text(
        aurc_vals[k] + 0.03,
        yi,
        f"{aurc_vals[k]:.2f}",
        va="center",
        fontsize=7,
        color=INK_SECONDARY,
    )
panel_letter(ax_a, "a", x=-0.62, y=1.02)

rho = [per_model[m]["ad_nn_distance"]["spearman_r"] for m in MODEL_ORDER]
yb = np.arange(len(MODEL_ORDER))
ax_b.barh(yb, rho, color=[MODEL_COLORS[m] for m in MODEL_ORDER], height=0.65)
ax_b.set_yticks(yb)
ax_b.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax_b.tick_params(axis="y", length=0)
ax_b.invert_yaxis()
ax_b.spines["left"].set_visible(False)
ax_b.axvline(0, color=INK_MUTED, lw=0.8)
ax_b.set_xlim(0, 0.55)
ax_b.set_xlabel("Spearman ρ, AD distance vs.\nabsolute external error")
ax_b.grid(axis="x")
for yi, r in zip(yb, rho, strict=True):
    ax_b.text(r + 0.01, yi, f"{r:.2f}", va="center", fontsize=7, color=INK_SECONDARY)
panel_letter(ax_b, "b", x=-0.75, y=1.02)

fig.tight_layout(w_pad=1.5)
save_all(fig, "fig6_risk_coverage_comparison")
plt.close(fig)
