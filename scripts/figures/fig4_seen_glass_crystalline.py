"""Fig. 4 (central result) -- validation-regime utility refined by material
class: seen-family and unseen-glass median gaps by regime (a), and per-model
external MAE on the under-powered unseen-crystalline subset (b, descriptive).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from shared import (
    AUTHORITATIVE_DIR,
    INK_PRIMARY,
    MAIN_WIDTH_IN,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    REGIME_COLORS,
    REGIME_LABELS,
    REGIME_ORDER,
    apply_style,
    panel_letter,
    save_all,
)

apply_style()

glass_split = json.loads((AUTHORITATIVE_DIR / "phase1_glass_split_corrected.json").read_text())
glass_cis = json.loads(
    (AUTHORITATIVE_DIR / "phase2_glass_subset_regime_cis_corrected.json").read_text()
)

regime_utility = glass_split["regime_utility_by_subset"]
per_model_subset = glass_split["per_model_subset_metrics"]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(MAIN_WIDTH_IN, 3.1), width_ratios=[1.0, 1.05])

x = np.arange(2)
width = 0.26
subset_labels = ["Seen family\n(n = 138)", "Unseen glass\n(n = 36)"]
subsets = ["SEEN", "UNSEEN_GLASS"]

for i, regime in enumerate(REGIME_ORDER):
    vals = [regime_utility[s][regime]["median_abs_mae_gap"] for s in subsets]
    ax_a.bar(
        x + (i - 1) * width,
        vals,
        width,
        color=REGIME_COLORS[regime],
        edgecolor="white",
        linewidth=0.4,
        label=REGIME_LABELS[regime],
    )

b_a_ci = glass_cis["pairwise_bootstrap_ci"]["B_minus_A"]["median_abs_mae_gap"]
c_a_ci = glass_cis["pairwise_bootstrap_ci"]["C_minus_A"]["median_abs_mae_gap"]
ax_a.text(
    1.0 + 1.5 * width,
    max(regime_utility["UNSEEN_GLASS"][r]["median_abs_mae_gap"] for r in REGIME_ORDER) + 0.06,
    f"B − A: {b_a_ci['point_delta']:.2f} [{b_a_ci['ci_low']:.2f}, {b_a_ci['ci_high']:.2f}]\n"
    f"C − A: {c_a_ci['point_delta']:.2f} [{c_a_ci['ci_low']:.2f}, {c_a_ci['ci_high']:.2f}]\n"
    "both 95% CIs exclude zero",
    fontsize=7,
    color=INK_PRIMARY,
    ha="right",
    va="bottom",
    linespacing=1.3,
)
ax_a.set_xticks(x)
ax_a.set_xticklabels(subset_labels)
ax_a.tick_params(axis="x", length=0)
ax_a.set_ylabel("Median |internal − external| MAE gap\n(lower is better)")
ax_a.legend(
    loc="upper left",
    ncol=1,
    fontsize=7.2,
    borderaxespad=0.2,
    handlelength=1.0,
)
ax_a.set_ylim(0, 2.0)
ax_a.grid(axis="y")
panel_letter(ax_a, "a", x=-0.2, y=1.02)

crystalline_mae = [per_model_subset["UNSEEN_CRYSTALLINE"][m]["mae"] for m in MODEL_ORDER]
y = np.arange(len(MODEL_ORDER))
ax_b.barh(y, crystalline_mae, color=[MODEL_COLORS[m] for m in MODEL_ORDER], height=0.7)
ax_b.set_yticks(y)
ax_b.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
ax_b.tick_params(axis="y", length=0)
ax_b.invert_yaxis()
ax_b.set_xlabel(r"External MAE, log$_{10}$($\sigma_{\mathrm{RT}}$ / S cm$^{-1}$)")
ax_b.set_title("Unseen crystalline (n = 19), descriptive only", loc="left", fontsize=8)
ax_b.set_xlim(0, 5.0)
ax_b.grid(axis="x")
ax_b.spines["left"].set_visible(False)
# Letter sits left of the long model labels, level with panel a's letter.
panel_letter(ax_b, "b", x=-0.86, y=1.02)

fig.tight_layout(w_pad=1.5)
save_all(fig, "fig4_seen_glass_crystalline")
plt.close(fig)
