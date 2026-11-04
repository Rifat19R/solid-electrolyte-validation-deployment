"""Shared plotting palette/style for manuscript M2 figures.

Colors are the validated default categorical/status/sequential palette from the
project's `dataviz` skill (`references/palette.md`), light-mode values only
(manuscript figures are print/PDF, not a themed UI). No new colors are invented
here; slot order and hex values are copied verbatim from that reference so the
CVD-safety validation already run on that palette applies unchanged.
"""

from __future__ import annotations

import matplotlib as mpl

# Categorical palette, fixed order (never cycled/re-ordered per chart).
CATEGORICAL = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
CATEGORICAL_ORDER = ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]

# Six mechanisms + baseline: fixed color assignment, used identically in every
# figure that shows more than one mechanism, so a mechanism's color is stable
# across F2-F6 (identity, not rank).
MECHANISM_COLORS = {
    "M1_structural_descriptors": CATEGORICAL["blue"],
    "M2_pretrained_embeddings": CATEGORICAL["orange"],
    "M3_transport_transfer": CATEGORICAL["aqua"],
    "M4_censor_aware_learning": CATEGORICAL["yellow"],
    "M5_family_aware_reweighting": CATEGORICAL["magenta"],
    "M6_family_free_reweighting": CATEGORICAL["violet"],
    "baseline": "#52514e",  # secondary ink, not a categorical slot (not a mechanism)
}

MECHANISM_LABELS = {
    "M1_structural_descriptors": "1. Structural descriptors",
    "M2_pretrained_embeddings": "2. CHGNet/MACE embeddings",
    "M3_transport_transfer": "3. Chemsys transport transfer",
    "M4_censor_aware_learning": "4. Censor-aware learning",
    "M5_family_aware_reweighting": "5. Family-feature + family-reweighting strategy",
    "M6_family_free_reweighting": "6. Family-free reweighting/Group-DRO",
    "baseline": "Retained baseline (Family-free)",
}

# Status palette (fixed, never themed) -- used ONLY for GO/NO-GO/PROMOTE verdict
# encoding, never reused as a categorical series color.
STATUS = {
    "go": "#0ca30c",
    "no_go": "#d03b3b",
    "partial": "#fab219",
    "promote": "#0ca30c",
    "retain_baseline": "#52514e",
}

# Sequential single-hue ramp (blue), light -> dark, for magnitude encodings.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Diverging pair (blue <-> red), neutral midpoint gray.
DIVERGING = {"low": "#2a78d6", "mid": "#f0efec", "high": "#e34948"}

# Chart chrome & ink (light mode only).
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

FIGURE_DPI = 600


def apply_style() -> None:
    """Apply consistent, colorblind-safe, print-quality matplotlib rcParams."""
    mpl.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
                "Helvetica",
                "sans-serif",
            ],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": BASELINE_AXIS,
            "axes.labelcolor": INK_PRIMARY,
            "axes.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.grid": True,
            "axes.axisbelow": True,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK_PRIMARY,
        }
    )


def verdict_color(verdict: str) -> str:
    v = verdict.strip().upper()
    if v in ("GO", "PROMOTE"):
        return STATUS["go"]
    if v in ("NO-GO", "NO_GO"):
        return STATUS["no_go"]
    if v in ("PARTIAL",):
        return STATUS["partial"]
    if v in ("RETAIN_BASELINE",):
        return STATUS["retain_baseline"]
    return INK_MUTED
