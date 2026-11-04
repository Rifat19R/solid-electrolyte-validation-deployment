"""N15 analysis 5 -- AURC over a finer risk-coverage grid (post-review
sensitivity analysis, NOT part of the N5 pretest lock).

Consumes the output of `uncertainty.risk_coverage.
risk_coverage_mae_at_retained_fraction` (reused unmodified; that function
already accepts an arbitrary `retained_fractions` tuple, so no change to it
was needed for a finer grid). This module adds only the AURC summary
statistic N8 did not compute.
"""

from __future__ import annotations

from typing import Any

import numpy as np

N15_AURC_VERSION = "1.0"


def aurc_from_risk_coverage(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Area under the risk-coverage curve (MAE vs. retained coverage
    fraction), trapezoidal integration over `rows` (each a dict with
    `retained_fraction` and `mae`, as returned by
    `risk_coverage_mae_at_retained_fraction`), normalized by the covered
    coverage span so the result is directly comparable in scale to a
    per-coverage-point MAE. Lower AURC = better selective-prediction
    behavior integrated across the whole grid.
    """
    if len(rows) < 2:
        raise ValueError("aurc_from_risk_coverage: need at least 2 coverage points to integrate")
    fractions = np.array([r["retained_fraction"] for r in rows], dtype=float)
    maes = np.array([r["mae"] for r in rows], dtype=float)
    order = np.argsort(fractions)
    fractions = fractions[order]
    maes = maes[order]
    span = float(fractions[-1] - fractions[0])
    if span <= 0:
        raise ValueError("aurc_from_risk_coverage: degenerate coverage span")
    area = float(np.trapezoid(maes, fractions))
    return {
        "aurc": area / span,
        "coverage_min": float(fractions[0]),
        "coverage_max": float(fractions[-1]),
        "n_points": int(len(rows)),
    }
