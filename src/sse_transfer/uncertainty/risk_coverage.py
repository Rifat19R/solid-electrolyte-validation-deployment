"""Risk-coverage / selective-prediction metric for the cross-dataset study
(N8). Kept in its own module rather than added to
`uncertainty.applicability_domain` because that file is one of the old
(closed) project's `CODE_HASH_FILES` -- its frozen Phase 2.7/2.8 pretest
lock re-verifies that file's SHA-256 on every run, so it must never change
(CLAUDE.md "Do not alter frozen old results/splits/artifacts"). This module
is new, unhashed by any old lock, and depends only on numpy -- callers pass
in an `uncertainty.applicability_domain.nearest_neighbor_ad_score` output
directly, no import from that module is needed here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

RISK_COVERAGE_VERSION = "1.0"


def risk_coverage_mae_at_retained_fraction(
    ad_scores: np.ndarray,
    abs_residual: np.ndarray,
    retained_fractions: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
) -> list[dict[str, Any]]:
    """Pooled MAE on the most in-domain `fraction` of rows (lowest AD score
    first), for each fraction in `retained_fractions` -- a selective-
    prediction / risk-coverage curve (N8). `fraction=1.0` always equals
    the ordinary pooled MAE over every row. Coverage is deterministic
    ascending-AD-score order (stable sort, ties broken by original row
    order), matching `applicability_domain.ad_score_quantile_error_table`'s
    convention."""
    ad_scores = np.asarray(ad_scores, dtype=float)
    abs_residual = np.asarray(abs_residual, dtype=float)
    if ad_scores.shape != abs_residual.shape:
        raise ValueError("risk_coverage_mae_at_retained_fraction: shape mismatch")
    n = len(ad_scores)
    if n == 0:
        raise ValueError("risk_coverage_mae_at_retained_fraction: empty input")
    for f in retained_fractions:
        if not (0.0 < f <= 1.0):
            raise ValueError(f"retained_fractions must be in (0, 1]; got {f}")

    order = np.argsort(ad_scores, kind="stable")
    sorted_err = abs_residual[order]
    out: list[dict[str, Any]] = []
    for fraction in retained_fractions:
        k = max(1, round(n * fraction))
        retained = sorted_err[:k]
        out.append(
            {
                "retained_fraction": fraction,
                "n_retained": int(k),
                "mae": float(np.mean(retained)),
            }
        )
    return out
