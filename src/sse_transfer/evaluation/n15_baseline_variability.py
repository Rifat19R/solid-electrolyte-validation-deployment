"""N15 analysis 6 -- naive-baseline and raw-measurement-variability sensitivity
(post-review sensitivity analysis, NOT part of the N5 pretest lock).

Two independent, model-free questions:
(a) how much of a model's apparent external skill a trivial constant
    predictor already captures;
(b) how much repeated raw Hargreaves RT-window measurements of the SAME
    nominal composition disagree with each other, reported as empirical
    measurement/temperature-protocol heterogeneity -- explicitly NOT an
    irreducible noise floor (that framing is never used here or in the
    driver script/report).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sse_transfer.evaluation.metrics import regression_metrics

N15_BASELINE_VARIABILITY_VERSION = "1.0"


def constant_predictor_metrics(y_true: np.ndarray, const_value: float) -> dict[str, float]:
    """Regression metrics for a predictor that always outputs `const_value`."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.full_like(y_true, fill_value=const_value, dtype=float)
    return regression_metrics(y_true, y_pred)


def within_composition_variability(
    rt_window_valid_ledger: pd.DataFrame,
    *,
    formula_column: str = "reduced_formula",
    value_column: str = "log_target",
    min_repeats: int = 2,
) -> dict[str, Any]:
    """Per-formula repeat-measurement dispersion over an already RT-window
    +valid-target-filtered ledger slice (caller applies those filters; this
    function only groups and computes dispersion, so it stays a pure
    function of an already-prepared DataFrame)."""
    groups = rt_window_valid_ledger.groupby(formula_column)[value_column]
    counts = groups.size()
    repeat_formulas = counts[counts >= min_repeats].index
    if len(repeat_formulas) == 0:
        return {
            "n_groups_with_repeats": 0,
            "min_repeats_threshold": min_repeats,
            "n_repeats_median": float("nan"),
            "n_repeats_q25": float("nan"),
            "n_repeats_q75": float("nan"),
            "range_median": float("nan"),
            "range_q25": float("nan"),
            "range_q75": float("nan"),
            "pooled_mad": float("nan"),
            "per_group": {},
        }

    n_repeats: list[int] = []
    ranges: list[float] = []
    pooled_centered_abs: list[float] = []
    per_group: dict[str, dict[str, float]] = {}
    for formula in repeat_formulas:
        values = groups.get_group(formula).to_numpy(dtype=float)
        n = int(len(values))
        rng = float(values.max() - values.min())
        median = float(np.median(values))
        centered_abs = np.abs(values - median)
        n_repeats.append(n)
        ranges.append(rng)
        pooled_centered_abs.extend(centered_abs.tolist())
        per_group[str(formula)] = {"n_repeats": n, "range": rng, "median": median}

    n_repeats_arr = np.array(n_repeats, dtype=float)
    ranges_arr = np.array(ranges, dtype=float)
    q25_n, q75_n = np.quantile(n_repeats_arr, [0.25, 0.75])
    q25_r, q75_r = np.quantile(ranges_arr, [0.25, 0.75])
    return {
        "n_groups_with_repeats": int(len(repeat_formulas)),
        "min_repeats_threshold": min_repeats,
        "n_repeats_median": float(np.median(n_repeats_arr)),
        "n_repeats_q25": float(q25_n),
        "n_repeats_q75": float(q75_n),
        "range_median": float(np.median(ranges_arr)),
        "range_q25": float(q25_r),
        "range_q75": float(q75_r),
        "pooled_mad": float(np.median(pooled_centered_abs)),
        "per_group": per_group,
        "interpretation_label": (
            "empirical measurement/temperature-protocol/synthesis heterogeneity across "
            "independent literature reports of a nominally identical composition -- NOT an "
            "irreducible noise floor"
        ),
    }
