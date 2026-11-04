"""Fold-construction helpers for the cross-dataset study's N4
internal validation regimes (Random 5-fold CV, canonical Family-GroupKFold,
leave-one-major-family-out).

Pure functions of already-loaded arrays/series only (CLAUDE.md "Keep
evaluation independent from training"): nothing here fits a model or reads
a raw file. Reuses the exact family-canonicalization and
"major family" (`n >= 15`) conventions already established and frozen in
the old project (`sse_transfer.data.family_harmonization.canonical_family`;
`LOMFO_MIN_FAMILY_N = 15` from `scripts/phase2_2_ood_robustness.py` and
reused identically in phase2_3/2_4b/2_5/2_6a) rather than inventing new
ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from sse_transfer.data.family_harmonization import canonical_family

#: Same sentinel used by scripts/phase1_6/1_8/1_9_*.py for an unlabeled
#: Family value -- an explicit category, never a silent drop or impute
#: (DATA_CONTRACT.md "Disorder" / CLAUDE.md N9 instruction).
MISSING_FAMILY = "__MISSING__"

#: Same threshold used identically across scripts/phase2_2/2_3/2_4b/2_5/2_6a
#: (`LOMFO_MIN_FAMILY_N = 15`): a family with at least this many rows in the
#: pool being split is "major" and eligible for a leave-one-family-out fold.
LOMFO_MIN_FAMILY_N = 15

#: Same outer-fold count used identically across the old project's Family-
#: GroupKFold usage (scripts/phase1_9_composition_ood.py, phase2_2, ...).
N_OUTER_SPLITS = 5


def canonicalize_and_fill_family(family: pd.Series) -> pd.Series:
    """Harmonize known spelling/alias variants, then fill any remaining
    unlabeled value with `MISSING_FAMILY` -- mirrors
    `scripts/phase1_9_composition_ood.py::_attach_composition_columns`'s
    `_family_filled` construction (canonicalize first, then fill) exactly."""
    return canonical_family(family).fillna(MISSING_FAMILY).astype(str)


def major_family_counts(
    family_filled: pd.Series, min_n: int = LOMFO_MIN_FAMILY_N
) -> dict[str, int]:
    """`{family: count}` for every family with `count >= min_n`, sorted by
    family name for determinism. `MISSING_FAMILY` is eligible like any other
    label -- it is not excluded automatically."""
    counts = family_filled.value_counts()
    major = counts[counts >= min_n]
    return {str(k): int(v) for k, v in sorted(major.items())}


def random_kfold_splits(
    n_rows: int, *, n_splits: int = N_OUTER_SPLITS, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Regime A: plain row-level K-fold, no grouping."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_rows)))


def family_group_kfold_splits(
    family_filled: pd.Series, *, n_splits: int = N_OUTER_SPLITS, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Regime B: canonical Family-GroupKFold -- every row of a given family
    falls entirely in one fold, matching the old project's
    `build_canonical_family_folds` convention
    (`GroupKFold(n_splits=5, shuffle=True, random_state=seed)`)."""
    groups = family_filled.to_numpy()
    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(gkf.split(np.arange(len(family_filled)), groups=groups))


def leave_one_major_family_out_splits(
    family_filled: pd.Series, major_families: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Regime C: `{family: (train_idx, test_idx)}` for every family in
    `major_families` -- `test_idx` is every row of that family, `train_idx`
    is every other row in the pool (all other families, including minor
    ones and `MISSING_FAMILY`, stay in training)."""
    fam = family_filled.to_numpy()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in major_families:
        test_idx = np.flatnonzero(fam == name)
        train_idx = np.flatnonzero(fam != name)
        if len(test_idx) == 0:
            raise ValueError(f"leave_one_major_family_out_splits: family {name!r} has 0 rows")
        if not set(train_idx.tolist()).isdisjoint(set(test_idx.tolist())):
            raise RuntimeError(f"leave_one_major_family_out_splits: bad partition for {name!r}")
        out[name] = (train_idx, test_idx)
    return out


def inner_early_stop_split(
    n_rows: int, *, stop_fraction: float = 0.2, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Carves a fold's *training* partition into (inner_train, inner_stop)
    purely to monitor early stopping for the torch candidates -- never uses
    the fold's held-out test rows for this (that would leak the evaluation
    partition into the epoch-selection decision). A plain seeded row-level
    split, not family-grouped: this is an internal training-mechanics detail
    of a single fold's fit, not itself one of the three validation regimes
    being compared."""
    if not (0.0 < stop_fraction < 1.0):
        raise ValueError(f"stop_fraction must be in (0, 1); got {stop_fraction}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    n_stop = max(1, round(n_rows * stop_fraction))
    inner_stop = np.sort(order[:n_stop])
    inner_train = np.sort(order[n_stop:])
    return inner_train, inner_stop


def size_matched_random_splits(
    n_rows: int,
    reference_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Post-hoc training-size-matched control (reviewer remediation Issue
    04): for each fold in `reference_splits` (e.g. Family-GroupKFold's 5
    folds), build a plain random row-level split of the IDENTICAL test-set
    size, ignoring family membership entirely. Used to test whether a
    reference regime's fold-specific degradation is attributable to
    training-size loss alone, independent of which rows/families were
    excluded. A single `seed` produces one full deterministic set of
    matched-size folds (every row still appears in exactly one test fold,
    same partition invariant as the reference regime); different seeds are
    independent random partitions of matching sizes, not resamples of a
    fixed split, to give a genuine seed-to-seed size-matched range."""
    test_sizes = [len(test_idx) for _, test_idx in reference_splits]
    if sum(test_sizes) != n_rows:
        raise ValueError(
            f"size_matched_random_splits: reference test folds sum to {sum(test_sizes)}, "
            f"expected {n_rows} (every row must appear in exactly one reference test fold)"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for size in test_sizes:
        test_idx = np.sort(order[start : start + size])
        train_idx = np.sort(np.setdiff1d(np.arange(n_rows), test_idx, assume_unique=True))
        splits.append((train_idx, test_idx))
        start += size
    return splits
