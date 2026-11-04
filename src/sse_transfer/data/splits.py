"""Deterministic, leakage-aware validation split construction.

See EXPERIMENT_PROTOCOL.md ("Primary split must preserve the official
benchmark split") and DATA_CONTRACT.md ("Split rules"). This module never
touches the official OBELiX test split; it only carves a validation set out
of the official training rows, keeping every group of rows that share a
`Reduced Composition` together (a composition-family holdout) so that near-
duplicate polymorphs/measurements of the same material cannot appear in both
the training and validation partitions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GROUPING_RULE = (
    "composition_group_holdout_v1: rows are grouped by exact 'Reduced Composition' "
    "string; whole groups (never individual rows) are shuffled with a seeded "
    "numpy Generator and greedily assigned to validation until the cumulative "
    "validation row count first reaches the target fraction of the input rows; "
    "all remaining groups form the post-split training set."
)


@dataclass(frozen=True)
class GroupHoldoutSplit:
    seed: int
    grouping_rule: str
    target_fraction: float
    train_ids: list[str]
    validation_ids: list[str]
    validation_groups: list[str]
    train_groups: list[str]


def build_composition_group_validation_split(
    train_df: pd.DataFrame,
    *,
    seed: int,
    target_fraction: float = 0.2,
    id_column: str = "ID",
    group_column: str = "Reduced Composition",
) -> GroupHoldoutSplit:
    """Split `train_df` into (post-split train, validation) by whole composition group.

    Deterministic for a fixed (seed, target_fraction, input row set): the group
    order is sorted before shuffling, so results do not depend on input row order.
    """
    if not (0.0 < target_fraction < 1.0):
        raise ValueError(f"target_fraction must be in (0, 1); got {target_fraction}")

    raw_groups = train_df.groupby(group_column)[id_column].apply(list).to_dict()
    groups: dict[str, list[str]] = {str(key): value for key, value in raw_groups.items()}
    sorted_group_keys = sorted(groups.keys())

    rng = np.random.default_rng(seed)
    shuffled_order = rng.permutation(len(sorted_group_keys))
    shuffled_keys = [sorted_group_keys[i] for i in shuffled_order]

    n_total = len(train_df)
    target_n = round(n_total * target_fraction)

    validation_groups: list[str] = []
    validation_ids: list[str] = []
    running = 0
    for key in shuffled_keys:
        if running >= target_n:
            break
        validation_groups.append(key)
        validation_ids.extend(groups[key])
        running += len(groups[key])

    validation_group_set = set(validation_groups)
    train_groups = [k for k in sorted_group_keys if k not in validation_group_set]
    train_ids = [rid for key in train_groups for rid in groups[key]]

    return GroupHoldoutSplit(
        seed=seed,
        grouping_rule=GROUPING_RULE,
        target_fraction=target_fraction,
        train_ids=sorted(train_ids),
        validation_ids=sorted(validation_ids),
        validation_groups=sorted(validation_groups),
        train_groups=sorted(train_groups),
    )
