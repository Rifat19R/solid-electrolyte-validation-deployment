"""Frozen, human-reviewed OBELiX Family-label harmonization (Phase 2.0R).

DATA_CONTRACT.md "Canonical identity and overlap audit" / CLAUDE.md: "Any
allowed harmonization must be explicit, versioned, tested, and recorded."
Phase 2.0's metadata audit (reports/phase2_0_verdict.md, section 2) found 4
pairs of raw `Family` strings that are the same physical family under a
spelling/casing/pluralization variant -- never previously harmonized, which
silently fragmented the Family-GroupKFold OOD splits used in Phase
1.6/1.8/1.9 (a row spelled "argyrodite" was treated as a different family
from 51 rows spelled "argyrodites").

This module never touches `data/raw/obelix/all.csv` (raw data stays
immutable) and never mutates the frozen split
(`data/processed/v1/obelix_index.json`): it only provides a deterministic,
versioned, tested string transform that callers apply to a copy of the
`Family` column when they want the harmonized view (Phase 2.0V's
family-sensitivity re-evaluation). No file is regenerated on disk -- exactly
like `sse_transfer.features.build_features.build_raw_feature_frame`, this is
recomputed on demand from raw data + a frozen, version-pinned mapping, which
is this project's existing convention for deterministic derived views (only
the actual train/val/test row assignment is materialized and hash-frozen,
because that involves a seeded random decision; a pure string remap does not).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

FAMILY_HARMONIZATION_VERSION = "1.0"

#: Source of the reviewed decision below.
SOURCE_AUDIT = "reports/phase2_0_verdict.md#2-family-label-harmonization-audit"

#: raw Family string -> canonical Family string. Only pairs verified by human
#: review as the same physical family are listed; canonical form chosen by
#: majority vote among the 562 frozen exact-label rows (Phase 2.0R). Every
#: Family value not listed here is already its own canonical form.
#:
#: Reviewed decisions (counts over the 562 exact-label rows):
#:   "Halide" (1 row) is the same family as "halides" (8 rows) -> "halides"
#:   "argyrodite" (1 row) is the same family as "argyrodites" (52 rows)
#:       -> "argyrodites"
#:   "cholorides" (1 row, a transcription typo: extra 'o') is the same
#:       family as "chlorides" (5 rows) -> "chlorides"
#:   "nitride" (1 row) is the same family as "nitrides" (3 rows)
#:       -> "nitrides"
#: "LISICON" vs "NASICON" was investigated and explicitly REJECTED as a
#: harmonization candidate: these are two chemically distinct, well-known SSE
#: families (Lithium vs Sodium Super Ionic CONductor), not a spelling
#: variant, despite a small raw edit distance (2 of 7 characters).
FAMILY_ALIAS_MAP: dict[str, str] = {
    "Halide": "halides",
    "argyrodite": "argyrodites",
    "cholorides": "chlorides",
    "nitride": "nitrides",
}


def canonical_family(family: pd.Series) -> pd.Series:
    """Map each raw Family value to its reviewed canonical form.

    Values not in `FAMILY_ALIAS_MAP` (including NaN/missing) pass through
    unchanged -- never guessed, never invented.
    """
    return family.map(lambda v: FAMILY_ALIAS_MAP.get(v, v) if pd.notna(v) else v)


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def detect_alias_clusters(family_values: pd.Series) -> dict[str, Any]:
    """Cluster distinct raw Family strings that are plausibly the same
    family under a spelling/casing/pluralization variant. Detection only --
    never applies a mapping (that is `FAMILY_ALIAS_MAP`/`canonical_family`,
    a separately reviewed decision).

    Used both by the Phase 2.0 audit (on raw Family, to surface candidates
    for review) and by Phase 2.0V tests/reports (on the post-harmonization
    Family, to confirm the reviewed map actually resolves every cluster it
    found).
    """
    uniques = sorted(family_values.dropna().unique().tolist())
    normalized = {v: v.strip().lower() for v in uniques}
    clusters: list[list[str]] = []
    seen: set[str] = set()
    for i, a in enumerate(uniques):
        if a in seen:
            continue
        na = normalized[a]
        group = [a]
        for b in uniques[i + 1 :]:
            if b in seen:
                continue
            nb = normalized[b]
            plural_match = na != nb and na.rstrip("s") == nb.rstrip("s")
            dist = _levenshtein(na, nb)
            ratio = dist / max(len(na), len(nb))
            # A short pair (e.g. LISICON/NASICON, distance 2 of 7 chars) is a
            # real false positive at a flat distance<=2 threshold: two known,
            # chemically distinct families. Require either a plural match, a
            # single-character edit, or a low edit-distance *ratio* (relative
            # to string length) before flagging -- verified against the known
            # true positives (Halide/halides, argyrodite/argyrodites,
            # chlorides/cholorides, nitride/nitrides) and the known false
            # positive (LISICON/NASICON) below.
            if plural_match or dist <= 1 or (dist == 2 and ratio <= 0.15):
                group.append(b)
        if len(group) > 1:
            clusters.append(sorted(group))
            seen.update(group)
    return {
        "n_distinct_family_labels": len(uniques),
        "all_distinct_labels": uniques,
        "suspected_same_family_clusters": clusters,
        "n_clusters": len(clusters),
        "method": (
            "Pairwise comparison of every distinct raw Family string: flagged as "
            "the same suspected cluster if (a) stripping a single trailing 's' "
            "makes the two case-insensitive strings equal, (b) case-insensitive "
            "Levenshtein distance == 1, or (c) distance == 2 with a length-"
            "normalized ratio <= 0.15 (excludes short, unrelated pairs such as "
            "LISICON vs NASICON, distance 2 of 7 chars, ratio 0.29 -- two "
            "chemically distinct, well-known SSE families, not a spelling "
            "variant). Human review required before any harmonization is "
            "applied; no mapping is applied by this function."
        ),
    }
