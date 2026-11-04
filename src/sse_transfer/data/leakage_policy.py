"""Frozen leakage-exclusion policy (Phase 0C).

Implements the hierarchy fixed by CLAUDE.md's Phase 0C instructions and
DATA_CONTRACT.md's "Canonical identity and overlap audit":

1. Final-test protection: any pretraining record whose material is an
   EXACT_ID_MATCH or EXACT_STRUCTURE_MATCH against an OBELiX final-test
   material must be excluded from pretraining/training.
2. Validation protection: the same rule applies to OBELiX validation-split
   materials, because validation is used for tuning.
3. Probable structure matches are never auto-excluded. Each one is reviewed
   deterministically (see `REVIEWED_PROBABLE_MATCHES` below) using
   canonical-identity evidence (anonymized formula, space-group number,
   volume-per-atom), and the decision is recorded, not inferred at run time.
4. Composition-only matches are never excluded: composition overlap alone is
   not structural leakage per RESEARCH_CONTRACT.md.

This module only classifies and computes exclusion sets; it never touches
raw data. bvse_hops and dft_neb are excluded by `material_id` cascade (every
row/edge belonging to an excluded material_id is excluded), because both are
edge-level datasets sharing the material_id namespace with bvel_percolation
(see reports/results/leakage/pretraining_id_overlap.json).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MatchType = Literal["EXACT_STRUCTURE_MATCH", "PROBABLE_STRUCTURE_MATCH"]
ProtectionTier = Literal["final_test_protection", "validation_protection"]

#: Deterministic, human-reviewed decisions for the three PROBABLE_STRUCTURE_MATCH
#: pairs found by reports/results/leakage/obelix_vs_pretraining_structure_overlap.json
#: (obelix query_id -> matched bvel_percolation material_id). Review method: compare
#: reduced formula, anonymized formula, space-group number, and volume-per-atom
#: computed via sse_transfer.data.canonical_id on both structures.
#:
#! query_id, matched_id -> (decision, rationale)
REVIEWED_PROBABLE_MATCHES: dict[tuple[str, str], tuple[str, str]] = {
    ("4kt", "mp-29582"): (
        "EXCLUDE",
        "Same reduced formula (Li2CrCl4), same anonymized formula (AB2C4), same "
        "space group (#15), and closely agreeing volume/atom (20.39 vs 21.14 A^3/atom, "
        "3.6% apart -- consistent with OBELiX's upstream coordinate-randomization "
        "noise plus a cell-choice/supercell difference, not a distinct material). "
        "Treated as equivalent to an exact structure match for exclusion purposes.",
    ),
    ("mdp", "mp-641703"): (
        "EXCLUDE",
        "Same reduced formula (Li7P3S11), same anonymized formula (A3B7C11), same "
        "space group (#2), identical site count (42=42), and closely agreeing "
        "volume/atom (19.75 vs 20.88 A^3/atom, 5.7% apart). Treated as equivalent "
        "to an exact structure match for exclusion purposes. This query_id is in "
        "the OBELiX official test split, so final-test protection applies.",
    ),
    ("x2i", "mp-1540429"): (
        "RETAIN",
        "Same reduced and anonymized formula (LiBiO2 / ABC2) but DIFFERENT space "
        "group (#72 vs #15) and different site count (32 vs 16 atoms/cell). A "
        "space-group mismatch at matching stoichiometry is decisive evidence of a "
        "distinct polymorph, not the same material blurred by CIF noise. Retained "
        "in pretraining; composition overlap alone is not structural leakage per "
        "RESEARCH_CONTRACT.md.",
    ),
}


@dataclass(frozen=True)
class LedgerEntry:
    original_dataset: str
    record_id: str
    material_id: str
    match_type: MatchType
    protection_tier: ProtectionTier
    matched_obelix_ids: list[str]
    exclusion_reason: str

    def as_dict(self) -> dict:
        return {
            "original_dataset": self.original_dataset,
            "record_id": self.record_id,
            "material_id": self.material_id,
            "match_type": self.match_type,
            "protection_tier": self.protection_tier,
            "matched_obelix_ids": self.matched_obelix_ids,
            "exclusion_reason": self.exclusion_reason,
        }


def excluded_material_ids(
    structure_overlap_matches: list[dict],
    *,
    protected_obelix_ids: dict[str, ProtectionTier],
) -> dict[str, list[tuple[str, MatchType, ProtectionTier]]]:
    """Return {material_id: [(obelix_query_id, match_type, protection_tier), ...]}
    for every pretraining material that must be excluded, given the frozen policy.

    `protected_obelix_ids` maps every OBELiX id that requires protection (test
    split ids -> "final_test_protection", validation split ids ->
    "validation_protection") to its tier.
    """
    result: dict[str, list[tuple[str, MatchType, ProtectionTier]]] = {}

    for match in structure_overlap_matches:
        query_id = match["query_id"]
        matched_id = match["matched_id"]
        classification = match["classification"]
        tier = protected_obelix_ids.get(query_id)
        if tier is None:
            continue

        if classification == "EXACT_STRUCTURE_MATCH":
            result.setdefault(matched_id, []).append((query_id, "EXACT_STRUCTURE_MATCH", tier))
        elif classification == "PROBABLE_STRUCTURE_MATCH":
            review = REVIEWED_PROBABLE_MATCHES.get((query_id, matched_id))
            if review is None:
                raise ValueError(
                    f"PROBABLE_STRUCTURE_MATCH ({query_id}, {matched_id}) has no recorded "
                    "review decision in REVIEWED_PROBABLE_MATCHES. Refusing to guess; "
                    "add a reviewed entry before freezing datasets."
                )
            decision, _rationale = review
            if decision == "EXCLUDE":
                result.setdefault(matched_id, []).append(
                    (query_id, "PROBABLE_STRUCTURE_MATCH", tier)
                )
        # COMPOSITION_ONLY_MATCH and NO_MATCH are never excluded.

    return result


def build_ledger(
    excluded: dict[str, list[tuple[str, MatchType, ProtectionTier]]],
    *,
    bvel_material_ids: set[str],
    bvse_edge_index: dict[str, list[str]],
    dft_edge_index: dict[str, list[str]],
) -> list[LedgerEntry]:
    """Expand excluded material_ids into per-record ledger entries across the
    three pretraining datasets. `bvse_edge_index` / `dft_edge_index` map
    material_id -> list of edge_ids in that dataset.
    """
    entries: list[LedgerEntry] = []
    for material_id, causes in sorted(excluded.items()):
        matched_obelix_ids = sorted({c[0] for c in causes})
        # Highest-severity cause wins for the recorded match_type/tier when several
        # OBELiX ids independently implicate the same material.
        match_type: MatchType = (
            "EXACT_STRUCTURE_MATCH"
            if any(c[1] == "EXACT_STRUCTURE_MATCH" for c in causes)
            else "PROBABLE_STRUCTURE_MATCH"
        )
        tier: ProtectionTier = (
            "final_test_protection"
            if any(c[2] == "final_test_protection" for c in causes)
            else "validation_protection"
        )
        reason = (
            f"{match_type} against OBELiX id(s) {matched_obelix_ids} "
            f"({tier}); see REVIEWED_PROBABLE_MATCHES for probable-match rationale."
        )

        if material_id in bvel_material_ids:
            entries.append(
                LedgerEntry(
                    "bvel_percolation",
                    material_id,
                    material_id,
                    match_type,
                    tier,
                    matched_obelix_ids,
                    reason,
                )
            )
        for edge_id in bvse_edge_index.get(material_id, []):
            entries.append(
                LedgerEntry(
                    "bvse_hops",
                    edge_id,
                    material_id,
                    match_type,
                    tier,
                    matched_obelix_ids,
                    reason,
                )
            )
        for edge_id in dft_edge_index.get(material_id, []):
            entries.append(
                LedgerEntry(
                    "dft_neb",
                    edge_id,
                    material_id,
                    match_type,
                    tier,
                    matched_obelix_ids,
                    reason,
                )
            )
    return entries
