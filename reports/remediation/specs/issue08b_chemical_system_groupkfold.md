# Spec — Issue 08b: chemical-system-grouped sensitivity control

**Question.** The exact-formula-disjoint control (regime D) only removes
duplicate *exact* reduced formulas from crossing outer folds. SI Note 15
already found 63 chemical systems (334 rows) share an element set across
more than one distinct stoichiometry (descriptive, not previously acted
on). Does grouping outer folds by chemical system (pymatgen element set)
instead of exact formula change the regime-utility statistics or the
qualitative "random CV wins" result — i.e., is redundancy broader than
exact-formula duplication actually driving any part of the primary result?

**Feasibility (already checked, no new work needed here).** 153 distinct
chemical systems over 449 rows; largest group is 4.5% of the pool — no
single group is large enough to make GroupKFold degenerate the way
Family-GroupKFold's NASICON/perovskite exclusion folds did. Reusing the
existing `chemical_system()` helper
(`scripts/npj/phase1_0_1_predeclare_and_glass_split.py:211`, pymatgen
element set, deterministic) as the grouping key — not a new definition.

**Method.** Exact structural copy of
`scripts/npj/n20b_formula_group_kfold.py` (itself a copy of N4b's regime B)
with the grouping vector changed from canonical reduced formula to
`chemical_system(formula)`. Same `family_group_kfold_splits` generic
splitter (it groups by whatever Series it is given), same nested per-fold
Optuna tuning protocol, same `CV_SEED=20260921`, same 8-model ladder, same
one-model-per-process convention. Internal-only: no Hargreaves file is
opened, N6's frozen external predictions are not touched.

**Decision rule.** Report this regime ("E: chemical-system-GroupKFold") as
a fifth row alongside A–D in the regime-utility table, and its E−A pairwise
deltas alongside B/C/D−A. State explicitly whether E tracks random CV (as D
does) or shows any of B/C's degradation pattern, and whether any model
shows a Family-GroupKFold-style extrapolation blowup under this grouping
(a priori not expected, since no chemical system is large enough to strip
an entire feature region from training, but checked rather than assumed).

**Compute note.** This is a full nested-CV rerun (8 models × 5 outer folds
× inner tuning), comparable cost to N20b (~1.5–2h on this machine). Per the
approved plan's Process Safeguard, run this sequentially, not concurrently
with any other heavy compute job (the in-flight Issue 04 sweep, or Issue 09
Variant B).

**Expected output.**
`artifacts/acs_issue08b_chemical_system_groupkfold.json`,
`artifacts/acs_issue08b_fragments/{model}.json`, and
`reports/acs/issue08b_chemical_system_groupkfold.md`.
