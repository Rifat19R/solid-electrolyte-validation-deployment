# Spec — Issue 05: strictly fold-local robust shift-distance metric

**Question.** `npj_n21e_robust_shift_distance.json`'s own
`threshold_definition` field states the near-degenerate-dimension guard
drops a fold's feature column when "that fold's train-only std < 5% of the
same column's std over the **full 449-row target-free development
population**." That reference population includes the current outer fold's
own held-out (test) rows' feature values — not labels, but features. Does
every per-fold decision instead depend only on that fold's own training
features (or an independently frozen, outer-test-blind threshold), and does
recomputing it that way change the qualitative conclusions (which regime is
closest to external by Wasserstein distance; whether the 94,559.5 → 26.65
correction still holds)?

**Inputs (frozen).** Same raw per-fold standardized-distance pipeline
already used by `scripts/npj/n21e_robust_shift_distance.py`
(`artifacts/npj_n18_nn_distance_forensic.json`'s per-fold feature matrices /
`artifacts/npj_n15_deployment_shift.json`'s distance computation
machinery) — no target `y` is read at any point in this analysis (already
true of the existing version and preserved here).

**Method.** Two independent fold-local reference constructions, both
avoiding any use of the current fold's own outer-test rows:
1. **Train-only reference**: for outer fold *k*, the reference standard
   deviation used in the 5% guard is that fold's own training-partition std
   (already used as the numerator's comparator; the fix is changing the
   *denominator* to also be train-only, per-fold, rather than the shared
   449-row population).
2. **Frozen population-of-other-folds reference** (secondary robustness
   check): a per-fold denominator computed from the union of every OTHER
   fold's training data, still never touching fold *k*'s own held-out rows.
Recompute the standardized nearest-neighbor distances and the Wasserstein
distances (regime vs. strict/source-disjoint external cohorts) under
construction (1) as the primary fix; report construction (2) only as a
secondary check that the qualitative conclusion is not an artifact of which
fold-local scheme is chosen.

**Decision rule.** Report original (population-wide denominator), fixed
(train-only denominator), and secondary-check numbers side by side for:
max/q90/median standardized distance per regime, dimensions dropped per
fold, and the Wasserstein distance + closest-regime-to-external verdict.
State explicitly whether the closest-regime verdict (currently LOMFO) or
any qualitative claim in the manuscript changes under the fix. If it does
not change, say so plainly rather than treating the fix as a non-event —
the point of doing it right is the audit trail, whether or not the
headline number moves.

**Expected output.**
`artifacts/acs_issue05_fold_local_shift_distance.json`,
`reports/acs/issue05_fold_local_shift_distance.md`, and a regenerated
Supplementary Fig. 1 in `manuscript_acs/figures/` if the qualitative
picture changes materially (kept identical to the existing figure, with an
updated caption/SI text, if it does not).
