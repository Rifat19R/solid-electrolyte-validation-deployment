# Spec — Issue 10: formula-balanced external-estimand sensitivity

**Question.** The primary external metrics (Table 2, Fig. 2, Fig. 3) weight
the strict Hargreaves cohort by measurement row. If several rows shared one
reduced formula, that formula would implicitly carry more weight than a
formula reported only once, which could inflate or deflate external MAE and
model ranking for reasons unrelated to how good a model's predictions are.
Does re-weighting the strict cohort to give every unique reduced formula
equal total influence change (1) external model ordering, (2) which
validation regime (A/B/C/D) best calibrates external MAE magnitude, (3)
rank-agreement conclusions, (4) selection-regret conclusions, or (5) the
paper's deployment-conditional central claim?

**Premise check (run first, before any sensitivity statistic; this is a
confirmation of frozen-artifact structure, not a new result).**
`data/processed/npj_external_cohort/hargreaves_strict.csv` is built by
`aggregate_by_formula()` (`src/sse_transfer/data/hargreaves_cohort.py`),
which groups the 213 pre-aggregation retained rows by `reduced_formula` and
emits the row-count-weighted median target per group — i.e. the strict
cohort is already constructed as exactly one row per unique reduced formula
before N6 ever reads it. This script asserts, rather than assumes:
(a) `len(hargreaves_strict.csv) == 204`; (b) `nunique(reduced_formula) ==
204` (zero repeated-formula groups) in that file; (c) for every one of the 8
models' frozen N6 predictions, the `reduced_formula` list has length 204,
zero duplicates, and the identical formula *set* as the cohort file. If any
of (a)-(c) fails, the script raises rather than proceeding — the sensitivity
math below assumes singleton formula groups only where that premise
actually holds, and must not silently fall back to a different computation.

**Expected outcome given the premise above.** Because the row key and the
formula key already coincide 1:1 in this frozen cohort, the formula-balanced
estimand is mathematically identical to the existing row-weighted estimand
(a group of size 1 contributes its own single residual with weight 1,
exactly as it already does under row weighting). This is stated here, before
computation, as the a priori expected result given the construction above —
not as a license to skip computing it. The script computes it in full via
an explicit group-then-average implementation (not a shortcut that assumes
the premise), and reports the verified numerical identity (or, if the
premise check instead fails, reports whatever non-trivial reweighting
actually results) rather than asserting equality without having computed
it.

**Inputs (frozen, no reruns).**
- `data/processed/npj_external_cohort/hargreaves_strict.csv` (204 rows, the
  `reduced_formula` row key).
- `artifacts/npj_n6_external_transfer.json` (already-frozen one-shot
  per-model, per-row predictions/residuals on the strict cohort; `run_count
  == 1`, `authorized == true`).
- `artifacts/npj_n4b_nested_internal_validation.json` (regimes A/B/C
  internal pooled MAE per model, nested-tuning-corrected).
- `artifacts/npj_n20b_formula_group_kfold.json` (regime D, formula-
  GroupKFold internal pooled MAE per model; post hoc leakage control).
- `artifacts/npj_n15_stable_model_sensitivity.json` (six-model subset
  definition: excludes ridge, shallow_mlp).
- `artifacts/npj_n7b_regime_utility_nested.json` (existing row-weighted
  regime-utility reference, for side-by-side comparison only).

No model is refit, no new inference is run, and the frozen 113-row OBELiX
official test split is not opened by this analysis.

**Definitions.**
For model `m` and unique reduced formula `f` with member rows `i ∈ f`:

    E_f,m = mean_{i∈f} |y_i − ŷ_{i,m}|
    MAE_formula,m = mean_f E_f,m

This gives every unique formula equal total weight regardless of how many
measurement rows (here, always exactly one) it contributes. Implemented as
an explicit `groupby(reduced_formula).mean()` followed by an unweighted mean
over the resulting per-formula values — never as a row-level mean re-used
under a different name — so the implementation is a genuine test of the
grouping logic, not a tautology.

**Formula-cluster bootstrap.** Resample unique formula groups (not rows)
with replacement, 2,000 iterations, fixed seed `20260921` (this project's
standing bootstrap-seed convention). Within each resampled draw, recompute
`MAE_formula,m` the same way (mean over the resampled formula instances of
that formula's own mean |residual|). Report 95% percentile CIs per model
and, by substituting formula-group resampling for row-level resampling in
`bootstrap_regime_pair_delta_ci`'s statistic logic (a new function reusing
the existing `cluster_bootstrap_regime_pair_delta_ci` machinery with cluster
id = formula, left unmodified for its existing DOI-cluster caller), the four
regime-utility pairwise deltas (B−A, C−A, D−A).

**Method (Phase 2 of the task).**
A. Verify existing per-model row-weighted MAE from N6 exactly, then compute
   formula-balanced MAE and formula-balanced ranking for all 8 models.
B. Recompute regime-utility statistics (median |gap|, Spearman ρ, Kendall
   τ, regret) for regimes A, B, C (pre-specified) and D (post hoc,
   formula-GroupKFold leakage control) against the formula-balanced external
   MAE.
C. Repeat B for the existing six-model stable-architecture subset.
D. Formula-cluster bootstrap (2,000 reps, seed 20260921): external MAE CIs
   and the regime-utility pairwise-delta CIs (B−A, C−A, D−A), reported
   alongside the existing row-level bootstrap CIs for direct comparison.
E. Compare A–D against the existing row-weighted results and answer the
   five decision-question items above explicitly; do not suppress a
   material change if the premise check in fact fails.

**Decision rule.** If the premise check confirms zero repeated-formula
groups (the expected case given `aggregate_by_formula`'s construction), the
formula-balanced point estimates for A and B above must equal the existing
row-weighted values to within floating-point tolerance (1e-9); this is
reported as a positive confirmatory finding that closes the reviewer
concern for this specific frozen cohort, not treated as "nothing to report."
The formula-cluster bootstrap CIs (D) are the only part of this analysis
that can differ numerically from the existing row-level bootstrap even under
a confirmed-trivial premise, because `rng.choice` over a degenerate
single-row-per-cluster population is a different pseudorandom draw sequence
than `rng.integers` over individual rows (same distributional target,
different draws) — any difference here is sampling noise, not evidence of a
population-weighting effect, and is reported as such.

**Expected output.**
`artifacts/acs_issue10_formula_balanced_sensitivity.json` and
`reports/acs/issue10_formula_balanced_sensitivity.md`.
