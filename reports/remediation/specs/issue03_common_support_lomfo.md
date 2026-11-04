# Spec — Issue 03: common-support LOMFO sensitivity

**Question.** LOMFO (regime C) only scores the 396 development-population
rows belonging to a "major family" (n≥15, `LOMFO_MIN_FAMILY_N`); regimes A
and B score all 449 rows. Does restricting A and B's *internal* pooled MAE
to the identical 396-row common support change the regime-utility
statistics (median |MAE gap|, Spearman ρ, Kendall τ, regret) or the
qualitative H1/H2 verdict, relative to the full-449-row primary result?

**Inputs (frozen, no reruns).**
- `artifacts/npj_n4b_fragments/{model}.json` → `A_random_5fold_cv.per_family_mae`
  and `B_family_group_kfold.per_family_mae` (each `{family: {mae, n}}`,
  already computed, per model, per regime).
- `artifacts/npj_n4b_nested_internal_validation.json` → `major_families`
  (the frozen ≥15-row family set defining the 396-row common support;
  reused unchanged, not redefined).
- `artifacts/npj_n7b_regime_utility_nested.json` → external MAE per model
  (row-paired external residuals from `npj_n15_external_uncertainty.json`
  for bootstrap CIs, matching N7b's own method).
- Regime C's own `pooled_mae_over_major_family_rows` is already exactly the
  common-support quantity; used unchanged.

**Method.** For each model and each of regimes A, B: common-support pooled
MAE = Σ(per_family_mae[f].mae × per_family_mae[f].n for f in major_families)
/ Σ(per_family_mae[f].n for f in major_families) — a weighted recombination
of already-frozen per-family aggregates, not a rerun. Assert the resulting
denominator equals 396 for every model (sanity check). Feed the three
common-support internal-MAE dicts (A, B, C-unchanged) into the existing
`regime_summary`/`bootstrap_regime_pair_delta_ci` functions
(`src/sse_transfer/evaluation/npj_regime_utility.py`), unmodified, against
the unchanged external MAE/residuals.

**Decision rule.** Report both the full-449 and common-396 versions of all
four statistics for A and B side by side. State explicitly, per statistic,
whether the common-support version's 95% bootstrap CI relative to A changes
sign or crosses/excludes zero differently than the full-support version
(the same "excludes zero" criterion already used for the primary result).
No result is discarded or re-run based on which answer is more favorable to
either hypothesis.

**Expected output.** `artifacts/acs_issue03_common_support_lomfo.json`
(point estimates + bootstrap CIs for A/B at both supports, C unchanged) and
`reports/acs/issue03_common_support_lomfo.md`.
