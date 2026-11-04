# Spec — Issue 06: DOI/source-cluster bootstrap

**Question.** The existing bootstrap CIs (external MAE, regime-utility
deltas) resample external rows independently, understating dependence when
multiple retained rows/formulas trace to the same literature source (or to
sources that co-report overlapping formulas). Do 95% CIs computed with a
source-cluster (not row-level) bootstrap unit materially widen the existing
CIs or change which comparisons "exclude zero"?

**Inputs (frozen, no reruns).**
- `data/processed/npj_external_cohort/hargreaves_overlap_ledger.csv`,
  filtered to `in_rt_window & valid_target & exclusion_reason=="RETAINED" &
  overlap_classification=="NO_MATCH"` (213 pre-aggregation rows, 204 unique
  `reduced_formula`, 91 unique `source` DOIs — reproduces SI Note 4's
  counts exactly; used as a sanity assertion before proceeding).
- Already-frozen per-model, per-row predictions/residuals on the 204-row
  strict cohort (`artifacts/npj_n6_external_transfer.json` /
  `npj_n15_external_uncertainty.json` — whichever stores row-aligned
  residuals; row order must match `hargreaves_strict.csv`'s
  `reduced_formula` order).

**Cluster construction (deterministic, frozen before any CI is computed).**
Build a bipartite graph of {204 formulas} ∪ {91 DOIs}, edge if that DOI
reports that formula among the 213 retained pre-aggregation rows. Take
connected components as clusters (a formula co-reported with another via a
shared DOI, or transitively via a chain of shared DOIs, lands in the same
cluster) — this is the correct dependency unit, since two different
formulas sharing a literature source are not independent measurements even
though they are different compositions. Report the resulting cluster-count
and max-cluster-size as a sanity check before computing anything
downstream.

**Method.** Resample CLUSTERS with replacement (not rows) to reconstruct a
bootstrap sample of formulas, 2000 iterations, same fixed seed convention
(20260921) as every other bootstrap in this study. Recompute: (a) external
MAE per model with percentile 95% CI; (b) the A/B/C-vs-A regime-utility
deltas' CIs (median |gap|, ρ, τ, regret), reusing
`bootstrap_regime_pair_delta_ci`'s statistic logic but with
cluster-level resampling substituted for its existing row-level resampling
(a new function, not a modification of the existing row-level one — the
existing function and its callers are left unchanged for the primary
analysis).

**Decision rule.** Report row-level vs. cluster-level CIs side by side for
every statistic. State explicitly, per statistic, whether cluster-level
resampling widens the CI (expected direction) and whether any comparison
that excluded zero under row-level bootstrap now includes zero under
cluster-level bootstrap (a real robustness downgrade, reported honestly,
not smoothed over).

**Expected output.** `artifacts/acs_issue06_doi_cluster_bootstrap.json` and
`reports/acs/issue06_doi_cluster_bootstrap.md`.
