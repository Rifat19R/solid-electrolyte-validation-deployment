# Spec — Issue 09: `__MISSING__` family sensitivity

**Question.** 18 of 449 development-population rows have no Family label
and are filled with the `__MISSING__` sentinel
(`src/sse_transfer/evaluation/npj_internal_validation.py::MISSING_FAMILY`),
which is then treated as its own group for Family-GroupKFold (regime B)
and, being ≥15 rows, as a "major family" for LOMFO (regime C). Does the
primary regime-utility result depend on this treatment?

**Variant A — exclude `__MISSING__` rows from evaluation (cheap, no
rerun).** Recompute regimes A/B/C's pooled internal MAE per model over the
431 non-`__MISSING__` rows only, by dropping the `__MISSING__` entry from
each model's already-frozen `per_family_mae` dict
(`artifacts/npj_n4b_fragments/{model}.json`) and taking the
row-count-weighted mean of the rest (same recombination method as Issue
03's common-support control, reusing `major_families`/`per_family_mae`,
no retraining). For LOMFO, `__MISSING__` is simply removed from the set of
major families the statistic sums over. Feed the resulting three internal-
MAE dicts through the unchanged `regime_summary`/
`bootstrap_regime_pair_delta_ci` functions against the unchanged external
MAE.

**Variant B — treat each `__MISSING__` row as its own singleton group
(real rerun, nested tuning).** Rebuild Family-GroupKFold (regime B) with a
modified grouping vector: every `__MISSING__` row gets a unique synthetic
per-row group ID instead of sharing one `"__MISSING__"` label; all other
rows keep their real family label unchanged. Rerun the identical nested
per-fold Optuna tuning protocol used throughout (`n4b`'s own inner
80/20-split convention, same seed 20260921, same trial budgets) — this is
a genuine recomputation, not a recombination, because GroupKFold's fold
assignment for every OTHER row can shift once 18 rows stop being one group.
LOMFO (regime C) is only trivially affected (`__MISSING__` drops out of
`major_families` automatically once no group has ≥15 rows under this
scheme — no separate rerun needed for C beyond re-deriving
`major_families` and reusing already-frozen per-family MAEs for the
remaining major families, since LOMFO's other fold assignments are
untouched by this change).

**Inputs.** `artifacts/npj_n4b_fragments/*.json` (Variant A); OBELiX dev
population load path + `family_group_kfold_splits` from
`npj_internal_validation.py`, and the nested-tuning driver pattern used by
`scripts/npj/n4b_nested_internal_validation.py` (Variant B, regime B only —
regime A is untouched by family-grouping changes and is reused unmodified
from N4b).

**Decision rule.** Report Variant A and B alongside the primary (18-row-
included) result for all four regime-utility statistics. State explicitly
whether either variant changes the H1/H2 verdict or the qualitative "random
CV wins" characterization. Compute in the order written above (A before B);
do not skip A even if B looks more decisive, and do not rerun either after
seeing external comparison results.

**Expected output.** `artifacts/acs_issue09_missing_family_sensitivity.json`
and `reports/acs/issue09_missing_family_sensitivity.md`.
