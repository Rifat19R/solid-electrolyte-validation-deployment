# Spec — Issue 04: training-size-matched random control

**Question.** Regime B (Family-GroupKFold)'s fold 0 removes NASICON plus
five smaller families from training (213/449 rows, 47.4% of the pool),
confounding two effects: (1) training-set size reduction, and (2) a
distribution shift from excluding an entire chemical family. Is the
catastrophic MAE blowup ridge/shallow-MLP show under regime B attributable
to size loss alone, independent of which rows/family were excluded?

**Method.** For each of regime B's 5 outer folds (fixed test-partition
sizes, from `family_group_kfold_splits(seed=20260921)`), construct a plain
random row-level holdout of the IDENTICAL test-set size using
`size_matched_random_splits` (`src/sse_transfer/evaluation/npj_internal_validation.py`),
for 3 deterministic control seeds (20260930, 20260931, 20260932). For each
(seed, fold, model), run the identical nested per-fold tuning pipeline N4b
used (inner 80/20 split of that fold's own outer-train partition, then
refit on the full outer-train partition) — full nested retuning, not a
reuse of regime B's frozen hyperparameters, so this control is judged under
the same protocol as the reference regime. Compute pooled MAE per model per
seed; compare against regime B's actual per-model pooled MAE (from N4b) and
regime A's (random CV) pooled MAE.

**Scope decision.** Matched-size controls are built for regime B only, not
regime C (LOMFO), to bound compute (~2–3h on this machine, 4 cores). This
is a disclosed scope reduction — N4b/N15 already diagnose LOMFO's
family-exclusion mechanism qualitatively, and B is the regime with the
catastrophic (100×+) blowup this control is specifically meant to isolate.

**Decision rule.** If a model's matched-size-random pooled MAE across the 3
seeds stays within the same order of magnitude as regime A and far below
regime B's actual blowup MAE, the degradation is attributable to
family-exclusion distribution shift, not size loss alone. If the
matched-size control also blows up to a similar order of magnitude as
regime B, size loss alone is a real contributor and the manuscript's
"architecture-specific extrapolation failure... driven... by family
exclusion" framing needs qualifying.

**Process note (disclosed, not hidden).** This exact computation was
already in flight, started by an out-of-scope background agent, when this
written spec was produced — see `_quarantine_unauthorized_fork_output/README.md`
for the incident. The agent's own pre-written spec
(`artifacts/acs_remediation_spec.json::issue_04_training_size_matched_control`)
was reviewed and found methodologically identical to this one; this
document is an independent restatement, not a copy made after seeing
results. The compute was allowed to finish rather than be discarded
(explicit user instruction), and its output will be independently
re-verified against this spec (at minimum, one model/seed recomputed from
scratch and compared) before being cited in the manuscript — see
`reports/acs/issue04_training_size_matched_control.md` for that
verification once written.

**Expected output.**
`artifacts/acs_issue04_size_matched_control.json`,
`reports/acs/issue04_training_size_matched_control.md`.
