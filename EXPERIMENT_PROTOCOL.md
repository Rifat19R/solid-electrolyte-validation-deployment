# EXPERIMENT PROTOCOL — v2.0 (ACTIVE)

> **Supersedes v1.0** (preserved verbatim below, §v1.0). See
> `RESEARCH_CONTRACT.md` v2.0 for why this supersession exists. Unlike
> v1.0, which was written before its study ran, v2.0 is written after the
> v2.0 study (OBELiX→Hargreaves CV-regime comparison) had already been
> executed, to reconcile the frozen protocol with what was actually
> pre-specified and run. This is disclosed explicitly rather than
> presented as prospective: the pre-specification itself happened inside
> the study's own SHA-256 hash-locked protocol artifact
> (`artifacts/frozen/external_transfer_lock.json`) and its written specs
> (`reports/remediation/specs/`), not in this document, which did not exist at
> that time. Both are internal, git-committed records written and SHA-256
> hash-locked before the computations they govern; neither is a public
> preregistration. This document's role going forward is to govern any further
> work on this study (revisions, reviewer-requested analyses, replication
> attempts), not to retroactively claim it governed what already ran.

## Primary endpoint

External MAE on `log10(sigma_RT / S cm^-1)`, evaluated once on the frozen
strict Hargreaves external cohort (n=204, composition-disjoint from the
OBELiX development population), under the pretest-lock protocol in
`artifacts/frozen/external_transfer_lock.json` (run_count=1, authorized=true, enforced
by a `RuntimeError` guard against re-invocation).

## Primary comparison statistics

For each internal CV regime (A/B/C, primary; D/E, post hoc leakage
controls), measured against the frozen external result:
- median |internal MAE − external MAE| ("median gap"),
- Spearman rho and Kendall tau (internal-vs-external model-ranking
  agreement),
- top-model selection regret (external-MAE cost of picking the
  internally-best model).
All four are row-level bootstrap 95% CIs (2,000 resamples, fixed seed)
over the external cohort; a source/DOI-cluster bootstrap sensitivity is
also reported (Supplementary Information Part I).

## Secondary metrics

- Constant-baseline skill scores (development-population mean/median).
- Applicability-domain (AD) distance vs. absolute external error
  (Spearman correlation) and risk-coverage curves summarized by AURC,
  benchmarked against a family-novelty flag, a kernel-density source
  score, and cross-model prediction dispersion.
- Deployment-scenario stratified MAE (seen-family / unseen glass-
  glass-ceramic / unseen crystalline, the last reported descriptively
  only — below the pre-declared n>=20 inference threshold).

No formal uncertainty calibration (coverage, interval width) is claimed
or reported for this study; risk scores are explicitly framed as
risk-stratification, not calibrated uncertainty, per
`RESEARCH_CONTRACT.md` v2.0's forbidden-claims list.

## Required model ladder

Eight composition-only models, all developed on the 449-row OBELiX
development population: ridge regression, random forest, extremely
randomized trees, XGBoost, CatBoost, a shallow MLP on Magpie features, a
shallow MLP on a pooled elemental-property vector ("pooled
elemental-property model" — a fixed, non-learned per-element lookup, not
a pretrained embedding despite its artifact/variable name), and a
composition-attention transformer model.

## Required internal-validation regimes

- **A**: random five-fold CV (pre-specified primary).
- **B**: Family-GroupKFold, grouped by harmonized chemical family
  (pre-specified primary).
- **C**: leave-one-major-family-out, families with >=15 rows
  (pre-specified primary).
- **D**: formula-GroupKFold, exact-duplicate-formula leakage control
  (post hoc).
- **E**: chemical-system-GroupKFold, pymatgen element-set-level leakage
  control (post hoc, added for this submission).

Nested hyperparameter tuning: every outer fold, in every regime, retunes
independently on an 80/20 inner split of that fold's own outer-train
partition only; the final external-transfer model for each of the 8
architectures uses a single frozen configuration selected once via the
original train/validation split, refit on the full development
population across 5 fixed seeds before the one-shot external prediction.

## Required ablations / sensitivity controls

- Six-model sensitivity excluding the two extrapolation-prone
  architectures (ridge, shallow MLP).
- Training-size-matched random-holdout control, isolating family-exclusion
  distribution shift from training-set-size loss.
- Common-support restriction (regimes A/B scored on LOMFO's 396-row
  support only).
- Source/DOI-cluster bootstrap (85 clusters) as an alternative to
  row-level resampling.
- `__MISSING__`-family-label sensitivity (excluded vs. singleton-grouped).
- DOI-source-disjoint (n=153) and relaxed (n=239) external-cohort
  sensitivities; fixed 20-30 degC window sensitivity.
- Cross-database label-compatibility audit (development rows only for
  the primary reported version; a historical variant that also read
  OBELiX test-split labels is retained only as a disclosed SI sensitivity
  check, not as a primary result).

No shuffled-label control is run in this study: v1.0's shuffled-transport-
label control tested whether a pretraining transfer's gain reflected real
transport signal rather than the structure of the pretraining task itself.
This study has no pretraining-transfer step for that control to apply to;
its analogous leak-free-ness checks are regimes D/E above, which test
whether exact-formula or chemical-system duplication (not label shuffling)
drives the primary result. This is a deliberate scope difference, not an
omitted requirement — see `RESEARCH_CONTRACT.md` v2.0 core contribution 1.

## Low-data learning curve

**NOT_RUN.** No 25/50/75/100% label-fraction learning curve was
pre-specified or executed for this study. This is an explicit gap against
v1.0's protocol template, not a silently dropped requirement; if a future
revision of this study needs one, it must be pre-specified and hashed
before execution, per this project's standing rules.

## OOD-adjacent evaluation

LOMFO (regime C) and the seen/unseen-glass/unseen-crystalline deployment
stratification serve the role v1.0's OOD-holdout requirement described,
using chemistry/material-class-defined splits, not model-error-defined
ones. The unseen-crystalline subset (n=19) was pre-declared below the
inference threshold and is reported descriptively only, not as a tested
OOD claim.

## Hyperparameter tuning

Optuna, fixed trial budget and seed, development population's train split
for fitting / validation split for the tuning objective only. The
resulting configuration is frozen before the one-shot external-transfer
fit. Never tuned against the external (Hargreaves) labels at any point.

## Seeds and statistics

Final external-transfer fits use 5 fixed seeds (point prediction = mean
across seeds). Regime A (random CV) is additionally repeated over 10
deterministic fold seeds as a sensitivity check, with regimes B/C held
fixed. All comparative statistics report bootstrap 95% CIs, not single-
seed point estimates alone, in the primary tables.

## Stopping rule

Stop adding post hoc sensitivity analyses once the required ladder,
regimes, ablations, deployment stratification, and risk-stratification
comparison are complete and every analysis added beyond the pre-specified
protocol has its own written, hashed spec (`reports/remediation/specs/`). Do not
add models or regimes merely to enlarge the comparison.

## Success criteria (as pre-specified in the protocol's own hash-locked
artifact, restated here for governance)

H1 (family-aware CV better estimates external error magnitude) is
supported if the family-aware regimes' median |gap| is smaller than
random CV's with a bootstrap CI excluding zero. H2 (family-aware CV
better estimates external ranking) is supported if family-aware regimes'
Spearman/Kendall rank agreement exceeds random CV's with a CI excluding
zero. Neither criterion is met unconditionally on the primary full-cohort
comparison (see `DEFINITION_OF_DONE.md` v2.0 Gate 5 for the actual,
evidence-cited outcome) — a negative or conditional result on H1/H2 is an
acceptable, valid outcome of this protocol, not a protocol failure.

---

## §v1.0 (SUPERSEDED — historical record, does not govern this repository)

v1.0 described the required model ladder (M0-M7, including GNN-from-
scratch, generic-materials-pretrained, percolation/BVSE/DFT-NEB-
pretrained, and shuffled-transport-label variants), low-data learning
curve, and OOD evaluation for the GNN transport-pretraining project. That
project closed `VALID_NEGATIVE_RESULT` at commit `8f37c89`
(`reports/phase2_9_final_scientific_closure.md`); its full original text
is preserved in git history (e.g. `git show 8f37c89:EXPERIMENT_PROTOCOL.md`)
and is not reproduced here. It no longer governs current work.
