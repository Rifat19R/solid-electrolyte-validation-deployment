# RESEARCH CONTRACT — v2.0 (ACTIVE)

> **Supersedes v1.0** (preserved verbatim below, §v1.0). v2.0 was written
> 2026-10-01 under explicit author authorization (`.planning_unlocked`) to
> reconcile this contract with the manuscript actually being submitted —
> see `CHANGELOG.md` for the change-control entry this supersession
> required. v1.0 described a different, earlier research direction that
> was itself closed out (not silently abandoned) before this pivot; see
> §v1.0 for that closure's status. Nothing about v1.0's closure is being
> revised here — this document is being brought into agreement with work
> that already happened, not retroactively changing what was done.

## Project

Validation–Deployment Alignment Shapes External Generalization in Lithium
Solid-Electrolyte Machine Learning

## Primary scientific question

Does chemistry-family-aware cross-validation (CV) anticipate a materials
ML model's performance on an independently curated external dataset
better than random CV — for experimentally measured lithium
solid-electrolyte ionic conductivity specifically — or is that a premise
that has gone untested against real external data rather than a
within-dataset holdout?

## Primary hypotheses

- **H1**: Family-aware CV (Family-GroupKFold / leave-one-major-family-out)
  gives a better estimate of external error magnitude than random CV.
- **H2**: Family-aware CV gives better agreement with the external
  model-ranking than random CV.

Both are tested on a one-shot, hash-locked transfer from an OBELiX-derived
development population (n=449) to a composition-disjoint external cohort
curated from Hargreaves et al.'s conductivity database (n=204, strict).

## Allowed claims

- Family-aware CV can be compared against random CV as candidate proxies
  for external deployment performance, using an independently curated
  external dataset as ground truth for that comparison.
- The comparison's outcome may be deployment-conditional (e.g. differ
  between within-family and unseen-material-class transfer) rather than a
  single global answer; a conditional or mixed result is a valid finding,
  not evidence of a failed study.
- Composition-only descriptors are a legitimate representation for this
  comparison because they are the only input computable identically in
  both source datasets — not because they are claimed to be sufficient to
  predict conductivity in general.
- Prediction-time risk/novelty scores (applicability-domain distance,
  family-novelty flag, cross-model dispersion) may be compared as
  post hoc risk-stratification signals, without claiming any of them is a
  calibrated uncertainty estimate.

## Forbidden claims

- A benchmark comparison on this dataset pairing proves family-aware CV
  is generally superior or inferior to random CV for materials ML.
- Random-split (or any single-split) performance alone establishes
  generalization to deployment.
- Any risk/novelty/applicability-domain score is a calibrated uncertainty
  estimate unless calibration is explicitly demonstrated.
- The composition-only representation used here fully captures what
  determines experimental conductivity (structure, disorder, synthesis,
  microstructure, and measurement protocol are explicitly out of scope of
  what these features can represent — see Limitations in the manuscript).
- A result on this one dataset pairing, one target property, and eight
  architectures generalizes to other properties, datasets, or model
  families without further testing.
- Migration barrier is equivalent to ionic conductivity, or any other
  atomistic-simulation quantity is equated with, or substitutes for,
  experimental conductivity (this study does not compute or use any such
  quantity — composition-only, no structure, no simulation).
- New DFT/NEB/AIMD/MD calculations are required or permitted (unchanged
  from v1.0 — this project still does not run atomistic simulations of
  any kind).

## Physical / methodological limitations to acknowledge

- Composition-only features discard structural and disorder information
  that OBELiX itself carries; they are used only because they are the
  representation available identically across both databases.
- Neither database's headline conductivity is a verified, uniform bulk or
  total quantity; cross-database and same-formula label disagreement is
  measured and reported, not hidden.
- The crystalline-unseen-family subset (n=19) is below the pre-declared
  inference threshold; conclusions about unseen crystalline chemistry
  remain explicitly unresolved, not extrapolated from the glass/
  glass-ceramic subset.
- Rank statistics over 8 (or 6) models are coarse; directional, not
  high-precision, evidence.
- The row-level bootstrap treats external rows as independent; a
  source/DOI-cluster bootstrap sensitivity shows this understates
  uncertainty for some (not all) statistics — reported, not concealed.

## Core contributions

1. A direct cross-dataset test of whether chemistry-family-aware CV
   anticipates external performance better than random CV for
   experimental Li-ion conductivity, including a duplicate-composition
   leakage control (formula-GroupKFold and chemical-system-GroupKFold).
2. Identification of an architecture-specific extrapolation failure mode
   under family exclusion (ridge, shallow MLP), mechanistically isolated
   from training-set-size loss via a training-size-matched control.
3. A deployment-scenario stratification (seen-family vs. unseen
   glass/glass-ceramic vs. unseen crystalline) showing the CV-regime
   comparison's answer is conditional on the kind of novelty deployed
   against, not a single global ranking.
4. A benchmark of a composition-space applicability-domain score against
   a family-novelty flag and cross-model prediction dispersion as
   prediction-time external-risk signals.

## Out of scope for this study

- Any new DFT/NEB/AIMD/MD (unchanged from v1.0).
- Structure-aware or graph-based modeling (composition-only by design;
  not attempted, not claimed superior or inferior to structure-aware
  approaches).
- Experimental synthesis or electrochemical measurement.
- Candidate-discovery or new-electrolyte claims.
- Formal uncertainty calibration (risk-stratification signals are
  compared; none is claimed calibrated).
- Generative inverse design, active learning, Pareto optimization.

## Change control

This file is frozen after each version's planning is locked. Further
scientific-scope changes require: (1) explicit human approval, (2)
rationale in `CHANGELOG.md`, (3) a version bump, (4) review of
leakage/evaluation consequences, exactly as this v2.0 supersession itself
was done. The Claude Code hook blocks automated edits to this file (and
`EXPERIMENT_PROTOCOL.md`, `DEFINITION_OF_DONE.md`) unless
`.planning_unlocked` exists.

---

## §v1.0 (SUPERSEDED — historical record, does not govern this repository)

v1.0 (frozen after the original planning pass, first committed as part of
this repository's early history) described a different project: *"From
Migration Barriers to Experimental Conductivity: Disorder-Aware
Multi-Fidelity Transport Pretraining and Graph Transfer Learning for
Lithium Solid Electrolytes"* — a GNN encoder pretrained on percolation/
BVSE/DFT-NEB migration-barrier supervision, transferred to OBELiX
experimental conductivity via a disorder-aware graph representation and
an M0–M7 model ladder.

**That project was not abandoned mid-stream without resolution.** It was
carried to a documented, gated closure:

- Final status: **VALID_NEGATIVE_RESULT** (`reports/phase2_9_final_scientific_closure.md`,
  commit `8f37c89`) — the primary hypothesis (transport-specific
  pretraining improves data-efficient, trustworthy experimental
  conductivity prediction over scratch/generic-pretraining baselines) was
  tested under the full required ladder, ablations, and controls and was
  **not supported**, by the project's own success criteria, written
  in the planning-frozen v1.0 governance documents before the tests
  were computed (an internal, git-committed record, not a public
  preregistration).
- Independently re-audited: four-layer validation, commit `36ec9e1`/
  `2743fbc`, verdict `VALIDATED_WITH_LIMITATIONS` (0 critical findings,
  1 high finding fixed, 3 medium, 3 low — see `reports/validation/`).

The full original v1.0 text (primary question, hypothesis, allowed/
forbidden claims, core contributions, out-of-scope list) is preserved
exactly as written in git history — see any commit before this
supersession, e.g. `git show 8f37c89:RESEARCH_CONTRACT.md` — and is not
reproduced a second time here to avoid two "live-looking" contracts in
one file. It no longer governs current work; v2.0 above does.

**Why the pivot to v2.0's study happened without a contemporaneous
change-control entry**: unknown to this author at the time of writing
v2.0 — no session record in this repository documents the decision to
start the composition-only CV-regime study as a distinct follow-on
project rather than as a new phase of the v1.0 contract. This gap is
disclosed, not concealed, and is the reason this supersession exists:
to bring the paper trail into agreement with the actual submitted work,
rather than leave `DEFINITION_OF_DONE.md` grading a manuscript against a
contract for a different study.
