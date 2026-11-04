# DEFINITION OF DONE — v2.0 (ACTIVE)

> **Supersedes v1.0** (preserved below, §v1.0). See `RESEARCH_CONTRACT.md`
> v2.0 for why. Gates below are graded against the study that is actually
> being submitted (OBELiX->Hargreaves CV-regime comparison), using
> `EXPERIMENT_PROTOCOL.md` v2.0 as the applicable protocol. Written
> 2026-10-01 as part of a hard pre-submission audit; every PASS below
> cites the evidence checked during that audit, not an assumption.
> Paths, script names and software-check counts were updated on 2026-11-04
> (final release pass, `CHANGELOG.md`) to the current repository layout;
> no gate verdict changed. Records cited here under `reports/` other than
> `reports/remediation/specs/`, `reports/results/leakage/` and
> `reports/rerun/` are repository-only and not part of the curated release.

The project is DONE only when every mandatory gate below is PASS or an
explicitly disclosed, author-accepted PARTIAL/N/A.

## Gate 0 — Scope integrity
- [x] No new DFT, NEB, AIMD, MD, or other prohibited atomistic calculation
      was run for this study (composition-only; grepped manuscript and SI,
      zero hits for "DFT"/"NEB"/"AIMD"). **PASS**
- [x] Research claims remain inside `RESEARCH_CONTRACT.md` v2.0 (written
      to match this manuscript). **PASS**, as of this supersession —
      v1.0 was violated by omission (undocumented pivot) before it;
      disclosed in `RESEARCH_CONTRACT.md` v2.0 §v1.0 and `CHANGELOG.md`.
- [x] The scope change (v1.0 GNN/transport-pretraining project -> v2.0
      composition-only CV-regime study) is now versioned and logged
      (`CHANGELOG.md`, this file, `RESEARCH_CONTRACT.md` v2.0,
      `.planning_unlocked`). **PASS**

## Gate 1 — Data provenance
- [x] Raw datasets have source/license/retrieval metadata:
      `data/manifests/obelix.json`, `data/manifests/hargreaves_conductivity.json`
      (source DOI, retrieval date, license, SHA-256). **PASS**
- [x] SHA-256 hashes recorded and independently re-verified in this audit
      for the frozen inputs listed in `artifacts/frozen/external_transfer_lock.json`
      (re-verified on every full-repository run by
      `tests/unit/test_path_migration.py::test_every_lock_hashed_path_resolves_to_matching_bytes`;
      PASS on 2026-11-04). **PASS**
- [x] Raw row/structure counts verified: 820 Hargreaves rows / 214 sources
      / 403 unique room-temperature compositions independently reconciled
      to the published abstract (`reports/cross_dataset_study/n1b_403_reconciliation.md`,
      documents an earlier 422-vs-403 bug, found and fixed). **PASS**
- [x] Parser tests pass (part of the full `pytest -q` suite, see Gate 3).
      **PASS**
- [x] Censored labels explicitly handled: dev-only lead population has 0
      inequality-censored rows; a 599-row SI sensitivity explicitly
      excludes 37 censored rows per the data-contract policy. **PASS**
- [x] Disorder/fractional occupancy: not applicable to this study's
      composition-only representation, and the manuscript explicitly
      discloses this as a limitation (discards OBELiX's structural/
      disorder information) rather than silently rounding a structure
      representation. **PASS (disclosed, not violated)**

## Gate 2 — Leakage control
- [x] Within-dataset duplicates audited: 52 duplicated formulas (136 rows,
      30.3%) identified; formula-GroupKFold (regime D) and chemical-
      system-GroupKFold (regime E) controls both run. **PASS**
- [x] Cross-dataset overlap audited: composition- and DOI-source-overlap
      exclusion built into strict/relaxed/DOI-disjoint cohort
      construction (Table 1 of the manuscript). **PASS**
- [x] Pretraining/final-test leakage: N/A for this study (no pretraining
      step); the analogous external-transfer leakage is blocked by the
      one-shot pretest lock below.
- [x] Split hashes frozen before final training:
      `artifacts/frozen/external_transfer_lock.json`, `run_count=1, authorized=true`,
      enforced by a `RuntimeError` guard against re-invocation
      (independently confirmed present and unmodified in this audit).
      **PASS**
- [x] Final test (external Hargreaves) labels were not used for tuning or
      design decisions before the one-shot evaluation; the one documented
      exception (a historical label-compatibility sensitivity that
      briefly accessed OBELiX test-split labels, not the external test)
      is explicitly disclosed in the manuscript, not hidden. **PASS
      (disclosed exception, not a silent violation)**

## Gate 3 — Software quality
(Counts below re-verified in the 2026-11-04 final release pass; they
supersede the 2026-10-01 audit's counts.)
- [x] `pytest -q` (full repository) passes: **729 passed, 0 failed, 0 skipped**. The curated
      release package runs its shipped `tests/` tree with
      `scripts/release/run_package_tests.py`: **488 passed, 0 skipped, 0 failed, 7 deselected; 495 collected** (the
      deselected tests need the withheld Hargreaves data and are listed in
      `TEST_COVERAGE_EXCEPTIONS.md`, `release/` in the repository). **PASS**
- [x] `ruff check .` passes (all checks passed). **PASS**
- [x] `ruff format --check .` passes (421 files already formatted). **PASS**
- [x] `mypy src` passes (58 source files, 0 issues). **PASS**
- [x] Key transformations/metrics have unit tests (`tests/unit/`,
      `tests/scientific/`, `tests/integration/`). **PASS**
- [x] End-to-end checks: `scripts/preflight.py` PASS;
      `scripts/release/verify_anchors.py` 29/29 PASS;
      `scripts/analysis/numeric_audit.py` 48/48 PASS;
      `scripts/manuscript/build_tables.py --check-only` 0 manuscript
      mismatches; `scripts/manuscript/build_submission.py` fail-closed
      document checks PASS; `scripts/release/verify_package.py` PASS.
      **PASS**

## Gate 4 — Baselines and controls
(Redefined for this study's design — v1.0's GNN ladder is N/A here, see
§v1.0 and `EXPERIMENT_PROTOCOL.md` v2.0 for why.)
- [x] Full 8-model composition-only ladder (ridge, random forest, extra
      trees, XGBoost, CatBoost, shallow MLP, pooled elemental-property
      model, composition attention) complete under all 5 CV regimes
      (A-E). **PASS**
- [x] Six-model sensitivity excluding the two extrapolation-prone
      architectures complete. **PASS**
- [x] Training-size-matched control (isolating family-exclusion shift
      from training-size loss) complete. **PASS**
- [x] Duplicate/leakage controls (regimes D, E) complete — serve this
      study's analogous role to v1.0's shuffled-label control (testing
      whether the primary result is an artifact of redundancy rather than
      a genuine regime effect). **PASS**
- N/A: GNN-from-scratch, generic-materials-pretrained GNN,
      transport-pretrained GNN variants, literal shuffled-transport-label
      control — these belong to v1.0's design, not this study's. Marking
      this gate FAIL against v1.0's literal text, as a prior audit pass
      correctly noted, would be grading the wrong study; see
      `RESEARCH_CONTRACT.md` v2.0 for the applicable ladder.

## Gate 5 — Scientific evaluation
- [x] Primary and secondary metrics complete (median gap, Spearman rho,
      Kendall tau, regret, bootstrap CIs, AURC, skill scores). **PASS**
- [x] Multiple seeds complete (5 seeds for final external-transfer refit;
      10 fold seeds for regime-A sensitivity). **PASS**
- [x] Bootstrap CI complete (2,000 resamples, row-level and a
      source/DOI-cluster alternative). **PASS**
- [ ] Low-data learning curve: **NOT_RUN**, explicitly disclosed in
      `EXPERIMENT_PROTOCOL.md` v2.0 as a gap against the v1.0 template,
      not silently dropped. **FAIL (disclosed)**
- [x] OOD-adjacent evaluation complete: LOMFO + seen/unseen-glass/
      unseen-crystalline deployment stratification, with the
      under-powered crystalline subset (n=19) correctly reported
      descriptively only rather than with an unsupported inferential
      claim. **PASS**
- [x] Error analysis by material/structure family complete (Table 4,
      per-model SEEN/glass/crystalline breakdown). **PASS**

## Gate 6 — Uncertainty
- [x] Calibration method: **not attempted, and not claimed** — the
      manuscript deliberately frames composition-space AD distance,
      family-novelty flag, and cross-model dispersion as
      risk-stratification signals, never as calibrated uncertainty,
      consistent with `RESEARCH_CONTRACT.md` v2.0's forbidden-claims
      list. Grepped for "uncertainty...meaningful"-type claims: zero
      hits. **PASS (correctly scoped, not attempted-and-hidden)**
- [x] Coverage/width: N/A, no calibrated interval is claimed.
- [x] Risk-coverage/selective-prediction behavior reported: AURC on a
      16-point and a 4-point coverage grid, benchmarked against 4
      comparator signals (Fig. 6). **PASS**
- [x] Uncertainty/risk failure cases documented: AD score adds little
      beyond family-novelty flag; reported as a limitation, not
      suppressed. **PASS**

## Gate 7 — Reproducibility
- [x] Every reported run has config/seed/commit/hash provenance: hash-
      locked protocol artifact, per-analysis written specs
      (`reports/remediation/specs/`, `HASHES.txt` re-verified), and the
      release package's `MANIFEST_SHA256.txt` (SHA-256 of every shipped
      file, regenerated by `scripts/release/build_package.py` and checked
      by `scripts/release/verify_package.py`). **PASS**
- [x] Final figures/tables regenerable by scripts: confirmed by
      independently re-running the figure/table/DOCX build pipeline in
      this audit (`scripts/manuscript/build_submission.py`,
      `scripts/manuscript/render_contact_sheets.py`,
      `scripts/release/build_package.py`) and re-verifying all QA/numeric
      audits pass against the rebuilt artifacts. **PASS**
- [x] Best-result cherry-picking prevented by the reporting pipeline: the
      one-shot external-transfer guard plus spec files for every post hoc
      analysis, each written and SHA-256 hash-locked before computation
      (internal records; there was no public preregistration). **PASS**
- [ ] Public repository + archival DOI: **required before submission**.
      The code license is resolved as MIT (`LICENSE`), and current
      redistribution terms are documented in `RIGHTS_AND_LICENSING.md`.
      The remaining external action is creation of the public repository
      and archival DOI, followed by insertion of the permanent URL/DOI
      into the submission materials. **PENDING EXTERNAL ACTION**

## Gate 8 — Manuscript readiness
- [x] Main claim (validation-regime utility is deployment-conditional, not
      universally ordered by split severity) is supported by the cited
      evidence, independently traced in this audit (8 headline
      figures/tables checked against `artifacts/authoritative/*.json` source, all
      exact). **PASS**
- [x] Forbidden claims (migration barrier != conductivity, etc.): zero
      violations found on a full-text scan of `manuscript/source/00_main.md`
      and `manuscript/source/01_si.md` against `RESEARCH_CONTRACT.md` v2.0's forbidden-claims
      list. **PASS**
- [x] Limitations of composition-only prediction explicit: Limitations
      section states composition-only features "discard the structural
      and disorder information that OBELiX carries." **PASS**
- [x] Novelty comparison against current literature: a prior Gate 8 audit
      (`reports/literature/gate8_*.md`) predates this study and does not
      cover it. This gap is now partially closed by this audit's own
      live reference verification: 9 of the most load-bearing citations
      (Hargreaves et al. ref 11, Therrien/OBELiX ref 12, Meredig et al.
      ref 32, Durdy et al. ref 34, Witman & Schindler/MatFold ref 35, Tan
      et al./MatUQ ref 43, Ottomano et al. ref 28, Bates/Hastie/Tibshirani
      ref 74, plus a training-era plausibility pass on the remaining ~70
      by an independent subagent) were independently checked against
      Crossref/DOI-resolver/publisher pages — **zero fabricated or
      garbled references found**, including resolving ref 43's apparent
      author-list mismatch (a real title/author-list change between its
      Nov 2025 arXiv preprint and Aug 2026 journal acceptance, not a
      citation error). The full 78-reference list was **not** exhaustively
      individually re-verified against external sources — remaining risk
      is low given this spot-check's 100% hit rate, but a full DOI-by-DOI
      pass before final submission is still recommended, not yet done.
      **PARTIAL (spot-checked clean, not exhaustive)**
- [x] No unsupported material-discovery or experimental claim: confirmed,
      the manuscript makes no candidate-discovery or electrolyte-
      superiority claim anywhere. **PASS**
- [x] README contains a reproduction workflow: present (not re-audited
      line-by-line in this pass; out of scope of this gate's core
      concern, which is scientific/citation integrity).

---

## §v1.0 (SUPERSEDED — historical record, does not govern this repository)

v1.0's gates described the GNN transport-pretraining project's own
requirements (M0-M7 ladder, disorder-aware graph ablations, percolation/
BVSE/DFT-NEB controls, shuffled-transport-label control). That project's
own v1.0 gate audit is `reports/phase2_9_final_scientific_closure.md`
§10 ("DEFINITION_OF_DONE.md gate audit (honest accounting, not all gates
are PASS)"), concluding:

**Final status: VALID_NEGATIVE_RESULT** (commit `8f37c89`). The primary
hypothesis (transport-specific pretraining improves data-efficient,
trustworthy experimental conductivity prediction) was tested under the
full required ladder and was not supported. Independently re-audited,
four-layer validation, commit `36ec9e1`/`2743fbc`,
`VALIDATED_WITH_LIMITATIONS` (0 critical, 1 high fixed, 3 medium, 3 low).

The full original v1.0 gate text is preserved in git history (e.g.
`git show 8f37c89:DEFINITION_OF_DONE.md`) and is not reproduced here.

Final status for v2.0 may be `PASS`, `FAIL`, or `VALID_NEGATIVE_RESULT`
(H1/H2 as stated were not supported on the primary full-cohort
comparison — see manuscript Results/Discussion — which is a valid,
reportable outcome under this project's own standing rule that negative
results are acceptable when the required gates are satisfied, not a
gate failure in itself).
