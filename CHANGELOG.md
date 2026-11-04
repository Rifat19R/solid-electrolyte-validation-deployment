# CHANGELOG

## v2.0.1 — Editorial governance and release-consistency fixes (2026-11-04)
- **Scope**: editorial only. No scientific scope, hypothesis, protocol,
  gate verdict, cohort, model, seed or reported number changed; H1/H2
  remain NOT SUPPORTED. Authorized by the project author (explicit
  instruction in the 2026-11-04 final release session); `.planning_unlocked`
  was created for these edits and removed afterwards.
- `RESEARCH_CONTRACT.md`, `EXPERIMENT_PROTOCOL.md`, `DEFINITION_OF_DONE.md`:
  stale pre-migration paths and script names replaced with current ones
  (`artifacts/frozen/external_transfer_lock.json`,
  `reports/remediation/specs/`, `scripts/analysis/numeric_audit.py`,
  `scripts/manuscript/`, `scripts/release/`); stale test/format/type-check
  counts replaced with counts re-verified in this pass; "pre-registered"
  replaced with accurate wording (internal records written and hash-locked
  before computation; no public preregistration).
- Manuscript sources: analysis-scoped external-transfer wording (SI Notes 7,
  16, 23 reuse the authoritative predictions: seven original frozen models
  plus the corrected pooled elemental-property model); SI Note 27/28 now
  state that pre-correction artifacts are retained in internal Git history
  and not included in the curated release; SI title block matches the main
  manuscript; main text gains a Supporting Information statement; cover
  letter distinguishes the primary hash lock from separately hash-locked
  post hoc specifications.
- `CITATION.cff`: stale `date-released` removed, version 1.0.0 (matching
  `pyproject.toml`); no DOI or URL added.
- Leakage/evaluation consequences reviewed: none (no analysis code, artifact,
  lock or specification changed; only the document builder's fail-closed
  text checks were extended).
- Entries below are historical and keep the paths and journal names that
  were current when they were written.

## v2.0 — Scope supersession: composition-only CV-regime study (2026-10-01)
- **Rationale**: a hard pre-submission audit found that the manuscript
  being prepared for submission to ACS JCIM ("Validation–Deployment Alignment
  Shapes External Generalization in Lithium Solid-Electrolyte Machine
  Learning") answers a different scientific question than v1.0's frozen
  contracts describe (GNN transport-pretraining vs. this study's
  composition-only, CV-regime-vs-external-transfer comparison on
  OBELiX/Hargreaves). The pivot itself happened at some earlier,
  undocumented point (no session record found); this entry is the
  first time it is logged, versioned, and explicitly authorized, per
  `RESEARCH_CONTRACT.md`'s own change-control clause.
- Authorized by: Md. Rifat Khandaker (project author), 2026-10-01, in the
  Claude Code session that ran this audit. `.planning_unlocked` created
  for this edit and deliberately left in place as the audit trail.
- **v1.0's project was not abandoned without resolution** — it closed
  `VALID_NEGATIVE_RESULT` at commit `8f37c89`
  (`reports/phase2_9_final_scientific_closure.md`), independently
  re-audited (`VALIDATED_WITH_LIMITATIONS`, commit `36ec9e1`/`2743fbc`).
  v2.0 does not revise that closure; it only brings the *governing*
  documents into agreement with the *different*, later study that is
  being prepared for submission.
- `RESEARCH_CONTRACT.md`, `EXPERIMENT_PROTOCOL.md`, `DEFINITION_OF_DONE.md`
  each rewritten with a new v2.0 section (the active contract) plus the
  original v1.0 content preserved/pointed-to as historical record (see
  each file's §v1.0). No scientific result, model prediction, metric,
  artifact, or manuscript number was changed by this entry — it is a
  governance/paper-trail fix, not a re-analysis.
- Leakage/evaluation consequences reviewed: none — v2.0's study already
  has its own one-shot external-transfer lock
  (`artifacts/npj_pretest_lock.json`) and leakage controls (regimes D/E),
  independently verified intact in the same audit pass.
- Also in this pass (mechanical/governance, not scientific-content
  changes): fixed 5 stale post-migration paths in
  `manuscript_acs/submission/JCIM_Supplementary_Information.docx` /
  `01_acs_si.md` / `scripts/transport_analysis/` left over from the
  prior npj->ACS namespace migration (commit `96b2f74`); added `LICENSE`
  (MIT; current rights and redistribution terms documented in `RIGHTS_AND_LICENSING.md`); removed the unrelated
  tracked file `preference/CO2_H2_ARC_MOF.docx`; independently
  spot-verified 9 of the manuscript's most load-bearing references
  against Crossref/DOI resolution (zero fabrications found).

## v1.0 — Planning freeze
- Locked project to lithium solid-state electrolytes.
- No new DFT/NEB/AIMD/MD.
- Corrected framing to multi-fidelity microscopic transport pretraining plus cross-property/cross-scale conductivity transfer.
- Made disorder-aware representation mandatory.
- Added censoring, leakage, low-data, OOD, uncertainty, and shuffled-label controls.
- Removed active learning and Pareto optimization from core claims.
