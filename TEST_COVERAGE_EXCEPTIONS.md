# Test coverage exceptions — public reproducibility package

The package runs its whole shipped `tests/` tree with
`python scripts/release/run_package_tests.py`. Nothing is ignored, nothing is
expected to skip (any skip fails the run), and `pytest --collect-only` has no
collection errors. Two kinds of exception exist, and both are listed here
test by test. `tests/unit/test_package_test_policy.py` checks this list
against the runner.

## 1. Restricted-data tests (category A): deselected only while the data are withheld

The Hargreaves et al. database is used under research-use terms that do not
grant redistribution (`RIGHTS_AND_LICENSING.md`). The package withholds the
raw file, the overlap ledger and the two cohort CSVs, and redacts the per-row
external targets in the external-transfer records
(`provenance/restricted_data.json`). The seven tests below read that material
and are deselected while it is withheld. After a lawful download of the raw
file, `python scripts/release/restricted_data.py restore` rebuilds and
hash-verifies every withheld or redacted file. The runner then detects the
restored state and runs every test with nothing deselected. All seven pass in
the full repository.

| test | needs | counterpart that runs in the package |
|---|---|---|
| `tests/unit/test_hargreaves_cohort.py::test_build_cohorts_gate_n2_passes_and_relaxed_is_superset_size` | raw Hargreaves database | `test_build_cohorts_synthetic_ledger_and_strict_relaxed_accounting` (synthetic 7-row fixture) |
| `tests/unit/test_hargreaves_cohort.py::test_build_cohorts_strict_cohort_has_no_overlap_with_obelix` | raw Hargreaves database | `test_build_cohorts_synthetic_strict_cohort_has_no_overlap_with_obelix` |
| `tests/unit/test_manifest_schema.py::test_hargreaves_raw_file_matches_manifest` | raw Hargreaves database | `test_hargreaves_manifest_records_source_counts_and_terms` |
| `tests/unit/test_manifest_schema.py::test_external_cohort_files_match_manifest` | withheld cohort files | `test_external_cohort_manifest_is_consistent_with_raw_manifest_and_lock` |
| `tests/unit/test_issue10_formula_balanced_sensitivity.py::test_formula_duplication_stats_matches_frozen_strict_cohort` | withheld strict cohort file | the other issue-10 unit tests (synthetic frames) |
| `tests/unit/test_issue10_formula_balanced_sensitivity.py::test_loader_functions_do_not_mutate_frozen_strict_cohort_file` | withheld strict cohort file | — |
| `tests/unit/test_path_migration.py::test_every_lock_hashed_path_resolves_to_matching_bytes` | withheld cohort files and overlap ledger (hashed by the lock) | `test_lock_hashed_paths_in_package_scope_match_and_only_restricted_are_out` |

## 2. Tests not shipped in the package

These directories stay in the git repository and run there (`python -m pytest`).
They are not copied into the package (`EXCLUDE_PREFIXES` in
`scripts/release/build_package.py`), because they test software or records
outside this study's package:

- `tests/closed_project/` (category D): tests of the closed earlier project
  (v1.0 GNN transport pretraining; `RESEARCH_CONTRACT.md` §v1.0). They import
  its `scripts/phase2_*.py`, need its LiTraj raw data, or test one of the 14
  modules no part of this study imports. Those modules (`CLOSED_PROJECT_SRC`
  in the builder) are not shipped either.
  `test_no_shipped_python_file_imports_an_unshipped_closed_project_module`
  checks that nothing shipped imports them.
- `tests/dev_tooling/` (category C): tests of the development editor hook.
- `tests/repository/` (category R): checks of repository-only records.

| module | tests | category | reason |
|---|---:|---|---|
| `tests/closed_project/test_canonical_id.py` | 4 | D | tests `src/sse_transfer/data/canonical_id.py`, a closed-project module not shipped |
| `tests/closed_project/test_censored_loss.py` | 26 | D | tests `src/sse_transfer/training/censored_loss.py`, a closed-project module not shipped |
| `tests/closed_project/test_cif_rounding_audit.py` | 2 | D | tests `src/sse_transfer/audit/cif_rounding.py`, a closed-project module not shipped |
| `tests/closed_project/test_closed_project_manifests.py` | 11 | D | BVEL/BVSE/DFT-NEB manifests (moved from `tests/unit/test_manifest_schema.py`) |
| `tests/closed_project/test_constants.py` | 2 | D | tests `src/sse_transfer/constants.py`, a closed-project module not shipped |
| `tests/closed_project/test_ensembling.py` | 9 | D | tests `src/sse_transfer/evaluation/ensembling.py`, a closed-project module not shipped |
| `tests/closed_project/test_final_candidate_promotion_gate.py` | 7 | D | Phase 2.6B final-candidate gate of the closed project |
| `tests/closed_project/test_leakage_audit.py` | 7 | D | tests `src/sse_transfer/audit/leakage.py`, a closed-project module not shipped |
| `tests/closed_project/test_phase2_6a_censored_fold_leakage.py` | 10 | D | imports `scripts/phase2_6a_censored_learning.py`, `scripts/phase2_3_*` (closed project, not shipped) |
| `tests/closed_project/test_phase2_6b_oof_and_manifest.py` | 8 | D | imports `scripts/phase2_6b_final_candidate.py` and the closed project's dispersion module |
| `tests/closed_project/test_phase2_7_pretest_lock.py` | 11 | D | imports `scripts/phase2_7_pretest_lock.py` (the closed project's lock) |
| `tests/closed_project/test_phase2_7b_reseal_pretest_lock.py` | 8 | D | imports `scripts/phase2_7b_reseal_pretest_lock.py` |
| `tests/closed_project/test_phase2_8_final_test_evaluation.py` | 12 | D | imports `scripts/phase2_8_final_test_evaluation.py`; re-verifies the closed project's `CODE_HASH_FILES` |
| `tests/closed_project/test_preprocessing.py` | 8 | D | tests `src/sse_transfer/features/preprocessing.py`, a closed-project module not shipped |
| `tests/closed_project/test_pretrained_structure_embeddings.py` | 17 | D | tests `src/sse_transfer/features/pretrained_structure_embeddings.py`, a closed-project module not shipped |
| `tests/closed_project/test_pretraining_parsers.py` | 4 | D | LiTraj BVEL/BVSE/DFT-NEB raw-data parsers (moved from `tests/integration/test_dataset_parsers.py`) |
| `tests/closed_project/test_structure_features.py` | 15 | D | tests `src/sse_transfer/features/structure_features.py`, a closed-project module not shipped |
| `tests/closed_project/test_structure_flag_reconciliation.py` | 4 | D | tests `src/sse_transfer/data/structure_flag_reconciliation.py`, a closed-project module not shipped |
| `tests/closed_project/test_structure_flag_reconciliation_real_data.py` | 2 | D | tests `src/sse_transfer/data/structure_flag_reconciliation.py`, a closed-project module not shipped |
| `tests/closed_project/test_structure_recovery.py` | 25 | D | tests `src/sse_transfer/data/structure_recovery.py`, a closed-project module not shipped |
| `tests/closed_project/test_structure_recovery_real_data.py` | 2 | D | tests `src/sse_transfer/data/structure_recovery.py`, a closed-project module not shipped |
| `tests/closed_project/test_transport_descriptors.py` | 14 | D | tests `src/sse_transfer/features/transport_descriptors.py`, a closed-project module not shipped |
| `tests/closed_project/test_uncertainty_dispersion.py` | 12 | D | tests `src/sse_transfer/uncertainty/dispersion.py`, a closed-project module not shipped |
| `tests/dev_tooling/test_block_prohibited_compute_hook.py` | 13 | C | checks the development editor hook in `.claude/hooks/` (not package software) |
| `tests/repository/test_repository_records.py` | 1 | R | checks repository-only historical records (`artifacts/historical/`, `reports/history/`, ...); package-scope counterpart `tests/unit/test_path_migration.py::test_migration_map_entries_in_package_scope_resolve` runs |

Totals: 220 closed-project, 13 dev-tooling and 1 repository-only, so 234
repository tests are not shipped. Every one passes in the full repository run.
