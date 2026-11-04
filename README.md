# SSE reproducibility package

Code, frozen protocol, analysis records and submission documents for the
manuscript *Validation–Deployment Alignment Shapes How Internal Evaluation
Predicts External Generalization in Lithium Solid-Electrolyte Machine
Learning* (M. R. Khandaker and M. A. Chowdhury).

## Scientific purpose

The study asks whether chemistry-family-aware cross-validation anticipates
external performance better than random cross-validation for experimentally
measured lithium solid-electrolyte ionic conductivity. Eight composition-only
models were developed on the 449-row OBELiX development population, evaluated
under random five-fold CV, Family-GroupKFold and leave-one-major-family-out
(LOMFO) CV with nested hyperparameter tuning, and transferred under a
hash-locked protocol to a composition-disjoint 204-formula external cohort
built from the Hargreaves et al. database. The pre-specified hypotheses H1
(family-aware CV estimates external error better) and H2 (family-aware CV
ranks models better) were **not supported**.

## Contents

| path | content |
|---|---|
| `src/sse_transfer/` | library: data loading, cohort construction, features, models, evaluation (this study's modules only) |
| `scripts/transport_analysis/`, `scripts/analysis/` | study pipeline (nested CV, external transfer, statistics, post hoc analyses, corrective reruns) |
| `scripts/figures/`, `scripts/manuscript/`, `scripts/manuscript_figures/` | figures, tables, document build and rendering |
| `scripts/release/` | rerun/compare pipeline, anchor and package verification, restricted-data restore |
| `scripts/preflight.py`, `scripts/fetch_raw_data.py`, `scripts/build_processed_v1.py`, `scripts/freeze_split.py`, `scripts/hash_file.py` | environment check, raw-data retrieval, processed OBELiX split, split freezing, hashing |
| `tests/` | this study's unit, integration and scientific tests (see `TEST_COVERAGE_EXCEPTIONS.md`) |
| `configs/` | run configuration files |
| `artifacts/frozen/` | hash-locked protocol (`external_transfer_lock.json`), frozen model ladder, primary one-shot external-transfer record (per-row targets redacted) |
| `artifacts/authoritative/` | all current results after the pooled elemental-property model correction (per-row external targets redacted) |
| `artifacts/fragments/` | recorded per-model, per-fold outputs merged into the authoritative results (per-row external targets redacted) |
| `data/manifests/`, `data/RECONSTRUCTION.md` | provenance of every raw file; lawful reconstruction of the withheld Hargreaves-derived files |
| `data/raw/obelix/`, `data/processed/v1/` | OBELiX source files (CC BY 4.0) and the frozen processed split |
| `manuscript/` | markdown sources, templates, figures and tables |
| `submission/` | Main_Manuscript, Supporting_Information, Cover_Letter, Data_and_Code_Availability (DOCX and PDF) |
| `provenance/` | path-migration map, dependency graph, authoritative-artifact promotions, frozen state, restricted-data and untracked-input hash records |
| `reports/remediation/specs/`, `reports/results/leakage/`, `reports/rerun/` | hashed analysis specifications, frozen OBELiX leakage ledgers, rerun comparison |
| `environment/` | pinned Python packages and version |
| `README.md`, `TEST_COVERAGE_EXCEPTIONS.md`, `VERSION.txt`, `MANIFEST_SHA256.txt` | this file, test exceptions, source commit, SHA-256 of every file |
| `LICENSE`, `CITATION.cff`, `RIGHTS_AND_LICENSING.md`, `CHANGELOG.md` | MIT licence, citation metadata, dataset terms, change log |
| `RESEARCH_CONTRACT.md`, `DATA_CONTRACT.md`, `EXPERIMENT_PROTOCOL.md`, `DEFINITION_OF_DONE.md` | governance documents |
| `pyproject.toml`, `Makefile` | package metadata and test/lint configuration; convenience targets |

Not shipped: the governance documents and `CHANGELOG.md` also cite
repository-only records (study and audit reports under `reports/`, such as
`reports/phase2_9_final_scientific_closure.md`, `reports/validation/`,
`reports/cross_dataset_study/` and `reports/literature/`), and the
superseded pre-correction artifacts. These are retained in the authors'
internal Git history and are not part of this package; no reported number
depends on them. `TEST_COVERAGE_EXCEPTIONS.md` lists the unshipped tests and
modules.

## Data and licensing

Code is MIT-licensed (`LICENSE`). OBELiX is CC BY 4.0 and is included with
attribution. The Hargreaves et al. database is used for research with
citation of the University of Liverpool. Its terms do not grant
redistribution, so the package contains no file from which its conductivity
values can be read. The raw database, the row-level overlap ledger and the
formula-level strict (204) and relaxed (239) cohort files are withheld. The
per-row `y_true` and `residual` fields of the external-transfer records are
set to null; every prediction and aggregate metric is kept.
`provenance/restricted_data.json` records the SHA-256 of every withheld file
and of every original and redacted record. `RIGHTS_AND_LICENSING.md` and
`data/RECONSTRUCTION.md` give the source URL, retrieval date (2026-09-21),
expected SHA-256 (`805e157d...0606`), row count (820) and the exact filters.

## Environment

Python 3.12 (CPU). Install the pinned packages and the package itself:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements-lock.txt
pip install -e . --no-deps
```

## Reproduction sequence

1. Verify the package: `sha256sum -c MANIFEST_SHA256.txt` and
   `python scripts/preflight.py --release`.
2. Verify the reported numbers: `python scripts/release/verify_anchors.py`
   (29 anchors), `python scripts/manuscript/build_tables.py --check-only`
   (Tables 1-4 and Supplementary Table 1 against the manuscript) and
   `python scripts/analysis/numeric_audit.py`. None of these needs the
   withheld data.
3. Tests: `python scripts/release/run_package_tests.py` runs the whole
   shipped `tests/` tree. While the Hargreaves material is withheld, exactly
   the seven restricted-data tests listed in `TEST_COVERAGE_EXCEPTIONS.md`
   are deselected; any other failure or any skip fails the run.
4. Restore the Hargreaves-derived material (requires the raw file obtained
   under its terms, placed at `data/raw/hargreaves_conductivity/LiIonDatabase.csv`):
   `python scripts/release/restricted_data.py restore`. This rebuilds the
   cohort files and ledger (`scripts/analysis/build_external_cohort.py`),
   refills the redacted fields and checks every file against its recorded
   SHA-256. Rerunning step 3 then runs every test with nothing deselected.
5. Rerun the analyses after step 4 (outputs go to `artifacts/rerun/` only; the
   recorded results are write-protected): `python scripts/release/rerun_pipeline.py
   --stage aggregate`, optionally `--stage refit` (about 12 h on 4 CPU cores),
   then `--stage compare`, which writes `reports/rerun/rerun_comparison.md`.
6. Figures: run each `scripts/figures/fig*.py` / `si_fig*.py` (byte-identical
   PNG/PDF expected; Supplementary Fig. 2 needs step 4). Documents:
   `python scripts/manuscript/build_submission.py`.

Expected outputs: every check in steps 1-4 passes; step 5 reproduces the
authoritative records exactly apart from run metadata (commit, runtime, file
paths), as recorded in `reports/rerun/rerun_comparison.md`.

## Validation anchors

Datasets: 599 processed OBELiX rows, 562 exact labels, 449 development, 113
exact-label test (not used), 37 censored excluded; external cohorts 204
strict, 153 DOI-source-disjoint, 239 relaxed. External MAE (strict cohort):
extra trees 1.2022, composition attention 1.2824, random forest 1.2861,
CatBoost 1.2916, XGBoost 1.3492, ridge 1.3820, shallow MLP 1.4831, pooled
elemental-property model 1.7163. Regime utility (median |gap|, Spearman,
Kendall, regret): random CV 0.4531/0.6429/0.5000/0.0000, Family-GroupKFold
0.6810/0.5476/0.3571/0.1471, LOMFO 0.5149/0.5714/0.4286/0.0802.

## Limitations

Composition-only features discard structure and disorder. Both databases
report heterogeneous conductivity measurements. Rank statistics over eight
models are coarse. The unseen-crystalline subset (n = 19) is too small for
inference. Two implementation corrections made before submission (nested-CV
retuning; the pooled elemental-property model's feature-lookup bug, with one
corrective external refit) are documented in the Supporting Information and
in `provenance/`. The public repository URL and archival DOI are added on
deposit; none is claimed here.
