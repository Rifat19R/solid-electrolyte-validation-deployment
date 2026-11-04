# Lawful reconstruction of the study inputs

The release does not redistribute the raw Hargreaves et al. database or any
file from which its conductivity values can be read (see
`RIGHTS_AND_LICENSING.md`). Users who obtain the raw files under the source
terms can rebuild every input byte-for-byte and verify it against the hashes
below.

## 1. OBELiX (CC BY 4.0, redistributable with attribution)

- Source: https://github.com/NRC-Mila/OBELiX (main branch, retrieved
  2026-09-12); paper DOI 10.1039/D5DD00441A.
- Files under `data/raw/obelix/` (SHA-256):
  - `all.csv` (114,856 bytes) `54f6265ef84d16a919d08c6815f14b0fd7465fa00090bee2511c16c042e57f3c`
  - `train.csv` (91,889 bytes) `01d87779862807e24bc45bb2c759852f892390b99f9340f13906c54805aa8776`
  - `test.csv` (23,220 bytes) `b2c266eb74e22f403743d52959bea7c43b1531a850994870a70dd909200e5297`
- Download: `python scripts/fetch_raw_data.py` (skips files already present).
- Processed split and labels: `python scripts/build_processed_v1.py`
  writes `data/processed/v1/obelix_index.json` (599 rows; official
  train/validation/test split preserved) and the exact-label target file
  (562 exact labels, 37 left-censored rows excluded from exact-label
  regression). The split hash `13692a31839e295dd88bd45ca527d869990e02d799a6878c5ded6c43d3b94d3e`
  is recorded in `data/processed/v1/obelix_index.json.sha256.json` and in the
  external-transfer lock. Development population: train + validation = 449
  exact-label rows; the 113-row exact-label test split is not used.

## 2. Hargreaves et al. conductivity database (research use with citation; not redistributed)

- Paper: Hargreaves et al., npj Comput. Mater. 9, 9 (2023),
  DOI 10.1038/s41524-022-00951-z. Cite the University of Liverpool as the
  source of the database. Commercial re-use requires a licence from the
  University of Liverpool.
- Source file: https://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.csv
  (landing page https://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.html),
  retrieved 2026-09-21.
- Expected file: `data/raw/hargreaves_conductivity/LiIonDatabase.csv`,
  76,402 bytes, SHA-256
  `805e157decb613e9d1ed651f4218c2ea69ed7d6fa2aa24b7b860485894570606`,
  820 rows from 214 sources (first 4 lines are a licence preamble).
  The upstream page may change; a different hash means a different
  version and the study will not reproduce exactly.

## 3. External cohorts (rebuilt locally, verified against the lock)

```bash
python scripts/analysis/build_external_cohort.py
```

Parser/pipeline: `src/sse_transfer/data/hargreaves_cohort.py::build_cohorts`
(hash-locked in `artifacts/frozen/external_transfer_lock.json`). Filters, in
order: temperature 15-35 degC (465 rows); positive finite target; parseable
reduced formula; overlap with OBELiX classified per reduced formula and
source DOI; strict cohort excludes any composition or DOI overlap, relaxed
cohort excludes only exact composition+DOI duplicates; aggregation is the
median log10 conductivity per reduced formula.

| file | rows | SHA-256 (lock) | public release |
|---|---:|---|---|
| `hargreaves_strict.csv` | 204 | `1bda972d893c3cf546b3638ea8935e59b345e1d14d9c607be5b1c59437b237c9` | withheld (carries the external targets) |
| `hargreaves_relaxed.csv` | 239 | `dac166ee16be91d76a255a30ec7f6af8cc47bc6d569db3ad74d2212209e99da8` | withheld (carries the external targets) |
| `hargreaves_overlap_ledger.csv` | 820 | `a1b4d0f425fc3048a03cdf4a82321292cbc80b9590ba7e309f7f353788098683` | excluded (row-level reconstruction of the raw database) |

The script writes a cohort file only if it is absent and its rebuilt bytes
match the lock hash; existing files are verified, never overwritten.
Analyses that read the overlap ledger (DOI-cluster bootstrap, source
overlap, family audits) need this step before they can be rerun.

## 4. Redacted external-transfer records (restored locally, verified)

In the public archive the one-shot external-transfer records
(`artifacts/frozen/n6_external_transfer_primary.json`,
`artifacts/authoritative/n6_external_transfer_corrected.json`) and the eight
per-model fragments (`artifacts/fragments/n6/*.json`) are shipped with the
per-row `y_true` and `residual` fields set to null; predictions and every
aggregate metric are unchanged. With the raw Hargreaves file in place:

```bash
python scripts/release/restricted_data.py restore
```

runs step 3, refills both fields from the rebuilt cohorts (residual =
prediction − target, as originally computed) and writes a record only if it
then matches its original SHA-256 in `provenance/restricted_data.json`. After
this step `scripts/release/run_package_tests.py` runs every test with nothing
deselected.
