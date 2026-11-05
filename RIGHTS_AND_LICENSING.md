# Rights and Licensing

## Code

The original software written for this project (`src/`, `scripts/`,
`tests/`, `configs/`) is released under the MIT License; the full licence
text is in `LICENSE` at the repository root. The MIT License covers this
software only. It grants no right over third-party datasets referenced or
used by this project (see below), and it does not cover the manuscript,
Supplementary Information text or journal-published figures (see
"Manuscript and Supplementary Information").

## Third-party datasets

The repository's code license does not grant redistribution rights over
third-party datasets.

### OBELiX

OBELiX is distributed under CC BY 4.0 and may be redistributed with
appropriate attribution, subject to its original terms.

### Hargreaves et al. conductivity database

The Hargreaves et al. lithium solid-electrolyte conductivity database is
used under its original academic/research-use terms.

The raw Hargreaves database is not redistributed in the public release.
Instead, the project provides source citation, retrieval information,
provenance metadata, and file hashes sufficient to identify and verify
the source used.

The terms do not grant redistribution, so the public archive ships no
file from which the external conductivity values can be read back:

- withheld (not shipped): the raw database, the row-level overlap ledger
  `hargreaves_overlap_ledger.csv`, and the formula-level cohort files
  `hargreaves_strict.csv` and `hargreaves_relaxed.csv` (they carry the
  median log10 conductivity of every external formula);
- redacted (shipped with the per-row `y_true` and `residual` fields set to
  null): the one-shot external-transfer records and their per-model
  fragments. Predictions and every aggregate metric are kept.

`provenance/restricted_data.json` records the SHA-256 of every withheld
file and of every original and redacted record. After obtaining the raw
database under its terms, `python scripts/release/restricted_data.py restore`
rebuilds the withheld files, restores the redacted fields, and accepts a
file only if it matches its recorded hash (`data/RECONSTRUCTION.md`).
Two repositories exist and must not be confused:

- the authors' internal development repository (not public) keeps the
  unredacted records under version control for provenance and the
  withheld files in its local working copy only (they are not committed);
  it is never pushed or published;
- the public GitHub repository listed below is created only from the
  curated, redacted reproducibility package and contains none of the
  withheld files or redacted values.

## Manuscript and Supplementary Information

The repository MIT License applies to software only.

The copyright/license applicable to the manuscript, Supplementary
Information, and journal-published figures will follow the publication
terms selected by the authors and the publisher.

## Public archival release

Public repository: https://github.com/Rifat19R/solid-electrolyte-validation-deployment
(curated reproducibility package; Hargreaves-derived files withheld or
redacted as described above).

Archival deposit: version 1.0.0 is to be deposited on Zenodo under the
reserved DOI 10.5281/zenodo.23987392. The DOI resolves only after the deposit is
published; until then it must not be described as published or
accessible.
