# Rights and Licensing

## Code

The source code in this repository is released under the MIT License.
See `LICENSE` at the repository root.

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
The git repository itself holds these files for provenance; a public
repository must be published without them (see the release notes).

## Manuscript and Supplementary Information

The repository MIT License applies to software only.

The copyright/license applicable to the manuscript, Supplementary
Information, and journal-published figures will follow the publication
terms selected by the authors and the publisher.

## Public archival release

The code/reproducibility archive is to be deposited publicly before
manuscript submission, with a permanent archival DOI. The final DOI and
repository URL must be inserted into the submission materials before
submission.
