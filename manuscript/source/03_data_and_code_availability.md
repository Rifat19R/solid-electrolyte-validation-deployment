# Data and Code Availability

## Data

<b>OBELiX</b> (Therrien et al., <i>Digital Discovery</i> 2026, 5, 910–918; DOI 10.1039/D5DD00441A) is licensed CC BY 4.0 and may be redistributed with attribution. Its official train/validation/test split was reused exactly; the 449-row train+validation union is this study's development population. The 113-row official test split was not used for fitting, tuning, internal validation, external-transfer evaluation or any reported predictive-performance estimate; one historical label-compatibility sensitivity read its labels and is disclosed in the Supplementary Information, and the reported compatibility analysis uses development rows only.

<b>The Hargreaves et al. lithium solid-electrolyte conductivity database</b> (Hargreaves et al., <i>npj Comput. Mater.</i> 2023, 9, 9; DOI 10.1038/s41524-022-00951-z) is not freely redistributable: research and academic use with citation of the University of Liverpool is permitted, and commercial re-use requires a separate licence from the University. Because these terms do not grant redistribution, no file from which the external conductivity values can be read is included in the public release: the raw database file, the row-level overlap ledger and the formula-level strict (204) and relaxed (239) cohort files are not included in the public release, and the per-formula external targets and residuals are removed from the archived external-transfer prediction records (every prediction and every aggregate metric is kept). The release records the source URL, retrieval date (2026-09-21), file size, row count and SHA-256 hash of the file used, the SHA-256 hash of every withheld or redacted file, and a script that rebuilds the cohorts and ledger from a lawfully obtained copy, restores the redacted fields and verifies every result against the recorded hashes (`data/RECONSTRUCTION.md`, `scripts/release/restricted_data.py`).

The OBELiX source files and processed split, model predictions, evaluation records and all sensitivity-analysis outputs are included in the release. No new experimental measurements or atomistic-simulation data were generated.

## Code

The code for cohort construction, nested internal validation, one-shot external transfer, every post hoc sensitivity analysis, and figure, table and manuscript generation is maintained in a version-controlled repository and is released under the MIT licence. The reproducibility archive contains the source code, the hash-locked protocol and analysis specifications, the authoritative analysis records, a pinned Python environment, a SHA-256 manifest of every file, and scripts that rerun the analyses and compare the results with the recorded values.

<b>Public deposit.</b> The public repository URL and archival DOI have not yet been created; they will be inserted here before submission. Until the deposit is complete, the code and reproducibility archive are available from the corresponding author on reasonable request.
