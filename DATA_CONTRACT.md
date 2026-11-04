# DATA CONTRACT

## Dataset roles
### A. Whole-crystal transport pretraining
LiTraj/BVEL-style percolation barriers. Prefer structure-level 1D/2D/3D transport quantities where available.

### B. Local migration pretraining
LiTraj approximate/BVSE local-hop migration labels.

### C. Higher-fidelity microscopic transport
Existing published DFT-NEB migration barriers only. No new DFT/NEB generation.

### D. Final experimental task
OBELiX room-temperature lithium solid-electrolyte ionic conductivity.

## Canonical label
Final regression target:
`y = log10(sigma_RT / S cm^-1)`

Do not silently mix:
- total vs bulk conductivity,
- room-temperature vs extrapolated values,
- conductivity and diffusion coefficients,
- experimental and computed conductivity.

Any allowed harmonization must be explicit, versioned, tested, and recorded.

## Disorder
Fractional occupancy is first-class information.
Do not:
- round occupancies to a single species,
- drop vacancy fractions silently,
- replace a disordered site with the majority element without an ablation.

Preferred site representation:
`h_i = sum_j occupancy_ij * embedding(element_j)`
plus explicit occupancy/vacancy/disorder features.

## Censoring
Values reported only as inequalities (for example `< threshold`) are not exact regression labels.
Primary analysis must use a censor-aware strategy or exclude such points with a documented sensitivity analysis.

## Provenance
Every raw file must have a manifest entry containing:
- dataset name and version,
- source URL / DOI,
- retrieval date,
- license if available,
- original filename,
- SHA-256,
- row/structure count,
- parser version,
- notes.

## Canonical identity and overlap audit
Create material fingerprints using, where possible:
- reduced composition,
- anonymized formula,
- space group,
- lattice/structure fingerprint,
- pymatgen StructureMatcher,
- source IDs (Materials Project/ICSD/etc.) when legally available.

Before any model training:
- detect duplicates within each dataset;
- detect overlaps across pretraining and OBELiX;
- ensure final test items cannot leak through pretraining supervision.

## Split rules
Primary split must preserve the official benchmark split when feasible.
Additional OOD splits may include:
- structural-family holdout;
- chemistry-family holdout;
- composition-group holdout.

Never select a split because it produces a better score.

## Immutable raw data
`data/raw/` and `data/external/` are append-only after checksums are recorded.
All transformations write to `data/interim/` or `data/processed/`.
