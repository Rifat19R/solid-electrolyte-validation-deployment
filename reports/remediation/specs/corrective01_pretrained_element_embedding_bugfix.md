# Spec — Corrective 01: `pretrained_element_embedding` feature-lookup bugfix

## CAUSE

A confirmed implementation bug in `src/sse_transfer/features/element_embeddings.py`:
`_raw_element_vector` called `ElementProperty.data_source.get_elemental_property(symbol, prop)`
with a raw Python `str` element symbol. `get_elemental_property` requires a
`pymatgen.core.Element` (it reads `elem.symbol` internally), so every call
raised `AttributeError`, caught by a bare `except Exception` and recorded as
`NaN`. Every element, every one of the 22 raw Magpie properties, was
therefore `NaN` unconditionally. In `TrainFitElementEmbedder.fit`, an
all-NaN column's median imputed to `0.0` (pre-fix fallback), giving an
all-zero `imputed` table, `mean=0`, `std=0`, guarded to `std=1.0` — so every
element's standardized vector, and therefore every composition's
fraction-pooled embedding, was the exact zero vector regardless of
chemistry. The `pretrained_element_embedding` model (manuscript name: "the
pooled elemental-property model") was trained on a constant, chemistry-blind
input for every historical result in this study. See
`reports/acs/corrective01_impact_map.md` §0 for the full trace, and
`tests/unit/test_element_embeddings.py::test_raw_element_vector_is_not_all_nan`
for the regression test.

The fix (already in the working tree, not yet committed): pass
`pymatgen.core.Element(symbol)` instead of `symbol`. A companion fix changes
the all-NaN-column fallback from "impute 0.0 for every column" to "only
`nanmedian`-impute columns with at least one finite value across the training
vocabulary, else 0.0" — relevant only if a future, much smaller element
vocabulary left some property totally undefined; immaterial for the
449-row OBELiX development population, where every column has finite values
once `Element` is passed correctly. Two further guard fixes (constant-input
NaN handling in `evaluation/metrics.py`'s `spearman_rho`/bootstrap CI, and
in `evaluation/reviewer_remediation.py`'s `pearson_r`) are pre-existing,
orthogonal correctness fixes needed because the broken model previously
produced literal constant predictions that triggered undefined-correlation
edge cases those functions did not handle; they remain correct and needed
regardless of this bug, since any model's prediction *could* in principle be
constant under a different fold/seed.

## SCOPE

Only `pretrained_element_embedding` is refit and re-evaluated. The other
seven models (ridge, random_forest, extra_trees, xgboost, catboost,
shallow_mlp, composition_attention) never touched `element_embeddings.py`
and are provably unaffected — confirmed by grep: none of their code paths
in `src/sse_transfer/models/ladder.py` or
`src/sse_transfer/features/composition_pipeline.py` import or call anything
in `features/element_embeddings.py`.

## UNCHANGED

- Raw data (`data/raw/`, `data/external/`), cohort definitions (449-row
  OBELiX development population; 204-row strict / 239-row relaxed Hargreaves
  external cohorts), and the train(359)/validation(90) split used for
  hyperparameter tuning.
- Regime A/B/C/D/E definitions and generators
  (`evaluation/internal_validation.py`, `n20b_formula_group_kfold.py`'s
  formula-GroupKFold, `phase1_0_1_predeclare_and_glass_split.py`'s
  `chemical_system()` grouping for regime E).
- Hyperparameter search space and trial budget for this model family
  (`training/tune_ladder.py::tune_torch_model`, `embed_search_space` in
  `n3_build_model_ladder.py`: `hidden_dim in {16,32,64}`, `dropout in
  [0,0.5]`, `lr log-uniform [1e-4,1e-2]`, `weight_decay log-uniform
  [1e-6,1e-2]`; `TORCH_N_TRIALS=20`; tuning seed `20260921`).
- Seed convention: tuning seed `20260921`; final external-transfer seeds
  `[17, 29, 43, 71, 101]`; regime-A repeated-seed sensitivity seeds
  `20260921..20260930`; issue04 control seeds `[20260930, 20260931,
  20260932]`.
- The official 113-row OBELiX test split remains unopened (verified: every
  script below reads `dataset.test` only for `len()`, never features/labels
  — same pattern as the historical scripts, unchanged by this corrective
  work).
- The other seven models' frozen hyperparameters, fragments, and every
  historical merged artifact remain byte-identical (verified by SHA-256
  before/after in Phase 7).

## CORRECTION POLICY

No historical artifact is overwritten. Specifically:
- `artifacts/npj_model_ladder_frozen.json`, `artifacts/npj_n4b_fragments/*`
  (other 7 models), `artifacts/npj_n4b_nested_internal_validation.json`,
  `artifacts/npj_n6_fragments/*` (other 7), `artifacts/npj_n6_external_transfer.json`,
  `artifacts/npj_n20b_fragments/*` (other 7), `artifacts/npj_n20b_formula_group_kfold.json`,
  `artifacts/acs_issue08b_fragments/*` (other 7), `artifacts/acs_issue08b_chemical_system_groupkfold.json`,
  `artifacts/npj_n21_fragments/*` (other 7 per seed), `artifacts/npj_n21_repeated_cv_sensitivity.json`,
  `artifacts/acs_issue04_fragments/*` (other 7 per seed), `artifacts/acs_issue04_size_matched_control.json`,
  and every `npj_n7b`/`npj_n8`/`npj_n9b`/`npj_n15_*`/`npj_n20`/`npj_n21c`/`npj_n21d`/
  `acs_issue03`/`acs_issue06`/`acs_issue09*`/`acs_issue10` artifact are left untouched.
- Before this model's existing (broken) fragment files are regenerated in
  place (the `--model pretrained_element_embedding` CLI path on each
  per-model-fragment script writes only to that model's own fragment file,
  which is not itself one of the "historical merged artifacts" enumerated
  above, but is still prior work product), a byte-identical archival copy
  of every such fragment is made first under
  `artifacts/corrective01_pretrained_element_embedding/pre_fix_archive/` —
  preserving the broken numbers for the Phase 5 old-vs-corrected comparison.
- New corrected artifacts use the `corrective01_` prefix/directory
  throughout, e.g.
  `artifacts/corrective01_pretrained_element_embedding/model_ladder_corrected.json`,
  `.../n4b_nested_internal_validation_corrected.json`,
  `.../n6_external_transfer_corrected.json`, etc.

## EXTERNAL TRANSFER

The original one-shot N6 result (`artifacts/npj_n6_external_transfer.json`,
`authorized=true`, `run_count=1`) remains preserved exactly as-is and is
never re-merged or re-authorized. A corrective one-shot transfer is
authorized for `pretrained_element_embedding` only, because the original
implementation did not instantiate the intended representation for this one
model (it fit a provably degenerate, chemistry-blind input for every
split in the study — not a legitimate first attempt at this architecture).
This corrective transfer:
- Uses the identical 449-row OBELiX development population, identical
  strict (204-row) and relaxed (239-row) external cohorts, identical five
  final seeds `[17, 29, 43, 71, 101]`, identical preprocessing semantics
  (`TrainFitElementEmbedder`, now producing non-degenerate features),
  identical hyperparameter-tuning protocol (Optuna, `TORCH_N_TRIALS=20`,
  tuning seed `20260921`, train/validation-only objective), and identical
  evaluation metrics (`evaluation/metrics.py::regression_metrics`).
- Does **not** call `n6_external_transfer.py`'s `check_run_authorization`/
  `merge_fragments` path (which would write to the locked historical
  artifact); it only invokes `run_one_model("pretrained_element_embedding")`
  (writes solely to `artifacts/npj_n6_fragments/pretrained_element_embedding.json`,
  a per-model fragment file, after archiving the pre-fix copy) and then a
  new, separate corrective-merge script (not `n6_external_transfer.merge_fragments`)
  combines the 7 untouched historical fragments with this 1 corrected
  fragment into `artifacts/corrective01_pretrained_element_embedding/n6_external_transfer_corrected.json`,
  marked `"authorized": "N/A_CORRECTIVE_NON_PRIMARY"` — it is never presented
  as a second primary one-shot result, only as the corrected drop-in
  replacement for this one model's numbers within a reconstructed 8-model
  view.
- The official 113-row OBELiX test split is not opened at any point in this
  corrective work (same leakage boundary as every historical script it
  reuses).

## Required reruns (enumerated, this model only)

1. **Global hyperparameter re-tune for final transfer** (N3-equivalent, new
   standalone script, not touching `npj_model_ladder_frozen.json`):
   re-run Optuna tuning for `pretrained_element_embedding` on the
   train(359)/validation(90) split, same search space/budget/seed as N3,
   using the now-corrected `TrainFitElementEmbedder`. Output: a corrected
   hyperparameter entry, hashed and written to
   `artifacts/corrective01_pretrained_element_embedding/model_ladder_corrected.json`
   (a full copy of the frozen ladder with only this one model's entry
   replaced, so downstream per-model-fragment scripts can be pointed at it
   via a path override without needing any of their source code changed).
2. **Nested internal validation, regimes A/B/C**
   (`n4b_nested_internal_validation.py --model pretrained_element_embedding`,
   pointed at the corrected ladder file).
3. **Regime D** (`n20b_formula_group_kfold.py --model pretrained_element_embedding`).
4. **Regime E** (`scripts/acs/acs08b_chemical_system_groupkfold.py --model pretrained_element_embedding`).
5. **Regime-A repeated-seed sensitivity**
   (`n21_repeated_cv_sensitivity.py --model pretrained_element_embedding --seed <each of 10>`).
6. **Final corrective external transfer** (`n6_external_transfer.py --model
   pretrained_element_embedding`, pointed at the corrected ladder file;
   strict + relaxed cohorts; 5 seeds).
7. **Training-size-matched control** (`scripts/acs/acs04_size_matched_control.py
   --model pretrained_element_embedding --seed <each of 3>`).
8. **DOI-source-disjoint subset**: re-derived from item 6's corrected
   per-row strict-cohort predictions (no separate model fit — this is a row
   filter + rescoring, matching how the historical analysis derives it).

No other model is refit. No hyperparameter search space, trial budget, fold
definition, or seed is changed for this model or any other.

## Decision rule for Phase 5

Every downstream statistic is recomputed from the corrected artifacts (see
`reports/acs/corrective01_impact_map.md` §3 for the full list of pure
aggregation scripts to re-execute against corrected inputs) and classified
UNCHANGED / NUMERIC_UPDATE_ONLY / INTERPRETATION_NARROWS /
INTERPRETATION_CHANGES against the manuscript's current text. The
pre-registered H1/H2 success criteria themselves
(`EXPERIMENT_PROTOCOL.md` "Success criteria") are not altered by this
correction — only the input numbers feeding them are.

## Expected outputs

All under `artifacts/corrective01_pretrained_element_embedding/` and
`reports/acs/corrective01_*.md`; see the impact map for the complete file
list. This spec is hashed into `reports/acs/specs/HASHES.txt` before any
computation in Phase 2 or Phase 3 begins.
