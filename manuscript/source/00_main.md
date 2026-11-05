Validation–Deployment Alignment Shapes How Internal Evaluation Predicts External Generalization in Lithium Solid-Electrolyte Machine Learning

Md. Rifat Khandaker¹*, Mohammad Asaduzzaman Chowdhury²

¹ Department of Chemical Engineering, Dhaka University of Engineering & Technology (DUET), Gazipur-1700, Bangladesh

² Department of Mechanical Engineering, Dhaka University of Engineering & Technology (DUET), Gazipur-1700, Bangladesh

*Corresponding author: Md. Rifat Khandaker (rifatkh.duet@gmail.com)


## Abstract


Chemistry-family-aware cross-validation (CV) is often assumed to anticipate deployment performance better than random CV; we tested this by evaluating eight composition-only models developed on OBELiX (n = 449) under random five-fold CV, Family-GroupKFold and leave-one-major-family-out CV and transferring them once to a separately curated Hargreaves cohort (n = 204), composition-disjoint from OBELiX but partly sharing its literature sources, making this a lineage-related curation-shift test rather than a fully source-independent one. Random CV led all four pre-specified full-cohort criteria by point estimate, including external rank agreement (Spearman ρ = 0.64 versus 0.55 and 0.57), but most between-regime bootstrap confidence intervals did not establish superiority, and neither hypothesis that family-aware CV better anticipates external error magnitude or model ranking was supported. Deployment stratification showed the answer was conditional: random CV closely estimated external error for the 68% of external records from development-represented families (median gap 0.09 versus 0.97 log units for the family-aware regimes), whereas for an unseen glass/glass-ceramic class (n = 36) family-aware CV better matched error magnitude but not model ranking, and unseen crystalline families (n = 19) were too few to resolve. A composition-distance applicability-domain score stratified external risk for every model but largely overlapped a family-novelty flag, and cross-model prediction dispersion was the strongest tested post hoc risk comparator; validation should therefore be matched to the expected deployment shift and decision target, error-magnitude estimation or model selection, rather than chosen because a split is more restrictive.


## Introduction


Inorganic solid electrolytes are central to all-solid-state lithium batteries, and room-temperature ionic conductivity remains one of the principal screening criteria for new candidates.<sup>1–3</sup> Because experimental synthesis and measurement are slow and the accessible chemical space is vast, machine learning (ML) is increasingly used to prioritize candidate conductors from composition or structure,<sup>4–10</sup> and curated databases of experimentally measured conductivities now make direct supervised prediction possible.<sup>11,12</sup>

Across materials informatics, reported model accuracy is almost always estimated internally, on held-out partitions of a single dataset,<sup>13–17</sup> and community guidelines and benchmarks have formalized how such estimates should be produced and reported.<sup>18–21</sup> Models are nevertheless deployed on compositions gathered later, by other groups and under other protocols. Accurate in-distribution scores do not guarantee useful behaviour on genuinely new materials,<sup>22–24</sup> random splits of redundant materials datasets can overstate performance,<sup>25–27</sup> and composition-level duplicates are a documented source of such optimism.<sup>28,29</sup> The problem mirrors the gap between benchmark and deployment performance described more broadly in machine learning.<sup>30,31</sup>

Chemistry-aware or cluster-based cross-validation (CV) was introduced to narrow this gap. Leave-one-cluster-out CV,<sup>32</sup> forward CV,<sup>33</sup> kernelised cluster CV<sup>34</sup> and standardized splitting protocols such as MatFold<sup>35</sup> withhold whole regions of chemical space, so that a model must predict chemistry it has not seen; scaffold splits play the same role for molecules.<sup>36</sup> Stricter splits consistently yield less optimistic internal estimates,<sup>11,32,35</sup> and out-of-distribution benchmarks show that accuracy can degrade sharply outside the training distribution.<sup>27,37–43</sup> It is therefore often assumed that family-aware CV is also the better proxy for deployment. That premise has rarely been tested against a separately curated external dataset for the same property: the studies above construct their out-of-distribution splits within a single dataset, so they show that stricter splits are harder, not that they predict external error or model ranking better.

Lithium solid-electrolyte conductivity offers an unusual opportunity to run that test, because two experimental databases curated through different pipelines exist: OBELiX<sup>12</sup> and the database of Hargreaves et al.<sup>11</sup> Hargreaves et al. compared leave-one-cluster-out with random k-fold CV on their own 820-row database and found their best model, a transfer-learned CrabNet regressor, markedly worse under the cluster split (MAE 0.99 vs. 0.85; R² 0.33 vs. 0.62), a result we re-derived from their public code and results notebook. Here we use the two databases as source and target. Because both draw partly on the same primary literature, they are separately curated but source-lineage-overlapping resources, and the transfer is a lineage-related curation-shift transfer rather than a fully source-independent pairing: the external cohort is composition-disjoint from OBELiX, but 51 of its 204 records (25.0%) share a literature-source DOI with a different OBELiX composition, and the DOI-source-disjoint subset (n = 153) is analysed as a sensitivity. Eight composition-only models were developed on OBELiX, evaluated under random five-fold CV, Family-GroupKFold and leave-one-major-family-out (LOMFO) CV, and transferred once, under an internally pre-specified protocol, to a composition-disjoint external cohort curated from the Hargreaves database, before any external label was used for fitting, tuning, selection or evaluation (Fig. 1). We ask which internal regime best anticipated external error and model ranking, whether the answer depends on the kind of novelty present in the external data, and how well simple prediction-time risk scores stratify external error.<sup>44–46</sup>

The study makes four contributions. First, it provides a direct cross-dataset test of whether chemistry-family-aware validation anticipates external performance better than random CV for experimental Li-ion conductivity, including a formula-grouped control for duplicate-composition leakage. Second, it identifies an architecture- and taxonomy-sensitive failure mode in which ridge regression and a shallow multilayer perceptron can extrapolate catastrophically under particular family-exclusion constructions. Third, a post hoc stratification by deployment scenario shows that the answer depends on the kind of novelty being deployed against: within known families random CV anticipated external error closely (median gap 0.09 log units), whereas for an unseen amorphous material class family-aware CV better anticipated the magnitude, but not the ranking, of external error. Fourth, it benchmarks a composition-space applicability-domain score against simpler and alternative risk signals on the same external task. To our knowledge this is the first such comparison for experimental Li-ion solid-electrolyte conductivity; we make no claim beyond this dataset pairing.

Fig. 1 | Study design, cohort construction and validation logic. Eight composition-only models were developed on the 449-row OBELiX development population and evaluated with three primary internal-validation regimes plus a post hoc formula-grouped leakage control. The official 113-row OBELiX test split was not used for model fitting, tuning, internal validation, external-transfer evaluation or any reported predictive-performance estimate. A historical post hoc label-compatibility sensitivity accessed test labels; the final reported compatibility analysis uses development rows only. The frozen models were transferred once to the 204-row strict Hargreaves cohort, with relaxed (n = 239) and DOI-source-disjoint (n = 153) sensitivities.


## Methods



### Datasets and cohort construction


<b>Internal development dataset.</b> OBELiX<sup>12</sup> is a curated database of room-temperature ionic conductivities of lithium solid electrolytes with an official, hash-verified train/validation/test split designed to limit leakage between similar entries. We reused this split exactly. The <b>development population</b> used throughout is train + validation (449 rows); the official 113-row test split was not used for model fitting, hyperparameter selection, internal validation, external-transfer evaluation or any reported predictive-performance estimate. A historical post hoc label-compatibility sensitivity later accessed test labels; the final compatibility analysis reported here was rerun using development rows only.

<b>External dataset.</b> The Hargreaves et al. database<sup>11</sup> contains 820 experimentally measured lithium solid-electrolyte conductivities from 214 literature sources. We reproduced the stated counts (820 rows, 214 sources) and the stated 403 unique room-temperature compositions exactly, using the authors' composition-normalization method on the same predeclared 15–35 °C window used throughout (Supplementary Information).

Cohort construction. We restricted the external data to a predeclared room-temperature window (15–35 °C), fixed before observing how many rows it would yield. Reduced chemical formulas were canonicalized with pymatgen.<sup>47</sup> Rows whose composition and literature source both matched an OBELiX entry were excluded from the primary (strict) cohort; rows matching only on composition were excluded from the strict cohort but retained in a relaxed sensitivity cohort. Retained rows sharing a reduced formula were then aggregated by `aggregate_by_formula` to one record per formula, using the median room-temperature target across that formula's retained measurements (213 pre-aggregation rows for strict, 263 for relaxed). This produced a strict cohort of 204 records representing 204 unique reduced formulas and a relaxed cohort of 239 records (strict ⊂ relaxed; Table 1); primary external metrics assign equal weight to these aggregated formula-level records. The strict cohort is composition-disjoint from OBELiX by construction but not DOI-source-disjoint: 51 of its 204 entries (25.0%) trace to a literature source that also contributes a different composition to OBELiX. We therefore also constructed a DOI-source-disjoint subset (n = 153) that excludes any row whose literature-source DOI appears anywhere among OBELiX's DOIs.

Glass/crystalline sub-split and pre-declared subset-size thresholds. Crosswalking Hargreaves' family labels onto OBELiX's family vocabulary (exact string match after normalization; no fuzzy matching) classifies each strict-cohort formula record as SEEN, UNSEEN or AMBIGUOUS (see 'Family-composition, duplicate-formula and label-compatibility audits'). To test whether the 55-row UNSEEN subset should be split further by material class, we wrote and hashed, before computing any subset statistic, a three-tier sample-size rule (n ≥ 30 for a full regime-ranking comparison with bootstrap confidence intervals (CIs); n ≥ 20 for MAE with a bootstrap CI only; below n = 20, descriptive statistics only) and a three-branch conditional framing rule, evaluated mechanically, that determines whether an observed pattern is described as unseen chemistry, unseen material class or unresolved. Each UNSEEN row's raw family label was then checked against {Glass, Glass-Ceramic}, giving 36 glass/glass-ceramic and 19 crystalline unseen-family rows. This sub-split is post hoc: the question was posed after the primary regime comparison was complete.


### Feature representation


All models use deterministic, composition-only representations computable from a formula string alone, because composition is the only input available identically in both databases. Composition-based models are widely used for experimental property prediction,<sup>48–53</sup> although structure-aware models are generally more accurate when reliable structures are available.<sup>22,54–56</sup> The primary schema is a 132-column Magpie elemental-property statistic vector<sup>57</sup> computed with matminer,<sup>58</sup> verified identical when computed on OBELiX and Hargreaves formulas. Two of the eight models instead use a fraction-weighted pooled vector of 22 raw per-element Magpie physical properties (a deterministic lookup, not a learned or pretrained embedding; we call this the pooled elemental-property model) or a learned per-element-token representation (composition-attention model). During pre-submission implementation auditing, an elemental-property lookup bug was identified in the pooled elemental-property representation (Supplementary Note 27). The implementation was corrected before submission, and all results reported here for this model were recomputed using the corrected representation.


### Model ladder and tuning


The eight models were ridge regression, random forest, extremely randomized trees, XGBoost,<sup>59</sup> CatBoost,<sup>60</sup> a shallow multilayer perceptron (MLP) on the Magpie vector, a shallow MLP on the pooled elemental-property vector (the pooled elemental-property model) and a composition-attention model in the style of Roost<sup>52</sup> and CrabNet.<sup>53</sup> Linear and bagged-tree models were implemented in scikit-learn<sup>61</sup> and neural models in PyTorch;<sup>62</sup> tree ensembles were included because they remain strong baselines for tabular descriptors.<sup>63,64</sup> Hyperparameters were selected with Optuna<sup>65</sup> (fixed trial budget and seed) using only the development population's train split for fitting and its validation split for the tuning objective; the resulting configuration was frozen for the final one-shot external-transfer fits. Internal cross-validation did not reuse this global configuration: as described below, hyperparameters were retuned independently within each outer fold using only that fold's outer-training partition. Search spaces, architectures and preprocessing are given in the Supplementary Information.


### Internal validation regimes


On the 449-row development population, every model was evaluated under <b>random five-fold CV</b> (regime A); <b>Family-GroupKFold</b> (regime B), grouped by a harmonized chemical-family label; <b>LOMFO</b> (regime C), holding out every row of each family with ≥15 rows in turn; and, as a post hoc diagnostic control, <b>formula-GroupKFold</b> (regime D), holding every exact-duplicate reduced formula within one outer fold. Regimes A–C belong to the pre-specified protocol.

<b>Nested model selection.</b> An initial pass reused each model's single train/validation-tuned hyperparameter set across all outer folds. Because the validation rows that shaped those hyperparameters later served as outer-test rows in most folds, this carried an internal optimism bias, a recognized form of information leakage in ML-based science.<sup>18,29</sup> The defect was identified and corrected after the external predictions had been generated. The nested-CV correction was completed about 10.5 h after the external predictions first became available, after the initial GO/NO-GO assessment; this correction did not rerun the external predictions, and all internal-versus-external comparisons were recomputed using the corrected values (timeline in Supplementary Information). In the corrected procedure, hyperparameters are retuned from scratch for every outer fold using an inner split of that fold's own training partition only. All internal-validation numbers reported here use this nested recomputation.


### One-shot external transfer


Before any Hargreaves target value was read by fitting or model-selection code, the complete downstream protocol and a GO/NO-GO decision rule were frozen in an internally pre-specified, hash-locked protocol. Each frozen model was then refit once on the full 449-row development population using five fixed seeds; the mean of the five seed-specific predictions defines the deployed external point predictor. The models were used to predict, once, on the strict and relaxed external cohorts. A run-authorization guard prevents a second invocation from overwriting this result. The external inference was not rerun for any model except the pooled elemental-property model, which received one disclosed corrective refit under the same frozen protocol after its feature-lookup bug was fixed (Supplementary Note 27). The internal regime scores reported below are pooled out-of-fold estimates from nested CV and therefore estimate the validation procedure rather than this final fitted five-seed predictor.


### Regime-utility, applicability-domain and risk-stratification statistics


For each internal regime, each model's pooled internal MAE was compared with its external MAE using four predeclared statistics: the median absolute gap between internal and external MAE; Spearman and Kendall rank correlation between the regime's model ranking and the external ranking; and top-model selection regret. The pre-specified hypotheses were H1, that family-aware validation would reduce the absolute internal–external MAE gap relative to random CV, and H2, that it would improve agreement with the external model ranking; selection regret was retained as the corresponding decision consequence. Between-regime differences were assessed with a row-level bootstrap (2,000 resamples, fixed seed) over the external cohort. Reporting these statistics for a six-model subset that excludes the two extrapolation-prone architectures, for regime D, and for the SEEN, glass and crystalline subsets are post hoc extensions of the pre-specified protocol.

Following the applicability-domain (AD) framework established for materials ML,<sup>44,45</sup> and the use of feature- or latent-space distance as an uncertainty signal,<sup>46</sup> we computed a model-free, composition-space nearest-neighbour distance (standardized on the development population only) as an AD score for every external row. The AD score was related to each model's absolute external error by Spearman correlation and by a risk-coverage curve summarized by its area (AURC); this is the pre-specified risk-stratification analysis. Benchmarking it against a random-rejection baseline, a Gaussian kernel-density source-domain score,<sup>45</sup> a family-seen/unseen novelty flag and a cross-model prediction-dispersion score, a committee-disagreement signal of the type used to flag out-of-distribution samples,<sup>37</sup> is post hoc. Uncertainty and risk signals of this kind have been compared extensively for chemistry and materials property prediction.<sup>66–71</sup>


### Post hoc sensitivity and audit analyses


Several additional analyses were run after the primary comparison, each specified in a written, hashed plan before its numbers were computed; none reran the external transfer or retrained a final model beyond the pooled elemental-property model correction described in Supplementary Note 27. The final reported analyses use development rows only; one historical label-compatibility sensitivity accessed the OBELiX test labels and is disclosed explicitly below and in the Supplementary Information. They comprise: (1) a DOI-source-disjoint external sensitivity (n = 153); (2) a target-free deployment-shift diagnostic comparing standardized composition-space nearest-neighbour distance distributions between each regime's held-out rows and the external cohorts, later recomputed with a fold-local robustness fix that drops, per fold, any feature dimension whose training-fold variance is below 5% of the leave-fold-out median of sibling folds' training-fold variances (never the shared full-development-population variance, so no held-out fold's feature values inform another fold's reference); (3) a six-model stable-architecture sensitivity excluding ridge and the shallow MLP; (4) row-paired bootstrap 95% CIs on external MAE and P(rank 1); (5) an AD/AURC recomputation on a 16-point coverage grid; (6) constant-predictor baselines and the empirical spread of repeated Hargreaves measurements of the same formula; (7) a fixed 20–30 °C window sensitivity; (8) the family-composition, duplicate-formula and label-compatibility audits described below; (9) repetition of regime A over ten deterministic fold seeds, reported as the observed range across seeds; (10) the glass/crystalline material-class refinement; (11) a common-support sensitivity restricting regimes A and B's internal MAE to the identical 396-row population LOMFO scores; (12) a training-size-matched random control isolating training-set-size loss from family-exclusion distribution shift under Family-GroupKFold; (13) a strictly fold-local recomputation of the robust shift-distance diagnostic's degeneracy-guard reference; (14) a source/DOI-cluster bootstrap (85 clusters) as an alternative to row-level resampling; (15) a `__MISSING__`-family sensitivity (excluding these rows from evaluation, and separately treating each as its own singleton group under Family-GroupKFold); (16) a chemical-system-GroupKFold sensitivity extending the exact-formula-disjoint control to the pymatgen element-set level; and (17) a formula-balanced external-estimand sensitivity giving every unique reduced formula equal total weight, plus a corresponding formula-cluster bootstrap, addressing the possibility that repeated-formula rows could implicitly over-weight the primary row-weighted estimate. Items 11–17 were added for this submission; full detail for each is in Supplementary Information Part I. Items 11–17 reuse the one-shot external predictions and the nested internal-validation results of the primary analysis; the only model refits involved are the pooled elemental-property model's corrective reruns (Supplementary Note 27) and the per-fold fits that items 12, 15 and 16 require by design.


### Family-composition, duplicate-formula and label-compatibility audits


Family composition. A deterministic crosswalk between Hargreaves' and OBELiX's family vocabularies was built by exact string match after normalization, explicitly rejecting one candidate merge on chemical grounds (Hargreaves' 'Zircon' versus OBELiX's 'zirconates'). Seven of fifteen raw Hargreaves family labels matched an OBELiX family present in the development population, seven had no counterpart and one non-informative label ('Other') was scored as ambiguous. Every strict-cohort formula record was classified using the frozen external predictions only.

Exact-formula duplicates. All 449 development-population formulas were canonicalized, and duplicate formulas whose member rows fell in more than one outer fold of the fixed random-CV partition were counted. Regime D is the corresponding exact-formula-disjoint control, following recommendations to control redundancy and leakage in materials datasets.<sup>26,28,29</sup>

Cross-database label compatibility. For every reduced formula present in both databases, OBELiX's log10(σRT) was compared with Hargreaves' value as a calibration check, restricting the OBELiX side to the 449-row development population (test-split labels were not read). These pairs were not used in the strict or DOI-source-disjoint external evaluations; the composition-only pairs are, by construction, part of the relaxed sensitivity cohort.


### Statistical reporting


Statistics use the full available external sample unless stated otherwise (n = 204 strict; n = 153 DOI-source-disjoint; n = 8 or 6 rankable models for regime-ranking statistics, which are therefore directionally informative rather than precise). The strict external cohort was aggregated by reduced formula before prediction (Cohort construction), using the median room-temperature target across retained source measurements for each formula; it therefore contains 204 records representing 204 unique reduced formulas, and primary external metrics assign equal weight to these aggregated formula-level records. As a post hoc identity check, external metrics were recomputed with an explicit equal-weight formula grouping, E_f,m = mean_{i∈f}|y_i − ŷ_{i,m}| and MAE_formula,m = mean_f E_f,m: because the frozen strict cohort already contains exactly one record per reduced formula, the formula-balanced and existing external estimands are mathematically identical, and explicit recomputation confirmed equality for all eight models and all regime-utility point statistics (Supplementary Note 36). Three bootstrap resampling units are used across the manuscript and reported separately: row-level (independent external rows), source/DOI-cluster (formulas sharing a literature source, Supplementary Note 33) and formula-cluster (occurrences of the same reduced formula, Supplementary Note 36). Spearman's ρ and Kendall's τ were computed with SciPy<sup>72</sup> tied-rank estimators and array operations used NumPy;<sup>73</sup> all p-values are two-sided. Confidence intervals are two-sided 95% percentile bootstrap intervals (2,000 resamples, fixed seed); internal nested-validation values are held fixed rather than resampled. The intervals therefore quantify external-cohort sampling variability conditional on the observed internal CV estimates, not joint uncertainty from both development and external populations. Cross-validation estimates the performance of a fitting procedure rather than of one fitted model,<sup>74</sup> so the internal–external comparisons are interpreted as validation-to-deployment alignment rather than as evaluations of an identical fitted object on two datasets. Analyses introduced after inspection of the primary results are treated as exploratory sensitivity analyses. Their confidence intervals quantify uncertainty within each analysis and were not multiplicity-adjusted; they are not interpreted as a family of confirmatory hypothesis tests.


### Use of AI assistance


Claude Code (Anthropic) was used for code implementation, debugging, automated testing, reproducibility checks, and limited manuscript language/revision support; ChatGPT (OpenAI) was used for manuscript language and revision assistance. All study design, hypotheses, methodological decisions, analyses, validation criteria, data interpretation and scientific conclusions were determined and verified by the authors. All AI-assisted outputs were reviewed by the authors, who take full responsibility for the accuracy and integrity of the work. No AI system is listed as an author.


## Results



### Cohort construction and label compatibility


Restricting the 820 Hargreaves measurements to the room-temperature window and excluding rows overlapping OBELiX by composition and/or literature source yielded a strict, composition-disjoint external cohort of 204 rows, above the predeclared minimum of 100 (Table 1). At the literature-source level, 25.0% of the strict cohort's rows trace to a source that also contributes a different composition to OBELiX; the DOI-source-disjoint subset (n = 153) removes these rows.

Table 1 | Cohort definitions and independence.


| Cohort | n | Role | Independence from OBELiX |
| --- | --- | --- | --- |
| OBELiX development population | 449 | Internal development set (train + validation) | Source dataset |
| OBELiX official test split | 113 | Not used for modelling or predictive-performance evaluation | Historical post hoc label-audit exposure disclosed; final lead audit is development-only |
| Hargreaves et al. database | 820 | External source (214 literature sources) | Not restricted |
| Room-temperature window (15–35 °C) | 465 | Predeclared window before exclusions | Not restricted |
| Strict external cohort (primary) | 204 | Primary external test set | Composition-disjoint; not DOI-source-disjoint |
| DOI-source-disjoint subset of strict cohort | 153 | Sensitivity cohort | Composition- and DOI-source-disjoint (51 strict rows removed) |
| Relaxed external cohort | 239 | Companion sensitivity cohort (strict ⊂ relaxed) | Excludes exact composition-and-source matches only |


Restricting the label-compatibility audit to the OBELiX development population gives 166 reduced formulas present in both databases: 135 exact composition-and-DOI matches, which are almost certainly the same underlying measurement, and 31 composition-only matches from independent literature sources. The independent pairs are the informative comparison. They disagree by a mean absolute 0.75 log<sub>10</sub> units (median 0.35), whereas the exact-match pairs disagree by a mean of 0.19 (median ≈0). Across all 166 pairs the agreement is strong in aggregate (Spearman ρ 0.92, 95% CI [0.86, 0.97]; Pearson r 0.96; median absolute discrepancy 0.007; mean 0.29, 95% CI [0.20, 0.39]), but this aggregate is dominated by duplicated measurements. The independent-pair discrepancy is consistent with interlaboratory variability in solid-electrolyte conductivity measurements<sup>75</sup> and with the sensitivity of reported values to impedance-analysis choices such as bulk versus total conductivity.<sup>76</sup> Neither database's headline conductivity is a verified, uniform bulk or total quantity (OBELiX's explicit bulk/total fields are populated for under 5% of rows; Hargreaves has no such field). An earlier version of this audit that used all 599 OBELiX rows, including the test split, gave similar statistics and is retained only as a Supplementary Information sensitivity check.


### External performance in context


External MAE point estimates and row-paired bootstrap 95% CIs were: extremely randomized trees 1.20 [1.04, 1.39], composition attention 1.28 [1.10, 1.46], random forest 1.29 [1.11, 1.48], CatBoost 1.29 [1.12, 1.49], XGBoost 1.35 [1.16, 1.55], ridge 1.38 [1.22, 1.56], shallow MLP 1.48 [1.30, 1.66] and the pooled elemental-property model 1.72 [1.49, 1.95] (Fig. 2). Extremely randomized trees had the lowest point-estimate MAE and retained rank 1 in 78.5% of resamples, but its CI overlaps those of composition attention, random forest and CatBoost, so we do not describe it as uniquely best. Two constant baselines, the development population's mean and median log10(σRT), score MAE 1.67 and 1.74; seven of eight models beat both (skill scores 0.11–0.31 against the mean baseline). The exception, the pooled elemental-property model, beats the median baseline (1.72 < 1.74) but not the mean baseline (1.72 > 1.67); it shows negative R² (−0.21) despite a real, positive rank correlation with the external labels (Spearman ρ = 0.39, Fig. 2; Supplementary Fig. 2) — a systematic scale/offset error, not the literal near-constant collapse an implementation bug in its feature pipeline previously produced (corrected before submission; see Supplementary Note 27).

Fig. 2 | External performance and bootstrap uncertainty. External MAE on the strict, formula-aggregated cohort (n = 204) is shown for all eight models with 95% row-paired bootstrap confidence intervals; MAE weights each of the cohort's 204 aggregated formula-level records equally. The shaded band marks the mean and median constant-predictor baselines (MAE 1.67–1.74). P(rank 1) is the empirical probability of retaining the lowest MAE across bootstrap resamples; overlapping intervals among the leading models preclude a unique-best claim.

Relative to regime A, most models degraded substantially on external transfer: XGBoost (0.81 → 1.35, 1.66×), CatBoost (0.80 → 1.29, 1.61×), extremely randomized trees (0.75 → 1.20, 1.60×) and random forest (0.83 → 1.29, 1.56×), with smaller degradation for the shallow MLP (1.03 → 1.48, 1.44×) and composition attention (1.05 → 1.28, 1.22×). The degradation was not universal: ridge (1.41 → 1.38, ratio 0.98) was essentially unchanged, and the pooled elemental-property model degraded modestly (1.55 → 1.72, ratio 1.11) — more than ridge, but substantially less than the tree-ensemble and attention models. The DOI-source-disjoint subset increased every model's external MAE modestly and uniformly (+0.12 to +0.22) without changing the ranking, and, when evaluated on the same 197-formula subset, narrowing the temperature window from 15–35 °C to 20–30 °C changed every MAE by less than 0.004 (Supplementary Table 1). Among 37 groups of repeated Hargreaves measurements of the same formula, the range of log10 conductivity had a median of 0.49 (IQR [0.15, 1.29]); this literature-level heterogeneity is not a formal noise floor, but it is a non-trivial part of every model's external error.<sup>69,75</sup>


### Regime comparison on the full external cohort


In the pre-specified eight-model comparison, random five-fold CV led on all four statistics by point estimate: the smallest median internal–external MAE gap (0.45 vs. 0.68 for Family-GroupKFold and 0.51 for LOMFO), the highest rank agreement with the external ranking (Spearman ρ = 0.64 vs. 0.55 and 0.57; Kendall τ = 0.50 vs. 0.36 and 0.43) and the lowest top-model selection regret (0 vs. 0.15 and 0.08) (Table 2, Fig. 3). Unlike in an earlier analysis affected by a since-corrected feature-pipeline bug in one of the eight models (Methods, Supplementary Note 27), the row-level bootstrap intervals for the B−A and C−A rank-agreement differences no longer exclude zero at this primary, full-cohort granularity, and neither do the median-gap differences (Table 3); only the B−A top-model-selection-regret difference excludes zero. Random CV's point-estimate lead on all four statistics is therefore not accompanied by primary-comparison statistical significance on three of them; neither hypothesis that family-aware validation improves error estimation (H1) or model ranking (H2) was supported.

Table 2 | Validation-regime utility. Median |gap|, median absolute difference between internal and external MAE (lower is better); Spearman ρ and Kendall τ, rank agreement between internal and external model rankings (higher is better); regret, external-MAE penalty for selecting the internally best model (lower is better). The six-model analysis excludes ridge and the shallow MLP.


| Regime | Model set | Median \|gap\| | Spearman ρ | Kendall τ | Regret |
| --- | --- | --- | --- | --- | --- |
| A: random five-fold CV | Eight models (pre-specified) | 0.4531 | 0.6429 | 0.5000 | 0.0000 |
| B: Family-GroupKFold | Eight models (pre-specified) | 0.6810 | 0.5476 | 0.3571 | 0.1471 |
| C: LOMFO | Eight models (pre-specified) | 0.5149 | 0.5714 | 0.4286 | 0.0802 |
| D: formula-GroupKFold (post hoc control) | Eight models | 0.4009 | 0.6429 | 0.5000 | 0.0000 |
| A: random five-fold CV | Six models (post hoc) | 0.4565 | 0.4857 | 0.3333 | 0.0000 |
| B: Family-GroupKFold | Six models (post hoc) | 0.5930 | 0.0857 | 0.0667 | 0.1471 |
| C: LOMFO | Six models (post hoc) | 0.5053 | 0.3714 | 0.3333 | 0.0802 |
| D: formula-GroupKFold (post hoc control) | Six models (post hoc) | 0.4430 | 0.4857 | 0.3333 | 0.0000 |


Table 3 | Pairwise differences between validation regimes (eight-model comparison). Differences are regime minus regime A; 95% percentile bootstrap intervals over external rows (2,000 resamples). For median |gap| and regret, positive values favour regime A; for ρ and τ, negative values favour regime A.


| Comparison | Statistic | Δ | 95% CI | CI excludes 0 |
| --- | --- | --- | --- | --- |
| B − A | Median \|gap\| | 0.2278 | [−0.0693, 0.5700] | No |
| B − A | Spearman ρ | −0.0952 | [−0.3339, 0.0476] | No |
| B − A | Kendall τ | −0.1429 | [−0.4286, 0.0000] | No |
| B − A | Regret | 0.1471 | [0.0337, 0.2638] | Yes |
| C − A | Median \|gap\| | 0.0618 | [−0.2029, 0.4233] | No |
| C − A | Spearman ρ | −0.0714 | [−0.6190, 0.1667] | No |
| C − A | Kendall τ | −0.0714 | [−0.5000, 0.0714] | No |
| C − A | Regret | 0.0802 | [−0.0997, 0.2557] | No |
| D − A | Median \|gap\| | −0.0522 | [−0.0562, −0.0019] | Yes |
| D − A | Spearman ρ | 0.0000 | [0.0000, 0.0000] | No |
| D − A | Kendall τ | 0.0000 | [0.0000, 0.0000] | No |
| D − A | Regret | 0.0000 | [0.0000, 0.0000] | No |


Because two models extrapolate catastrophically under Family-GroupKFold (see 'Extrapolation failure and shift diagnostic'), we repeated the comparison post hoc on the remaining six models. Random CV led on all four statistics: rank agreement (Spearman ρ = 0.49 vs. 0.09 and 0.37; Kendall τ = 0.33 vs. 0.07 and 0.33), regret (0 vs. 0.15 and 0.08), and median gap (0.46 vs. 0.59 and 0.51). H1 and H2 remained unsupported. The primary result is therefore not driven solely by the two unstable architectures; unlike in an earlier analysis affected by a since-corrected feature-pipeline bug in one of the six retained models (Methods, Supplementary Note 27), no statistic in this six-model subset favours family-aware CV over random CV.

Exact-formula duplicates were common in the development population: 52 formulas were duplicated (136 rows, 30.3%), and 45 of these groups (122 rows, 89.7%) crossed more than one outer fold of the random-CV partition, a known route to optimistic estimates in redundant materials data.<sup>25,26,28</sup> The exact-formula-disjoint control (regime D) changed every model's internal MAE only modestly (largest change: composition attention, +23.9%) and left the ranking essentially unchanged. Its regime-utility statistics (median gap 0.4009, ρ 0.6429, τ 0.5000, regret 0.0000; six-model: 0.4430, 0.4857, 0.3333, 0.0000) closely track random CV. The D−A difference in median gap is small but excludes zero (−0.052, 95% CI [−0.056, −0.002]), and the rank-agreement and regret differences are exactly zero (Table 3). Duplicate-formula leakage therefore does not explain random CV's more favourable read of external performance. This control removes only exact-reduced-formula duplication; it does not control for broader chemical redundancy such as compositions that share an element set without being identical formulas. A chemical-system-grouped sensitivity extending this control to the 153-group pymatgen element-set level (Supplementary Information Part I) shows no extrapolation-failure cost (largest ratio to random CV: 1.30×, against up to 941.8× under Family-GroupKFold) but, unexpectedly, a substantially smaller median |MAE gap| than random CV (0.10 vs. 0.45, 95% CI [−0.43, −0.14], excluding zero) with near-tied rank agreement (Spearman ρ exactly tied; Kendall τ very slightly lower for chemical-system grouping) and slightly worse regret — evidence that some redundancy control beyond exact-formula improves internal–external MAE-magnitude calibration without the family-exclusion extrapolation cost, though not model-selection reliability, and not itself a test of chemistry-family-awareness.

Fig. 3 | Validation-regime utility. Four statistics compare internal validation with strict-cohort external performance for A, random five-fold CV; B, Family-GroupKFold; C, LOMFO; and D, formula-GroupKFold. Filled bars show the pre-specified eight-model comparison and hatched bars the post hoc six-model sensitivity excluding ridge and the shallow MLP. Lower values are better for median absolute MAE gap and regret; higher values are better for rank agreement.

Repeating regime A over ten fold seeds, with regimes B and C fixed, showed that the four statistics differ in robustness. In the eight-model comparison, random CV's median-gap advantage held robustly across the ten-seed range (range [0.379, 0.453] vs. 0.681 and 0.515), but its rank-agreement advantage did not: the ten-seed Spearman (range [0.452, 0.762]) and Kendall (range [0.357, 0.571]) ranges now contain both family-aware regimes' fixed values (Spearman 0.548 and 0.571; Kendall 0.357 and 0.429), so at least some random-CV fold realizations show rank agreement no better than the family-aware regimes. Regret remained the least fold-robust statistic, with its range [0.00, 0.15] overlapping LOMFO's 0.08. In the six-model sensitivity, the median-gap range (unchanged at [0.414, 0.467]) is now robustly below both family-aware regimes' fixed values (0.593 and 0.505) rather than above them, reversing the one statistic that previously favoured family-aware CV in this subset; Spearman and Kendall's ten-seed ranges ([0.20, 0.60] and [0.07, 0.47]) now contain LOMFO's fixed value (0.37 and 0.33 respectively), echoing the same loss of robustness seen in the eight-model comparison, though Family-GroupKFold's fixed value remains outside both ranges. Random CV's median-gap advantage in the eight-model comparison is thus the single most fold-robust part of the comparison; its rank-agreement advantage, previously reported as similarly robust, is not, in either the eight- or six-model view.

Two further checks probe whether this primary result depends on incidental features of the evaluation design rather than the validation regimes themselves (Supplementary Information Part I). First, LOMFO scores only the 396 rows belonging to a major family, while regimes A and B score all 449; restricting A and B's internal MAE to this identical 396-row common support leaves the median-|MAE-gap| point estimate favouring A, as in the full-cohort comparison, but by a much narrower margin (0.61 vs. 0.64 for B, against 0.45 vs. 0.68 under full support), and its 95% CI still includes zero. Rank agreement moves the other way: the common-support B-minus-A differences exclude zero in A's favour (Spearman −0.29 [−0.55, −0.10]; Kendall −0.36 [−0.64, −0.07]), unlike the full-cohort primary comparison, where neither excludes zero after this correction (Table 3). Second, a source/DOI-cluster bootstrap (85 clusters over the 204-row strict cohort, replacing independent row-level resampling) widens every model's external-MAE CI and all but one regime-comparison CI (the B-minus-A Kendall-tau interval is unchanged) but changes no comparison's zero-exclusion status relative to the already-corrected row-level primary result (every B/C-minus-A rank-agreement CI already includes zero at the row level; Table 3), while leaving every point estimate and the median-gap/regret comparisons unchanged. Taken together, the rank-agreement component of "random CV wins" is not significant in the primary, full-support, row-level comparison itself, is restored by restricting to LOMFO's common support, and is unaffected one way or the other by source/DOI clustering; the median-gap component favours random CV by point estimate throughout but is not statistically significant under any of these three resampling/support choices, and narrows further under common support.


Family-label construction was itself consequential. In a post hoc sensitivity, the 18 rows carrying the `__MISSING__` family sentinel were either excluded from evaluation or, for Family-GroupKFold, assigned to singleton groups. Under the singleton treatment, the catastrophic ridge and shallow-MLP Family-GroupKFold blowups disappeared, and Family-GroupKFold's already-weak rank agreement with the external ranking fell further, from Spearman ρ = 0.55 to 0.19 (Kendall τ 0.36 to 0.14), excluding zero in random CV's favour. H1 and H2 remained unsupported, but the result shows that the most extreme family-held-out failures depend on how missing family labels are represented; they should not be interpreted as an invariant property of Family-GroupKFold itself (Supplementary Information Part I).


As a post hoc identity check, external metrics were recomputed with explicit equal-weight formula grouping, addressing whether the primary external estimate could be vulnerable to repeated-formula over-weighting if several retained measurements had shared one reduced formula. The strict cohort's own construction already forecloses this for the present analysis: `aggregate_by_formula` collapses the 213 pre-aggregation retained rows sharing a reduced formula to one aggregated formula-level record, using the median target, before the one-shot external evaluation is run, so the 204-record strict cohort already has exactly 204 unique reduced formulas and zero repeated-formula groups (verified directly from the frozen cohort file and from all eight models' frozen predictions, not assumed). Because the frozen strict cohort already contains exactly one record per reduced formula, the formula-balanced and existing external estimands are mathematically identical; explicit recomputation confirmed equality for all eight models and all regime-utility point statistics, and a formula-cluster bootstrap (2,000 resamples, same seed and resample count as the row-level and source/DOI-cluster bootstraps) reproduced the already-reported pairwise regime-utility point deltas exactly and the already-reported confidence-interval endpoints to floating-point precision as an observed match, not an asserted one (Supplementary Note 36). This is reported as a positive, checked finding rather than an assumption: repeated-formula weighting is not a vulnerability of the strict-cohort external estimate reported here, because formula-level aggregation already happens upstream of prediction. None of the paper's central conclusions change as a result.


### Deployment scenario: seen families, unseen glasses and unseen crystalline families


Crosswalking family labels classified 138 of the 204 strict-cohort formula records (67.6%) as SEEN, 55 (27.0%) as UNSEEN and 11 (5.4%) as ambiguous. Two families, NASICON (61/204, 29.9%) and perovskite (45/204, 22.1%), make up 52.0% of the cohort; these are the same families whose exclusion folds produced the extrapolation failures described below. The external cohort is therefore dominated by formula records from families represented in development data. Every model's external MAE was substantially higher on UNSEEN than on SEEN rows, with non-overlapping 95% CIs for all eight models (for example, extremely randomized trees 0.83 [0.71, 0.97] → 1.96 [1.52, 2.40] and the pooled elemental-property model 1.12 [0.93, 1.31] → 3.12 [2.59, 3.63]); per-model values for the SEEN, glass and crystalline subsets are given in Table 4.

On the SEEN subset, random CV anticipated external error closely (median gap 0.09, vs. 0.97 for Family-GroupKFold and 0.97 for LOMFO). The 55-row UNSEEN subset is not homogeneous: 36 rows (65.5%) are glass or glass-ceramic, an amorphous material class absent from the crystal-structure-based OBELiX database, and 19 (34.5%) are crystalline compositions from families absent from training. Composition-only descriptors cannot distinguish glassy from crystalline forms of similar stoichiometry, a recognized difficulty for data-driven glass science.<sup>1,77,78</sup> The crystalline subset falls below the pre-declared threshold for any CI or regime comparison; its per-model MAE (mean 2.67, range 1.86–4.48; Table 4) is reported descriptively only.

Fig. 4 | Validation utility depends on deployment scenario. a, Median absolute internal-to-external MAE gap for random five-fold CV, Family-GroupKFold and LOMFO on seen-family rows (n = 138) and the unseen glass/glass-ceramic subset (n = 36). Family-aware regimes more closely estimate error magnitude for the glass subset, whereas random CV closely matches seen-family transfer. b, Per-model external MAE for the crystalline unseen-family subset (n = 19), reported descriptively because the subset is below the predeclared inference threshold.

For the glass subset, random CV had the largest median gap (1.13), against 0.41 for Family-GroupKFold and 0.15 for LOMFO (Fig. 4a). The Family-GroupKFold−random and LOMFO−random differences were −0.72 (95% CI [−0.83, −0.37]) and −0.98 (95% CI [−1.09, −0.37]), both excluding zero. The rank-agreement and regret differences, by contrast, all had CIs including zero (Spearman ρ difference +0.17 [−0.36, 0.48] and +0.24 [−0.62, 0.64]; Kendall τ 0.00 [−0.43, 0.43] and +0.07 [−0.64, 0.50]; regret +0.05 [−0.28, 0.39] and −0.08 [−0.79, 0.65]). On this subset, family-aware validation therefore anticipated the magnitude of external error but not which model to select. The formula-GroupKFold control showed no CI-supported difference from random CV on the same subset (median-gap difference −0.05, 95% CI [−0.09, 0.03]).

The pre-declared framing rule mechanically labels this pattern 'unseen material class (amorphous), not chemistry novelty'; we use the more conservative evidentiary interpretation here rather than that label, because the unseen-chemistry branch cannot itself be evaluated — it requires a crystalline-unseen comparison the sample does not support (n = 19, below the pre-declared threshold). The supported effect is specific to the unseen amorphous glass/glass-ceramic subset; whether it generalizes to unseen crystalline chemistry remains unresolved. Only 8 of 36 (22.2%) glass rows share even a chemical system (element set) with any development-population row, which illustrates why this subset is hard; we do not treat this as a leakage finding.

Table 4 | External MAE by deployment scenario. MAE in log10(S cm⁻¹) with 95% bootstrap CIs for the SEEN-family (n = 138) and unseen glass/glass-ceramic (n = 36) subsets. The unseen crystalline subset (n = 19) is below the pre-declared n ≥ 20 threshold and is reported as a point estimate only, descriptively. Eleven formula records with a non-informative family label are excluded.


| Model | SEEN families | Unseen glass/glass-ceramic | Unseen crystalline (descriptive) |
| --- | --- | --- | --- |
| Extremely randomized trees | 0.83 [0.71, 0.97] | 1.92 [1.38, 2.48] | 2.05 |
| CatBoost | 0.84 [0.72, 0.97] | 2.06 [1.55, 2.61] | 2.40 |
| Random forest | 0.86 [0.74, 1.00] | 2.08 [1.58, 2.63] | 2.23 |
| Composition attention | 0.89 [0.76, 1.04] | 1.84 [1.44, 2.26] | 2.61 |
| XGBoost | 0.92 [0.78, 1.07] | 1.97 [1.45, 2.51] | 2.33 |
| Shallow MLP | 1.00 [0.84, 1.16] | 2.13 [1.76, 2.51] | 3.40 |
| Ridge | 1.11 [0.96, 1.26] | 1.91 [1.39, 2.49] | 1.86 |
| Pooled elemental-property model | 1.12 [0.93, 1.31] | 2.40 [1.94, 2.87] | 4.48 |



### Extrapolation failure and shift diagnostic


Under Family-GroupKFold, ridge and the shallow MLP produced catastrophic pooled MAEs (143.1 and 969.4, i.e. 101.5× and 941.8× their best MAE under either other regime), driven almost entirely by the NASICON (n = 130; ridge 412.7, shallow MLP 2833.6) and perovskite (n = 26; ridge 380.9, shallow MLP 2550.8) exclusion folds (Fig. 5). Every other model stayed below 2.5×. A linear model's unbounded output map and an unconstrained ReLU network's final linear layer do not constrain predictions to the training-target range and can therefore yield extreme extrapolations when an entire family's feature distribution is removed from training, consistent with the extrapolation risks identified for cluster-based splits;<sup>32,37</sup> bagged tree ensembles are range-bounded by construction, and the gradient-boosted models remained empirically stable here. A training-size-matched control (Supplementary Information Part I) helps separate training-size loss from family-exclusion shift: a plain random holdout matched to each Family-GroupKFold fold's exact test size, with identical nested retuning, keeps both ridge and the shallow MLP within the same order of magnitude as random CV (ridge: 1.41, 1.30, 1.28 across three control seeds, vs. 1.41 for regime A and 143.1 for regime B; shallow MLP: 1.15, 1.15, 1.14, vs. 1.03 for regime A and 969.4 for regime B), supporting the interpretation that family-exclusion distribution shift, rather than training-size reduction alone, is the primary driver of the blowup; the control does not by itself establish a unique causal mechanism.

Fig. 5 | Family-holdout extrapolation failure under the primary family grouping is architecture-specific. a, Per-family Family-GroupKFold MAE for ridge and the shallow MLP, with Extra Trees as a well-behaved reference; the failure is dominated by the NASICON (n = 130) and perovskite (n = 26) exclusion folds. b, Family-GroupKFold pooled-MAE blowup relative to each model's best pooled MAE under random five-fold CV or LOMFO. The catastrophic effect is concentrated in ridge and the shallow MLP; all other models remain at or below 2.5-fold.

A target-free diagnostic tested whether the family-aware regimes' held-out rows resemble the external shift more closely than random CV's. With a robustness fix that removes near-degenerate training-fold feature dimensions, an earlier extreme Family-GroupKFold outlier (maximum standardized distance ≈94,560, traced to a magnetic-moment feature with near-zero training-fold variance once all magnetic-element families were excluded) fell to 26.6, consistent with that regime's 90th percentile (9.2), without changing which regime is closest to the external cohort by Wasserstein distance (LOMFO before and after; Supplementary Fig. 1). Median distances were 0.21 for random CV, 4.43 for Family-GroupKFold and 4.26 for LOMFO, against 3.01 (strict) and 3.79 (DOI-source-disjoint) for the external cohorts, and LOMFO was the closest match by robust Wasserstein distance (1.10 vs. strict; 0.86 vs. DOI-source-disjoint). The family novelty fraction was 3.1% for random CV and 100% for both family-aware regimes. This shift alignment did not translate into better rank agreement or model selection for LOMFO in either the primary or six-model comparison, so we treat it as a diagnostic of the extrapolation created by family exclusion rather than as a predictor of which regime best anticipates external performance.


### Risk stratification: applicability domain, family flag and dispersion


The composition-space AD score, fitted on the development population without external labels, stratified external risk for all eight models. On the 16-point coverage grid, restricting predictions to progressively more in-domain rows monotonically reduced pooled MAE for every model; AURC ranged from 0.92 (extremely randomized trees) to 1.25 (pooled elemental-property model), and for extremely randomized trees the most in-domain quarter of rows had MAE 0.67 against 1.20 at full coverage. All eight models showed positive Spearman correlations between AD distance and absolute external error (r = 0.24–0.43, all p < 0.001).

Fig. 6 | Prediction-time risk stratification. a, Mean area under the risk-coverage curve (AURC; lower is better) across eight models for five risk scores on the strict external cohort. Cross-model prediction dispersion is the strongest tested comparator; the composition-space applicability-domain score performs similarly to the simple family-novelty flag. b, Spearman association between composition-space AD distance and absolute external error for each model; all associations are positive.

On a four-point coverage grid shared by all comparators (Fig. 6), the AD score beat random rejection for every model but beat the plain family-novelty flag in only three of eight models, and the two gave nearly identical mean AURC (1.05 vs. 1.06). The composition-space AD signal is therefore largely coincident with family novelty on this task, in line with evidence that kernel-density and distance-based domain scores separate chemically distinct groups.<sup>44,45</sup> A cross-model prediction-dispersion score, used as an ensemble-disagreement proxy because per-row five-seed dispersion was not stored, was the strongest comparator (mean AURC 0.97 vs. 1.05–1.10 for the other non-trivial comparators), consistent with reports that committee disagreement flags out-of-distribution samples.<sup>37,66,68</sup> We report this as an incidental finding rather than a recommended method.


## Discussion


The central result is deployment-conditional: how well internal validation predicts external performance depends on the shift and performance quantity it is intended to approximate. In the primary full-cohort point estimates, which are dominated by families already represented in development data, random CV most closely anticipated external error and model ranking. That advantage was less definitive than the point estimates suggest: neither its median-gap nor its rank-agreement differences excluded zero in the primary row-level comparison, the common-support sensitivity narrowed the median-gap margin further, and the source/DOI-cluster bootstrap widened intervals without changing any comparison's zero-exclusion status. On seen-family rows random CV's median internal-to-external MAE gap was only 0.09 log units. In contrast, for the unseen glass/glass-ceramic subset, family-aware validation more closely anticipated the magnitude of the much larger external error, although it still did not identify the best model. The chemical-system-grouped control provides a complementary result: grouping by element set produced a much smaller median |MAE gap| than random CV while tying its rank agreement, showing that grouped validation is not a single operation and that the choice of grouping variable can alter which aspect of external performance is reproduced. The subset and grouping sensitivities therefore qualify the full-cohort result rather than reversing it: no regime was uniformly superior across error-magnitude calibration and model-selection reliability.

This result complements rather than contradicts earlier work. Leave-one-cluster-out studies, MatFold and Hargreaves et al. showed that stricter splits give more honest, usually worse, internal accuracy numbers,<sup>11,32,35</sup> and we observe the same (Family-GroupKFold gave worse pooled MAE than random CV for all eight models, and LOMFO for seven). Our question is different: how well an internal regime's error estimate and ranking predict performance on a composition-disjoint external cohort curated through a different database pipeline. Here the stricter split was internally harder but not externally more predictive of ranking, a distinction that single-dataset studies cannot expose. Matching the marginal distribution of composition-space shift was also insufficient: LOMFO's distance distribution was closest to the external cohorts, yet it did not improve rank agreement or selection. Whether a regime imposes a realistic amount of shift and whether its outcome predicts external performance are empirically separable questions, as expected from the fact that cross-validation estimates the behaviour of a fitting procedure rather than of a particular deployed model.<sup>74</sup>

The duplicate-formula audit and the label-compatibility audit address the two most obvious alternative explanations. Exact-formula leakage in random CV was real but did not drive the result, because the exact-formula-disjoint control reproduced random CV's regime-utility statistics. Label heterogeneity is large: independently sourced measurements of the same nominal composition disagree by a mean of 0.75 log10 units, comparable to the best models' error on seen families (0.83–0.92 for the tree ensembles and composition attention), and interlaboratory studies show that conductivity values for identical samples can differ substantially.<sup>75,76</sup> Because the strict cohort is aggregated to one record per reduced formula before prediction, the primary external estimand already weights every formula equally; the within-formula literature-measurement spread described above instead reflects heterogeneity among the raw source measurements that were collapsed into each formula's median target, not an artifact of how the final evaluation weights repeated formulas. This was checked directly rather than assumed away: the strict cohort already contains zero repeated-formula groups, so an explicit equal-weight formula-balanced recomputation reproduces the primary result exactly rather than merely approximately (Supplementary Note 36). Measurement and label heterogeneity limit the observable accuracy of any model,<sup>69</sup> so part of the residual external error in the within-family regime reflects measurement heterogeneity rather than model failure. The same consideration argues for caution when interpreting small differences in external MAE between leading models, as in other settings where benchmark and deployment data differ in how they were collected.<sup>30,31,37</sup>

Prediction-time risk scores were useful but less distinctive than expected. The composition-space AD score stratified external risk for every model, yet it carried little information beyond knowing whether a composition's family appeared in training, and, in a post hoc comparison, simple cross-model dispersion stratified risk better. For materials-property models deployed on new external data, this suggests a practical workflow: decide which kind of novelty the intended candidates represent; use random or formula-grouped CV to estimate within-family performance; recognize that family-holdout CV may provide a more conservative error estimate when deployment enters material classes absent from training, as observed here for the amorphous subset; and accompany every prediction with inexpensive novelty and disagreement flags.<sup>37,45,68</sup>


## Limitations


Composition-only features discard the structural and disorder information that OBELiX carries, but they are the only representation available identically in both databases. External MAE values (about 1.2–1.8 log10 units) are large in absolute terms, and part of this magnitude reflects measurement and protocol heterogeneity in the source literature rather than model error alone; the repeated-measurement spread is not a formally derived noise floor. External evaluation is performed on formula-level, median-aggregated targets: the primary external estimand already weights every aggregated reduced-formula record equally, not raw literature measurements. The separate source-measurement heterogeneity analysis above remains relevant because raw literature measurements can vary substantially before aggregation, even though that variability is absorbed into each formula's median target rather than left in the evaluated cohort. A post hoc equal-weight formula-grouping identity check (not part of the frozen protocol) was run and found to be an exact identity for this cohort (Supplementary Note 36), because the strict cohort is already built as one record per unique reduced formula; this closes the concern for the present cohort but is not a general guarantee that row weighting and formula weighting will coincide in a future, differently constructed external cohort, and that assumption should be re-checked explicitly, not presumed, if this evaluation is repeated on a cohort built without formula-per-record aggregation. Accordingly, this study does not establish deployment-ready conductivity prediction; its conclusions concern validation-deployment alignment under this cross-dataset setting rather than absolute predictive readiness.

All findings come from one dataset pairing, one target property and eight architectures. They were replicated under source- and composition-overlap sensitivity cohorts and, for the full-cohort result, under a six-model sensitivity, but we do not claim that family-aware CV is generally worse than random CV or that its glass-subset advantage generalizes. The crystalline unseen-family subset (n = 19) was pre-declared too small for any regime-ranking statistic, so whether family-aware validation helps for genuinely unseen crystalline chemistry remains unresolved.

The explanation for the extrapolation failures (range-boundedness of bagged trees) is empirical rather than proven, and why the gradient-boosted and attention models did not fail is not fully characterized. Composition-only and source-only literature overlaps could not be ruled out as non-independent measurements; both were tested with dedicated sensitivity cohorts without changing any headline conclusion. Rank statistics computed over eight (or six) models are coarse, and with three regimes and four statistics some regime must win at least two; the observed result exceeds that bar in the primary comparison, but small-n rank values remain directionally informative. The row-level bootstrap used for the primary regime-utility CIs treats external rows as independent; a source/DOI-cluster bootstrap (Supplementary Information Part I) widens every model-level interval and all but one regime-comparison interval, so the row-level CIs understate uncertainty, although no comparison changes its zero-exclusion status and every point estimate is unchanged. These CIs also condition on the observed internal nested-CV estimates and therefore quantify external-cohort sampling variability rather than joint uncertainty from both development and external populations. External predictions average five fixed-seed final refits, whereas internal values are nested-CV procedure estimates; the comparison therefore evaluates validation-to-deployment alignment rather than the same fitted object under two datasets. The deployment-shift diagnostic uses a simple standardized Euclidean distance rather than a validated general shift measure. Regime B's rank agreement in the primary comparison is itself not robust to how the 18-row `__MISSING__` (no-family-label) sentinel is grouped: treating each such row as its own singleton group, rather than one shared group, removes the catastrophic ridge/shallow-MLP blowups but drives Family-GroupKFold's already-modest rank agreement with external ranking substantially lower, this time excluding zero in random CV's favour (Supplementary Information Part I) — H1/H2 remain unsupported either way, but this is a genuine added fragility, not merely a robustness check that passed. Finally, neither database's primary conductivity is a verified, uniform bulk or total quantity, and independent same-formula measurements disagree substantially even though aggregate agreement is strong.


## Conclusion


In this OBELiX-to-Hargreaves transfer study, how internal evaluation predicted external generalization was not ordered by split severity alone; it depended on the deployment scenario and the performance quantity being compared. Random CV closely anticipated seen-family external error under the primary formula-level external estimand, whereas family-aware validation more closely estimated error magnitude for an unseen amorphous material class without improving model ranking. Exact-formula leakage did not explain the full-cohort result, and cross-database measurement heterogeneity contributes materially to the remaining external error. Composition-space applicability-domain distance stratified risk but largely duplicated a simple family-novelty flag, while cross-model prediction dispersion was the strongest tested post hoc risk signal. Together, these results support a practical principle for materials machine learning: internal validation should be chosen to reproduce the novelty and decision target expected at deployment, and model predictions should be accompanied by inexpensive novelty or disagreement diagnostics rather than relying on any single split as a universally realistic benchmark.


## Supporting Information


The Supporting Information is available free of charge. It contains Supplementary Notes 1–36, Supplementary Figs. 1–2 and Supplementary Table 1, covering external dataset provenance, cohort construction and cross-database label compatibility (Notes 1–5); per-model external-transfer, nested internal-validation and regime-utility results (Notes 6–10); robustness controls, including the DOI-source-disjoint, temperature-window, constant-baseline, six-model, duplicate-formula and ten-seed sensitivities (Notes 11–16); the family crosswalk and glass/crystalline refinement of the unseen subset (Notes 17–18); shift-distance diagnostics (Notes 19–20); applicability-domain and risk-stratification benchmarks (Notes 21–23); model hyperparameters and implementation details (Notes 24–25); the protocol hash lock, the chronology of both implementation corrections and the figure, table and source-record provenance maps (Notes 26–29); and additional post hoc robustness controls (Notes 30–36).


## Data availability


OBELiX<sup>12</sup> and the Hargreaves et al. conductivity database<sup>11</sup> were used under their respective terms and are not redistributed beyond what those terms permit; the Hargreaves data are licensed for academic use with attribution to the University of Liverpool. Provenance for every raw file (source DOI, retrieval date, licence, file hash, row count and parser version), model predictions, evaluation metrics, the pre-specified protocol and all sensitivity-analysis outputs are described in the Supplementary Information and are included, with the code, in the public repository and reproducibility archive (see Code availability). Because the Hargreaves terms do not grant redistribution, the public archive does not include the raw Hargreaves database, the row-level overlap ledger or the formula-level external cohort files, and the per-formula external conductivity values and residuals are removed from the archived prediction records. Users who obtain the raw database under its terms can rebuild all of these and verify them byte for byte against the SHA-256 hashes recorded in the archive, following its reconstruction instructions. No new experimental measurements or atomistic-simulation data were generated.


## Code availability


The code for cohort construction, nested internal validation, one-shot external transfer, sensitivity analyses and figure and table generation is publicly available under the MIT licence at https://github.com/Rifat19R/solid-electrolyte-validation-deployment. Version 1.0.0 of the reproducibility archive is to be deposited on Zenodo under the reserved DOI 10.5281/zenodo.23987392, which resolves once the deposit is published.


## Author contributions


M.R.K.: conceptualization, methodology, software, formal analysis, investigation, data curation, visualization, writing – original draft, writing – review and editing. M.A.C.: supervision, project administration, writing – review and editing.


## Competing interests


The authors declare no competing interests.


## Acknowledgements


M.R.K. conducted this research under the supervision of Prof. Dr. Mohammad Asaduzzaman Chowdhury at the Department of Mechanical Engineering, Dhaka University of Engineering & Technology (DUET). This research received no external funding. Claude Code (Anthropic) and ChatGPT (OpenAI) were used as described in Methods ("Use of AI assistance"); all AI-assisted outputs were reviewed by the authors, who take full responsibility for the work.


## References


1. Famprikis, T., Canepa, P., Dawson, J. A., Islam, M. S. & Masquelier, C. Fundamentals of inorganic solid-state electrolytes for batteries. <i>Nat. Mater.</i> <b>18</b>, 1278–1291 (2019).

2. Janek, J. & Zeier, W. G. Challenges in speeding up solid-state battery development. <i>Nat. Energy</i> <b>8</b>, 230–240 (2023).

3. Randau, S. et al. Benchmarking the performance of all-solid-state lithium batteries. <i>Nat. Energy</i> <b>5</b>, 259–270 (2020).

4. Sendek, A. D. et al. Machine learning-assisted discovery of solid Li-ion conducting materials. <i>Chem. Mater.</i> <b>31</b>, 342–352 (2019).

5. Cubuk, E. D., Sendek, A. D. & Reed, E. J. Screening billions of candidates for solid lithium-ion conductors: a transfer learning approach for small data. <i>J. Chem. Phys.</i> <b>150</b>, 214701 (2019).

6. Zhang, Y. et al. Unsupervised discovery of solid-state lithium ion conductors. <i>Nat. Commun.</i> <b>10</b>, 5260 (2019).

7. Sendek, A. D., Cheon, G., Pasta, M. & Reed, E. J. Quantifying the search for solid Li-ion electrolyte materials by anion: a data-driven perspective. <i>J. Phys. Chem. C</i> <b>124</b>, 8067–8079 (2020).

8. Laskowski, F. A. L., McHaffie, D. B. & See, K. A. Identification of potential solid-state Li-ion conductors with semi-supervised learning. <i>Energy Environ. Sci.</i> <b>16</b>, 1264–1276 (2023).

9. Kim, J., Mok, D. H., Kim, H. & Back, S. Accelerating the search for new solid electrolytes: exploring vast chemical space with machine learning-enabled computational calculations. <i>ACS Appl. Mater. Interfaces</i> <b>15</b>, 52427–52435 (2023).

10. Ahmad, Z., Xie, T., Maheshwari, C., Grossman, J. C. & Viswanathan, V. Machine learning enabled computational screening of inorganic solid electrolytes for suppression of dendrite formation in lithium metal anodes. <i>ACS Cent. Sci.</i> <b>4</b>, 996–1006 (2018).

11. Hargreaves, C. J. et al. A database of experimentally measured lithium solid electrolyte conductivities evaluated with machine learning. <i>npj Comput. Mater.</i> <b>9</b>, 9 (2023).

12. Therrien, F. et al. OBELiX: a curated dataset of crystal structures and experimentally measured ionic conductivities for lithium solid-state electrolytes. <i>Digit. Discov.</i> <b>5</b>, 910–918 (2026).

13. Butler, K. T., Davies, D. W., Cartwright, H., Isayev, O. & Walsh, A. Machine learning for molecular and materials science. <i>Nature</i> <b>559</b>, 547–555 (2018).

14. Schmidt, J., Marques, M. R. G., Botti, S. & Marques, M. A. L. Recent advances and applications of machine learning in solid-state materials science. <i>npj Comput. Mater.</i> <b>5</b>, 83 (2019).

15. Choudhary, K. et al. Recent advances and applications of deep learning methods in materials science. <i>npj Comput. Mater.</i> <b>8</b>, 59 (2022).

16. Xu, P., Ji, X., Li, M. & Lu, W. Small data machine learning in materials science. <i>npj Comput. Mater.</i> <b>9</b>, 42 (2023).

17. Pyzer-Knapp, E. O. et al. Accelerating materials discovery using artificial intelligence, high performance computing and robotics. <i>npj Comput. Mater.</i> <b>8</b>, 84 (2022).

18. Wang, A. Y.-T. et al. Machine learning for materials scientists: an introductory guide toward best practices. <i>Chem. Mater.</i> <b>32</b>, 4954–4965 (2020).

19. Artrith, N. et al. Best practices in machine learning for chemistry. <i>Nat. Chem.</i> <b>13</b>, 505–508 (2021).

20. Dunn, A., Wang, Q., Ganose, A., Dopp, D. & Jain, A. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. <i>npj Comput. Mater.</i> <b>6</b>, 138 (2020).

21. Choudhary, K. et al. JARVIS-Leaderboard: a large scale benchmark of materials design methods. <i>npj Comput. Mater.</i> <b>10</b>, 93 (2024).

22. Bartel, C. J. et al. A critical examination of compound stability predictions from machine-learned formation energies. <i>npj Comput. Mater.</i> <b>6</b>, 97 (2020).

23. Riebesell, J. et al. A framework to evaluate machine learning crystal stability predictions. <i>Nat. Mach. Intell.</i> <b>7</b>, 836–847 (2025).

24. Borg, C. K. H. et al. Quantifying the performance of machine learning models in materials discovery. <i>Digit. Discov.</i> <b>2</b>, 327–338 (2023).

25. Li, K. et al. Exploiting redundancy in large materials datasets for efficient machine learning with less data. <i>Nat. Commun.</i> <b>14</b>, 7283 (2023).

26. Li, Q., Fu, N., Omee, S. S. & Hu, J. MD-HIT: machine learning for material property prediction with dataset redundancy control. <i>npj Comput. Mater.</i> <b>10</b>, 245 (2024).

27. Omee, S. S., Fu, N., Dong, R., Hu, M. & Hu, J. Structure-based out-of-distribution (OOD) materials property prediction: a benchmark study. <i>npj Comput. Mater.</i> <b>10</b>, 144 (2024).

28. Ottomano, F., De Felice, G., Gusev, V. V. & Sparks, T. D. Not as simple as we thought: a rigorous examination of data aggregation in materials informatics. <i>Digit. Discov.</i> <b>3</b>, 337–346 (2024).

29. Kapoor, S. & Narayanan, A. Leakage and the reproducibility crisis in machine-learning-based science. <i>Patterns</i> <b>4</b>, 100804 (2023).

30. Recht, B., Roelofs, R., Schmidt, L. & Shankar, V. Do ImageNet classifiers generalize to ImageNet? In <i>Proc. 36th International Conference on Machine Learning</i> (eds Chaudhuri, K. & Salakhutdinov, R.) Vol. 97, 5389–5400 (PMLR, 2019).

31. Koh, P. W. et al. WILDS: a benchmark of in-the-wild distribution shifts. In <i>Proc. 38th International Conference on Machine Learning</i> (eds Meila, M. & Zhang, T.) Vol. 139, 5637–5664 (PMLR, 2021).

32. Meredig, B. et al. Can machine learning identify the next high-temperature superconductor? Examining extrapolation performance for materials discovery. <i>Mol. Syst. Des. Eng.</i> <b>3</b>, 819–825 (2018).

33. Xiong, Z. et al. Evaluating explorative prediction power of machine learning algorithms for materials discovery using k-fold forward cross-validation. <i>Comput. Mater. Sci.</i> <b>171</b>, 109203 (2020).

34. Durdy, S., Gaultois, M. W., Gusev, V. V., Bollegala, D. & Rosseinsky, M. J. Random projections and kernelised leave one cluster out cross validation: universal baselines and evaluation tools for supervised machine learning of material properties. <i>Digit. Discov.</i> <b>1</b>, 763–778 (2022).

35. Witman, M. D. & Schindler, P. MatFold: systematic insights into materials discovery models' performance through standardized cross-validation protocols. <i>Digit. Discov.</i> <b>4</b>, 625–635 (2025).

36. Wu, Z. et al. MoleculeNet: a benchmark for molecular machine learning. <i>Chem. Sci.</i> <b>9</b>, 513–530 (2018).

37. Li, K., DeCost, B., Choudhary, K., Greenwood, M. & Hattrick-Simpers, J. A critical examination of robustness and generalizability of machine learning prediction of materials properties. <i>npj Comput. Mater.</i> <b>9</b>, 55 (2023).

38. Li, K. et al. Probing out-of-distribution generalization in machine learning for materials. <i>Commun. Mater.</i> <b>6</b>, 9 (2025).

39. Segal, N., Netanyahu, A., Greenman, K. P., Agrawal, P. & Gómez-Bombarelli, R. Known unknowns: out-of-distribution property prediction in materials and molecules. <i>npj Comput. Mater.</i> <b>11</b>, 345 (2025).

40. Hu, J., Liu, D., Fu, N. & Dong, R. Realistic material property prediction using domain adaptation based machine learning. <i>Digit. Discov.</i> <b>3</b>, 300–312 (2024).

41. Shimakawa, H., Kumada, A. & Sato, M. Extrapolative prediction of small-data molecular property using quantum mechanics-assisted machine learning. <i>npj Comput. Mater.</i> <b>10</b>, 11 (2024).

42. Muckley, E. S., Saal, J. E., Meredig, B., Roper, C. S. & Martin, J. H. Interpretable models for extrapolation in scientific machine learning. <i>Digit. Discov.</i> <b>2</b>, 1425–1435 (2023).

43. Tan, L., Wang, X., Zou, Y., Chen, P. & Zou, Q. MatUQ: a benchmark for uncertainty-aware out-of-distribution materials property prediction with graph neural networks. <i>npj Comput. Mater.</i> https://doi.org/10.1038/s41524-026-02272-x (2026).

44. Sutton, C. et al. Identifying domains of applicability of machine learning models for materials science. <i>Nat. Commun.</i> <b>11</b>, 4428 (2020).

45. Schultz, L. E., Wang, Y., Jacobs, R. & Morgan, D. A general approach for determining applicability domain of machine learning models. <i>npj Comput. Mater.</i> <b>11</b>, 95 (2025).

46. Janet, J. P., Duan, C., Yang, T., Nandy, A. & Kulik, H. J. A quantitative uncertainty metric controls error in neural network-driven chemical discovery. <i>Chem. Sci.</i> <b>10</b>, 7913–7922 (2019).

47. Ong, S. P. et al. Python Materials Genomics (pymatgen): a robust, open-source Python library for materials analysis. <i>Comput. Mater. Sci.</i> <b>68</b>, 314–319 (2013).

48. Jha, D. et al. ElemNet: deep learning the chemistry of materials from only elemental composition. <i>Sci. Rep.</i> <b>8</b>, 17593 (2018).

49. Jha, D. et al. Enhancing materials property prediction by leveraging computational and experimental data using deep transfer learning. <i>Nat. Commun.</i> <b>10</b>, 5316 (2019).

50. Gupta, V. et al. Cross-property deep transfer learning framework for enhanced predictive analytics on small materials data. <i>Nat. Commun.</i> <b>12</b>, 6595 (2021).

51. De Breuck, P.-P., Hautier, G. & Rignanese, G.-M. Materials property prediction for limited datasets enabled by feature selection and joint learning with MODNet. <i>npj Comput. Mater.</i> <b>7</b>, 83 (2021).

52. Goodall, R. E. A. & Lee, A. A. Predicting materials properties without crystal structure: deep representation learning from stoichiometry. <i>Nat. Commun.</i> <b>11</b>, 6280 (2020).

53. Wang, A. Y.-T., Kauwe, S. K., Murdock, R. J. & Sparks, T. D. Compositionally restricted attention-based network for materials property predictions. <i>npj Comput. Mater.</i> <b>7</b>, 77 (2021).

54. Xie, T. & Grossman, J. C. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. <i>Phys. Rev. Lett.</i> <b>120</b>, 145301 (2018).

55. Chen, C., Ye, W., Zuo, Y., Zheng, C. & Ong, S. P. Graph networks as a universal machine learning framework for molecules and crystals. <i>Chem. Mater.</i> <b>31</b>, 3564–3572 (2019).

56. Choudhary, K. & DeCost, B. Atomistic line graph neural network for improved materials property predictions. <i>npj Comput. Mater.</i> <b>7</b>, 185 (2021).

57. Ward, L., Agrawal, A., Choudhary, A. & Wolverton, C. A general-purpose machine learning framework for predicting properties of inorganic materials. <i>npj Comput. Mater.</i> <b>2</b>, 16028 (2016).

58. Ward, L. et al. Matminer: an open source toolkit for materials data mining. <i>Comput. Mater. Sci.</i> <b>152</b>, 60–69 (2018).

59. Chen, T. & Guestrin, C. XGBoost: a scalable tree boosting system. In <i>Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i> 785–794 (ACM, 2016).

60. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. & Gulin, A. CatBoost: unbiased boosting with categorical features. In <i>Advances in Neural Information Processing Systems</i> Vol. 31, 6638–6648 (Curran Associates, 2018).

61. Pedregosa, F. et al. Scikit-learn: machine learning in Python. <i>J. Mach. Learn. Res.</i> <b>12</b>, 2825–2830 (2011).

62. Paszke, A. et al. PyTorch: an imperative style, high-performance deep learning library. In <i>Advances in Neural Information Processing Systems</i> Vol. 32, 8024–8035 (Curran Associates, 2019).

63. Grinsztajn, L., Oyallon, E. & Varoquaux, G. Why do tree-based models still outperform deep learning on typical tabular data? In <i>Advances in Neural Information Processing Systems</i> Vol. 35, 507–520 (Curran Associates, 2022).

64. Shwartz-Ziv, R. & Armon, A. Tabular data: deep learning is not all you need. <i>Inf. Fusion</i> <b>81</b>, 84–90 (2022).

65. Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. Optuna: a next-generation hyperparameter optimization framework. In <i>Proc. 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining</i> 2623–2631 (ACM, 2019).

66. Tran, K. et al. Methods for comparing uncertainty quantifications for material property predictions. <i>Mach. Learn. Sci. Technol.</i> <b>1</b>, 025006 (2020).

67. Hirschfeld, L., Swanson, K., Yang, K., Barzilay, R. & Coley, C. W. Uncertainty quantification using neural networks for molecular property prediction. <i>J. Chem. Inf. Model.</i> <b>60</b>, 3770–3780 (2020).

68. Palmer, G. et al. Calibration after bootstrap for accurate uncertainty quantification in regression models. <i>npj Comput. Mater.</i> <b>8</b>, 115 (2022).

69. Heid, E., McGill, C. J., Vermeire, F. H. & Green, W. H. Characterizing uncertainty in machine learning for chemistry. <i>J. Chem. Inf. Model.</i> <b>63</b>, 4012–4029 (2023).

70. Varivoda, D., Dong, R., Omee, S. S. & Hu, J. Materials property prediction with uncertainty quantification: a benchmark study. <i>Appl. Phys. Rev.</i> <b>10</b>, 021409 (2023).

71. Hu, Y., Musielewicz, J., Ulissi, Z. W. & Medford, A. J. Robust and scalable uncertainty estimation with conformal prediction for machine-learned interatomic potentials. <i>Mach. Learn. Sci. Technol.</i> <b>3</b>, 045028 (2022).

72. Virtanen, P. et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. <i>Nat. Methods</i> <b>17</b>, 261–272 (2020).

73. Harris, C. R. et al. Array programming with NumPy. <i>Nature</i> <b>585</b>, 357–362 (2020).

74. Bates, S., Hastie, T. & Tibshirani, R. Cross-validation: what does it estimate and how well does it do it? <i>J. Am. Stat. Assoc.</i> <b>119</b>, 1434–1445 (2024).

75. Ohno, S. et al. How certain are the reported ionic conductivities of thiophosphate-based solid electrolytes? An interlaboratory study. <i>ACS Energy Lett.</i> <b>5</b>, 910–915 (2020).

76. Vadhva, P. et al. Electrochemical impedance spectroscopy for all-solid-state batteries: theory, methods and future outlook. <i>ChemElectroChem</i> <b>8</b>, 1930–1947 (2021).

77. Ravinder et al. Artificial intelligence and machine learning in glass science and technology: 21 challenges for the 21st century. <i>Int. J. Appl. Glass Sci.</i> <b>12</b>, 277–292 (2021).

78. Ravinder, R. et al. Deep learning aided rational design of oxide glasses. <i>Mater. Horiz.</i> <b>7</b>, 1819–1827 (2020).
