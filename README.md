# Low-density EEG classification of PSEN1 E280A carriers

Code for the analyses of a manuscript in preparation for Frontiers in Neurology (Research Topic "AI-Enhanced Neuroimaging: Transforming Neurodegenerative Disease Management").

The study classifies **asymptomatic (ACr, n = 91)** and **symptomatic (SCr, n = 47)** PSEN1 E280A carriers against **683 healthy controls (HC)** from ten recording sites, using 544 resting-state EEG features computed from an eight-channel montage. All carriers were recorded in Medellín (Colombia); controls come from three Medellín sites and seven public or collaborating cohorts, so carrier status is partly confounded with recording site. The code addresses this with a reference-based harmonization and a set of sensitivity analyses.

> **Status.** This repository replaces an earlier version of the analysis (matched controls, headline AUC 0.892) that is no longer the basis of the manuscript. The earlier files are kept unchanged in [`legacy_2026-08/`](legacy_2026-08/) for provenance.

## Repository layout

| Path | Content |
|---|---|
| `pipeline/` | Final analysis code (nested cross-validation, feature importance, sensitivity analyses). |
| `results/` | Aggregated results only (mean ± SD over five cross-validation partitions). No participant-level data. |
| `legacy_2026-08/` | Previous pipeline and its scripts (1_make_dataframe, PSM, harmonization, reports). |

## Pipeline (`pipeline/`)

1. **Input.** A participant × feature table (`.feather`) with the 544 EEG features, `group`, `orig_group`, `SITE`, `age` and `sex`. Features are computed upstream with APPLEE ([github.com/GRUNECO/portables](https://github.com/GRUNECO/portables)); the harmonized table is produced by reference-based ComBat (site effects learned on controls only and applied to all participants; see `legacy_2026-08/optional_neuroharmonize.py`). The data are **not** included (see *Data availability*).
2. **Classification** (`3_train_ml_v2.py`). Nested cross-validation (10 outer × 5 inner folds). Inside each training fold: kNN imputation, linear residualization of age, z-scoring and, where specified, SMOTE. Classifiers: random forest, SVM, logistic regression, XGBoost and a soft-voting ensemble, each with ANOVA or RFE feature selection and a randomized hyperparameter search. Age is the only covariate; sex is not used because it is missing for 315 of 821 participants.
3. **Experiments** (`experiments_registry.py`, `run_experiments.py`). Every analysis is a registry entry; `python run_experiments.py --ids E05 E03` runs a subset. `CV_SEED` (environment variable, default 42) changes the cross-validation partition; the manuscript reports mean ± SD over seeds 42 and 1–4. `run_parallel.py` runs experiments in parallel. Each run writes `oof_predictions.csv` (out-of-fold predictions per participant) to its output folder; these files contain participant-level information and are not included here.
4. **Post hoc analyses.**
   - `oof_site_stratified_auc.py`: AUC of the out-of-fold predictions restricted to Medellín participants, per site and in the age range shared by carriers and controls.
   - `partial_confounder_test_v2.py`: partial confounder test ([mlconfound](https://github.com/pni-lab/mlconfound); Spisak, GigaScience 2022) on the out-of-fold predictions averaged over partitions. Sex is evaluated only in participants with recorded sex.
   - `effect_sizes_median_g.py`: Hedges' g for each feature, summarized by the median |g| with bootstrap interval and permutation null.

## Mapping between manuscript analyses and experiment IDs

| Manuscript analysis | SCr vs HC | ACr vs HC |
|---|---|---|
| Primary model (age residualized) | E05 | E03 |
| No age adjustment | E12 | E11 |
| Earlier variance filter | E05L | — |
| Spectral power only | E21 | E15 |
| Synchronization likelihood only | E22 | E16 |
| Coherence only | E23 | E17 |
| Permutation entropy only | E24 | E18 |
| Cross-frequency amplitude modulation only | E25 | E19 |
| Node-level features only | E26 | E20 |
| Portable-device participants excluded | E32 | E33 (combined carriers: E29 vs E31) |
| Age-comparable controls | E30 | E63 |
| Same-site controls (65 Medellín) | E34 | E35 |
| Without ComBat | E36 | E37 |
| Cross-fitted ComBat | E41 | E40 |
| Label permutation | — | E57–E59 (harmonized), E52–E56 (unharmonized) |
| One control site removed (Seoul, Dortmund, Cuba, Poland, small sites) | E75–E79 | E70–E74 |
| Medellín vs other controls, no carriers (without / with ComBat) | E50 / E51 | |
| Age 20–45 years (without / with ComBat) | | E60 / E61 |
| ACr vs HC_ACr without ComBat | | E62 |
| Within each Medellín site | E67–E69 | E64–E66 |

Aggregated values for every row are in `results/summary_5partitions.csv` (cross-validated and bootstrap AUC, mean and SD over the five partitions). The other files in `results/` contain the within-Medellín evaluation (`within_medellin_auc.csv`), the effect sizes (`effect_size_median_g.csv`) and the partial confounder test (`partial_confounder_test.csv`).

## Installation

```bash
pip install -r pipeline/requirements.txt
```

`mlconfound` depends on `pygam`, which uses a `scipy.sparse` attribute removed in recent SciPy versions; `partial_confounder_test_v2.py` includes a small compatibility shim.

## Data availability

The EEG recordings are not distributed here. The control cohorts are public (for example, the Cuban Human Brain Mapping Project at <https://chbmp-open.loris.ca/>). The Medellín recordings are not publicly available at the time of submission; they can be shared after acceptance of the article, or earlier under a data-sharing agreement, upon request to the corresponding author (John Fredy Ochoa Gómez, john.ochoa@udea.edu.co).

## Known limitations of this release

- The script that builds the cross-fitted ComBat table (used for the cross-fitting analysis) is not included yet.
- The upstream feature-extraction code is in the APPLEE repository, not here.

## License

MIT
