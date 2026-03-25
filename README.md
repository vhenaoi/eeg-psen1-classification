# EEG Resting-State Biomarkers for Pre-symptomatic PSEN1 Mutation Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Machine learning pipeline for classifying **PSEN1 familial Alzheimer's disease mutation carriers** vs. healthy controls using resting-state EEG. Designed for multi-site data with rigorous confound control.

**Primary result:** AUC = 0.892 ± 0.059 (PSM 1:1, Random Forest + RFE, nested 10×5-fold CV)

---

## Scientific Context

PSEN1 (presenilin-1) mutation carriers develop familial Alzheimer's disease with 100% penetrance, typically between ages 30–45. Identifying pre-symptomatic EEG biomarkers is clinically relevant for early monitoring.

**Core challenge:** All PSEN1 carriers come from a single site (Medellín, Colombia), while controls span 10 international sites. This confounds GROUP with SITE, which breaks standard harmonization approaches (see *Reference-Based Harmonization* below).

**Key methodological contributions:**
- Reference-based two-step ComBat: site effects estimated from controls only, applied to all subjects
- Propensity Score Matching (PSM) on age to control the 28-year group age gap
- Nested cross-validation with within-fold age residualization
- SAGE (Shapley Additive Global importancE) for statistically rigorous feature importance with 95% CI

---

## Pipeline

```
Raw EEG features (.feather)
        │
        ▼
1_make_dataframe.py        ← merge 5 metrics + demographics
        │
        ▼
optional_neuroharmonize.py ← reference-based ComBat (controls only → all)
        │
        ▼
2_apply_psm.py             ← Propensity Score Matching on age
        │
        ▼
3_train_ml_v2.py           ← nested CV + RF/SVM/LR + SAGE + SHAP + robustness
        │
        ▼
4_predict_new_data.py      ← inference on new subjects
```

---

## Installation

```bash
pip install -r requirements.txt
```

`shap` and `sage-importance` are optional (interpretability only):
```bash
pip install shap sage-importance
```

---

## Configuration

Every script has a `BASE_PATH` constant at the top. Set it to your local data root before running:

```python
# In each script, update this line:
BASE_PATH = r'C:\your\path\to\data'
```

The expected directory structure under `BASE_PATH`:

```
BASE_PATH/
├── Resultados/
│   ├── Data_complete_ce_roi.feather          ← output of step 1
│   ├── Data_complete_ce_roi_HARMONIZED.feather ← output of step 2
│   ├── PSM_datasets/                         ← output of step 3
│   │   ├── Data_matched_ce_roi_PSEN1_1to1.feather
│   │   ├── Data_matched_ce_roi_PSEN1_2to1.feather
│   │   └── Data_matched_ce_roi_PSEN1_5to1.feather
│   └── graphics/
│       └── ML_v2/                            ← output of step 4
└── datos_filtrados_concatenados.xlsx         ← demographics file
```

---

## Data Format

Input feather files must have one row per EEG epoch/segment per subject, with columns:

| Column | Description |
|---|---|
| `subject` | Subject identifier |
| `group` | `'PSEN1'` or `'Control'` |
| `SITE` | Acquisition site label |
| `age` | Age in years |
| `sex` | `'M'` / `'F'` (optional) |
| `[feature columns]` | EEG metrics (power, sl, coherence, entropy, crossfreq) |

Subjects are aggregated to **mean per subject** at training time.

---

## Usage

Run scripts in order:

```bash
# Step 1 — Build unified dataframe
python 1_make_dataframe.py

# Step 2 — Reference-based site harmonization (optional but recommended)
python optional_neuroharmonize.py

# Step 3 — Propensity Score Matching
python 2_apply_psm.py

# Step 4 — ML training (nested CV, 8 conditions, SAGE + SHAP)
python 3_train_ml_v2.py

# Step 4b — Run SAGE on existing models without re-training
python run_sage_only.py

# Step 5 — Predict on new subjects
python 4_predict_new_data.py
```

---

## Reference-Based Harmonization

Standard ComBat applied jointly to all subjects fails here: PSEN1 carriers are 100% from Medellín, so the Medellin site-effect estimate absorbs disease-related EEG differences (empirically: −0.101 AUC vs. reference-based approach).

**Two-step solution (`optional_neuroharmonize.py`):**
1. `harmonizationLearn` on **controls only** → estimates site effects from healthy subjects
2. `harmonizationApply` to **all subjects** → applies the correction without contamination

---

## Experimental Conditions

| Condition | Data | Age handling | N |
|---|---|---|---|
| `covariates_in_model` | Full sample | age as feature | 539 |
| `residualization` | Full sample | age residualized per fold | 539 |
| `psm_1to1_residualization` ⭐ | PSM 1:1 | age residualized per fold | 160 |
| `psm_1to1_covariates` | PSM 1:1 | age as feature | 160 |
| `psm_2to1_residualization` | PSM 2:1 | age residualized per fold | 147 |
| `psm_2to1_covariates` | PSM 2:1 | age as feature | 147 |
| `psm_5to1_residualization` | PSM 5:1 | age residualized per fold | 120 |
| `psm_5to1_covariates` | PSM 5:1 | age as feature | 120 |

⭐ Primary condition.

---

## Results

### Classification Performance (best combo per condition)

| Condition | AUC (CV) | Bootstrap AUC | Model |
|---|---|---|---|
| covariates_in_model | 0.974 ± 0.015 | 0.977 ± 0.008 | RF_rfe |
| residualization | 0.963 ± 0.015 | 0.926 ± 0.024 | RF_rfe |
| **psm_1to1_residualization** ⭐ | **0.892 ± 0.059** | **0.882 ± 0.055** | RF_rfe |
| psm_1to1_covariates | 0.905 ± 0.057 | 0.881 ± 0.039 | RF_rfe |
| psm_2to1_residualization | 0.847 ± 0.152 | 0.808 ± 0.056 | LR_rfe |
| psm_2to1_covariates | 0.880 ± 0.068 | 0.870 ± 0.046 | RF_rfe |

### Top SAGE Features — Primary Condition (psm_1to1_residualization)

| Feature | SAGE Value | 95% CI | Interpretation |
|---|---|---|---|
| C3_Beta3 | 0.088 ± 0.002 | [0.085, 0.092] | Central Beta-3 power (~25–30 Hz) |
| O2_Beta1/MDelta | 0.059 ± 0.002 | [0.055, 0.062] | Occipital cross-frequency ratio |
| O1_Theta_coh | 0.053 ± 0.001 | [0.051, 0.055] | Occipital theta coherence |

All SAGE values have CI > 0, confirming statistical informativeness. SAGE/SHAP agreement: Spearman ρ > 0.90 in primary conditions.

---

## Citation

> [Manuscript in preparation] — PSEN1 EEG ML study, 2026.

---

## License

MIT
