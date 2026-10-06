"""
Generate scientific Word report for PSEN1 EEG ML study.
"""

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

# ── helpers ────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def set_cell_borders(cell, top=True, bottom=True, left=False, right=False,
                     color='000000', sz='4'):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for side, flag in [('top', top), ('bottom', bottom),
                       ('left', left), ('right', right)]:
        if flag:
            el = OxmlElement(f'w:{side}')
            el.set(qn('w:val'), 'single')
            el.set(qn('w:sz'), sz)
            el.set(qn('w:space'), '0')
            el.set(qn('w:color'), color)
            tcBorders.append(el)
    tcPr.append(tcBorders)

def add_paragraph(doc, text='', style='Normal', bold=False, italic=False,
                  size=None, align=None, space_before=None, space_after=None,
                  color=None):
    p = doc.add_paragraph(style=style)
    if align:
        p.alignment = align
    if space_before is not None:
        p.paragraph_format.space_before = Pt(space_before)
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    if text:
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        if size:
            run.font.size = Pt(size)
        if color:
            run.font.color.rgb = RGBColor(*color)
    return p

def add_heading(doc, text, level=1, space_before=12, space_after=4):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    return p

def make_table(doc, headers, rows, col_widths=None, header_bg='1F4E79',
               alt_bg='E8F0FE', font_size=9):
    n_cols = len(headers)
    tbl = doc.add_table(rows=1 + len(rows), cols=n_cols)
    tbl.style = 'Table Grid'
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    # header row
    hdr = tbl.rows[0]
    for j, h in enumerate(headers):
        cell = hdr.cells[j]
        cell.text = h
        set_cell_bg(cell, header_bg)
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(font_size)
            run.font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # data rows
    for i, row_data in enumerate(rows):
        row = tbl.rows[i + 1]
        bg = alt_bg if i % 2 == 0 else 'FFFFFF'
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            cell.text = str(val)
            set_cell_bg(cell, bg)
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(font_size)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    if col_widths:
        for j, w in enumerate(col_widths):
            for row in tbl.rows:
                row.cells[j].width = Cm(w)
    return tbl

# ── document ────────────────────────────────────────────────────────────────

doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(3.0)
    section.right_margin  = Cm(2.5)

# ── TITLE ────────────────────────────────────────────────────────────────────
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_after = Pt(6)
r = p_title.add_run(
    'EEG Resting-State Biomarkers for Early Detection of PSEN1 Mutation Carriers: '
    'A Multi-Site Machine Learning Study with Propensity Score Matching'
)
r.bold = True
r.font.size = Pt(14)
r.font.color.rgb = RGBColor(31, 78, 121)

add_paragraph(doc, 'Universidad de Posgrado — Tesis Doctoral', align=WD_ALIGN_PARAGRAPH.CENTER,
              italic=True, size=10, space_before=2, space_after=2)
add_paragraph(doc, 'Date: March 2026 | Status: Results report for thesis', italic=True, size=9,
              align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=12)

doc.add_paragraph('─' * 100)

# ── ABSTRACT ─────────────────────────────────────────────────────────────────
add_heading(doc, '1. Abstract', level=1)
abstract_text = (
    'Background: PSEN1 mutation carriers represent the most aggressive form of familial '
    'Alzheimer\'s disease (FAD), with 100% penetrance and early symptom onset (typically 30-45 years). '
    'Identifying reliable EEG biomarkers in the pre-symptomatic phase could enable early monitoring. '
    'However, confounds including age differences, multi-site EEG heterogeneity, and demographic '
    'imbalances challenge the validity of classification studies.\n\n'
    'Methods: Resting-state EEG was acquired from 91 PSEN1 mutation carriers (Medellin, Colombia) '
    'and 442 healthy controls from 10 international sites. Five quantitative EEG metrics were computed: '
    'spectral power, synchronization likelihood, coherence, spectral entropy, and cross-frequency coupling. '
    'Site harmonization was performed with neuroHarmonize (ComBat). '
    'Age confounding was addressed via two complementary strategies: '
    '(i) propensity score matching (PSM) on age, and '
    '(ii) within-fold age residualization of EEG features. '
    'Classification was performed using nested cross-validation (10-fold outer x 5-fold inner) '
    'with Random Forest, SVM, and Logistic Regression. Feature interpretability was assessed '
    'using SAGE (Shapley Additive Global importancE) as the primary analysis — providing global '
    'Shapley values with 95% bootstrap confidence intervals via PermutationEstimator — '
    'supplemented by SHAP beeswarm plots for direction of per-sample effects. '
    'Robustness was characterized through bootstrap stability, feature noise injection, '
    'and hyperparameter sensitivity analysis.\n\n'
    'Results: In the primary PSM 1:1 condition (n=160, balanced), the best model (Random Forest + RFE) '
    'achieved AUC=0.892 (SD=0.059), bootstrap AUC=0.882 (SD=0.055), Brier=0.135, ECE=0.049. '
    'Near-identical performance with age included as feature (AUC=0.905) vs. age residualized '
    '(AUC=0.892) confirms that EEG features, not residual age differences, drive classification. '
    'The most discriminative features identified by SAGE in the primary PSM condition were '
    'C3_Beta3 (central beta power), O2_Beta1/MDelta (occipital cross-frequency ratio), '
    'and O1_Theta_coh (occipital theta coherence).\n\n'
    'Conclusions: Age-matched, site-harmonized EEG features can discriminate PSEN1 carriers from '
    'controls with strong accuracy (AUC=0.892). Central-occipital beta and gamma oscillations emerge '
    'as candidate pre-symptomatic EEG biomarkers. The use of reference-based harmonization '
    '(fitting ComBat only on controls) was critical: standard ComBat applied to all subjects reduced '
    'AUC by ~0.10 by absorbing PSEN1 biological signal into the Medellin site effect estimate.'
)
p_abs = doc.add_paragraph(abstract_text)
p_abs.paragraph_format.space_after = Pt(6)
for run in p_abs.runs:
    run.font.size = Pt(10)

add_paragraph(doc, 'Keywords: PSEN1, familial Alzheimer\'s disease, EEG biomarkers, machine learning, '
              'propensity score matching, neuroHarmonize, SAGE, SHAP, nested cross-validation, '
              'Beta/Gamma oscillations, pre-symptomatic detection.',
              italic=True, size=9, space_before=2, space_after=10)

# ── 2. PARTICIPANTS ───────────────────────────────────────────────────────────
add_heading(doc, '2. Participants and Data', level=1)
add_heading(doc, '2.1 Full Sample Demographics', level=2)

demo_text = (
    'The full dataset comprised 539 subjects: 91 PSEN1 mutation carriers '
    '(all from Medellin, Colombia; three acquisition protocols) and 448 healthy controls '
    'from 10 international sites. PSEN1 carriers were significantly younger than controls '
    '(31.9 +/- 6.5 vs. 60.3 +/- 16.6 years; t=-16.09, p<0.001), reflecting the early-onset '
    'nature of the mutation. Sex data were available for 50/91 PSEN1 (20M/30F) and 71/442 '
    'controls (30M/41F); Seoul site (n=210, 47% of controls) contributed 0% sex data, '
    'precluding sex-balanced analyses. Education data were available for 91/91 PSEN1 '
    '(11.9 +/- 3.1 years) and 31/442 controls (11.8 +/- 6.2 years).\n\n'
    'Given that PSEN1 carriers are exclusively from Medellin while controls span 10 sites, '
    'site harmonization via neuroHarmonize (ComBat framework) was applied prior to all analyses '
    'to remove site-related variance from EEG features while preserving biological variance '
    '(age, sex as covariates). Education and sex were excluded from the classifier feature set '
    'due to severely imbalanced data availability (Seoul controls: 0% for both), which would '
    'create artifactual classification signal unrelated to EEG biology.'
)
doc.add_paragraph(demo_text).runs[0].font.size = Pt(10)

# Table 1: Demographics
add_paragraph(doc, 'Table 1. Demographic characteristics of the full sample.',
              bold=True, size=10, space_before=6, space_after=3)

make_table(doc,
    headers=['Variable', 'PSEN1 (n=91)', 'Control (n=448)', 'Statistic', 'p-value'],
    rows=[
        ['Age (years)', '31.9 +/- 6.5 [20-45]', '60.3 +/- 16.6 [20-88]', 't = -16.09', '< 0.001'],
        ['Sex M/F (% with data)', '20/30 (55% coverage)', '30/41 (16% coverage)', 'Chi-sq', 'N/A (incomplete)'],
        ['Education (years)', '11.9 +/- 3.1 (100%)', '11.8 +/- 6.2 (7%)', 'N/A', 'N/A (incomplete)'],
        ['Sites (n)', 'Medellin x3 (100%)', '10 sites (Seoul=47%)', '—', '—'],
        ['EEG rows', '5,298 total', '539 unique subjects', '—', '—'],
    ],
    col_widths=[4.5, 4.0, 4.0, 3.0, 2.0]
)
add_paragraph(doc, 'Note: Mean +/- SD [range]. Age t-test: Welch\'s two-sample t-test.',
              italic=True, size=9, space_before=3, space_after=8)

# ── 2.2 PSM ──────────────────────────────────────────────────────────────────
add_heading(doc, '2.2 Propensity Score Matching (PSM)', level=2)
psm_text = (
    'To control for the pronounced age difference between groups, propensity score matching (PSM) '
    'was performed using nearest-neighbor matching without replacement (caliper = 0.2 SD of the '
    'log-odds). Propensity scores were estimated via logistic regression with age as the sole '
    'matching covariate, yielding the following matched samples:\n\n'
    '  - PSM 1:1 (n=160): 80 PSEN1 vs. 80 Controls. Age PSEN1: 32.2+/-6.7; Controls: 32.4+/-6.7 (t=-0.17, p=0.869)\n'
    '  - PSM 2:1 (n=147): 49 PSEN1 vs. 98 Controls. Age PSEN1: 33.9+/-7.1; Controls: 34.2+/-7.3 (t=-0.23, p=0.821)\n'
    '  - PSM 5:1 (n=120): 20 PSEN1 vs. 100 Controls (limited statistical power)\n\n'
    'After PSM, age was no longer significantly different between groups (p>0.8 for 1:1 and 2:1 '
    'conditions), confirming successful matching. The PSM 1:1 condition is considered the primary '
    'analysis. PSM was applied on harmonized data to ensure consistency with downstream EEG feature analyses.\n\n'
    'PSM 1:1 matched sample site distribution: PSEN1: Medellin_ld (n=32), Medellin_hd (n=26), '
    'Medellin_duque (n=22); Controls: Dortmund (n=47), Cuba (n=17), Medellin_duque (n=6), '
    'Medellin_ld (n=4), Medellin_hd (n=4), Seoul (n=2).'
)
doc.add_paragraph(psm_text).runs[0].font.size = Pt(10)

# Table 2: PSM balance
add_paragraph(doc, 'Table 2. PSM balance: age statistics before and after matching.',
              bold=True, size=10, space_before=6, space_after=3)
make_table(doc,
    headers=['Scenario', 'N (PSEN1/Control)', 'Age PSEN1', 'Age Control', 't', 'p'],
    rows=[
        ['Before PSM (full)', '91 / 448', '31.9 +/- 6.5', '60.3 +/- 16.6', '-16.09', '< 0.001'],
        ['PSM 1:1 (primary)', '80 / 80',  '32.2 +/- 6.7', '32.4 +/- 6.7',  '-0.17', '0.869'],
        ['PSM 2:1',           '49 / 98',  '33.9 +/- 7.1', '34.2 +/- 7.3',  '-0.23', '0.821'],
        ['PSM 5:1',           '20 / 100', '~32',           '~32',            'N/A',   'N/A'],
    ],
    col_widths=[4.0, 4.0, 3.5, 3.5, 1.5, 1.5]
)
add_paragraph(doc, 'Note: Values are mean +/- SD. t: Welch t-test. PSM caliper = 0.2 SD log-odds.',
              italic=True, size=9, space_before=3, space_after=8)

# ── 3. METHODS ────────────────────────────────────────────────────────────────
add_heading(doc, '3. Methods', level=1)

add_heading(doc, '3.1 EEG Features', level=2)
eeg_text = (
    'Resting-state EEG was processed to extract five quantitative metrics per electrode pair/channel:\n\n'
    '(1) Spectral Power (power): Absolute spectral power per frequency band '
    '(Delta 1-4 Hz, Theta 4-8 Hz, Alpha-1 8-10 Hz, Alpha-2 10-13 Hz, Beta1/2/3, Gamma) '
    'per ROI channel.\n\n'
    '(2) Synchronization Likelihood (sl): A non-linear measure of generalized synchronization '
    'between channel pairs, sensitive to coordinated neural activity.\n\n'
    '(3) Coherence (coh): Spectral coherence between electrode pairs per frequency band, '
    'quantifying linear synchrony.\n\n'
    '(4) Spectral Entropy (entropy): Shannon entropy of the power spectrum, reflecting '
    'the complexity/regularity of the EEG signal.\n\n'
    '(5) Cross-Frequency Coupling (crossfreq): Amplitude ratios between frequency bands '
    '(e.g., Alpha/Delta, Alpha/Theta), capturing nested oscillation dynamics.\n\n'
    'After aggregation across recording epochs (mean per subject), 544 EEG features were '
    'available prior to pre-filtering.'
)
doc.add_paragraph(eeg_text).runs[0].font.size = Pt(10)

add_heading(doc, '3.2 Site Harmonization — Reference-Based Two-Step ComBat', level=2)
harm_text = (
    'Multi-site EEG heterogeneity was addressed using neuroHarmonize (Python implementation '
    'of ComBat; Johnson et al., 2007). A critical methodological challenge arose from the '
    'confounded study design: PSEN1 carriers are exclusively from Medellin, Colombia (3 '
    'acquisition protocols), while controls span 10 international sites. In standard ComBat '
    'applied jointly to all subjects, the Medellin site effect is estimated using both '
    'PSEN1 and control data. Since PSEN1 subjects constitute 22-52% of Medellin recordings, '
    'disease-related EEG differences are partially absorbed into the Medellin site effect '
    'estimate, producing a conservative (downward) bias in AUC.\n\n'
    'To address this, a REFERENCE-BASED TWO-STEP HARMONIZATION was implemented:\n'
    '  Step 1 (harmonizationLearn): ComBat site-effect parameters (additive gamma, '
    'multiplicative delta) are estimated exclusively from HEALTHY CONTROLS (n=448 subjects, '
    '5,207 recording rows), including 32 Medellin controls (Medellin_duque: 9, Medellin_ld: '
    '13, Medell\'n_hd: 10 subjects; 280 rows total). Age and sex were included as biological '
    'covariates to protect within-group variation.\n'
    '  Step 2 (harmonizationApply): The learned site correction is applied to ALL subjects '
    '(controls + ADMCI + PSEN1, 6,283 rows), using a consistent SITE encoding derived from '
    'the full dataset. This ensures the same technical correction is applied to PSEN1 subjects '
    'without those subjects contributing to the estimation of site effects.\n\n'
    'Consequence: The gamma_Medellin and delta_Medellin parameters now reflect purely technical '
    'variance (recording equipment, protocol differences) rather than a mix of technical and '
    'biological variance. Empirical validation: the reference-based approach yielded AUC=0.892 '
    'in the primary PSM 1:1 condition, compared to AUC=0.791 with standard ComBat — a 0.101 '
    'AUC recovery attributable to preserved PSEN1 biological signal.\n\n'
    'Features were log-transformed prior to harmonization (log(0.001+x)) and back-transformed '
    'after (exp(x)-0.001). All covariate values were restored from the original data after '
    'harmonization. Education and sex were excluded from the ML feature set (retained as '
    'harmonization covariates) because Seoul controls (n=210, 47% of controls) lack both '
    'variables, which would create artifactual classification signal.'
)
doc.add_paragraph(harm_text).runs[0].font.size = Pt(10)

add_heading(doc, '3.3 Machine Learning Pipeline', level=2)
ml_text = (
    'Classification was implemented using a nested cross-validation (nested CV) framework '
    'to obtain unbiased performance estimates while simultaneously optimizing hyperparameters. '
    'All preprocessing steps were performed strictly inside the CV folds to prevent data leakage.\n\n'
    'NESTED CV STRUCTURE:\n'
    '  - Outer loop: 10-fold stratified K-fold (performance estimation)\n'
    '  - Inner loop: 5-fold stratified K-fold (hyperparameter optimization via RandomizedSearchCV, 50 iterations)\n'
    '  - Random state: 42 (reproducibility)\n\n'
    'PREPROCESSING PIPELINE (inside each fold):\n'
    '  (1) KNN Imputation (k=5) on training and test sets (imputer fitted on train only)\n'
    '  (2) Optional age residualization: linear regression of age on each EEG feature, '
    'trained on the training fold and applied to both train and test, removing linear age effects\n'
    '  (3) StandardScaler (zero mean, unit variance; fitted on train)\n'
    '  (4) SMOTE + RandomUnderSampler (when class ratio < 0.8): minority class oversampled '
    'to 80% of majority, then majority undersampled; applied only to training data\n\n'
    'FEATURE SELECTION (two parallel methods, majority vote):\n'
    '  - SelectKBest: ANOVA F-test, top 20 features\n'
    '  - RFE (Recursive Feature Elimination): light Random Forest estimator, 20 features\n'
    '  Stable features: selected in >= 50% of outer folds\n\n'
    'PRE-FILTERING (applied before nested CV, no labels used):\n'
    '  (1) VarianceThreshold (threshold=0.01): removes near-zero variance features\n'
    '  (2) Correlation filter (r > 0.85): removes redundant features\n'
    '  Result: 44 features (unmatched) / 31-34 features (PSM conditions)\n\n'
    'CLASSIFIERS AND HYPERPARAMETER GRIDS:\n'
    '  - Random Forest (RF): n_estimators in {100,200,300,500}, max_features, max_depth, '
    'min_samples_split, criterion, class_weight\n'
    '  - Support Vector Machine (SVM, RBF/linear): C in logspace(-3,3,20), gamma, class_weight\n'
    '  - Logistic Regression (LR, L1/L2): C in logspace(-4,4,20), solver, class_weight, max_iter=1000\n'
    '  Total: 6 classifier-feature selection combinations (RF_kbest, RF_rfe, SVM_kbest, '
    'SVM_rfe, LR_kbest, LR_rfe)\n\n'
    'EVALUATION METRICS (outer CV):\n'
    '  Primary: AUC-ROC (area under receiver operating characteristic curve)\n'
    '  Secondary: F1-score, Recall, Accuracy, Brier Score\n'
    '  The best combination was selected by mean outer-fold AUC.'
)
doc.add_paragraph(ml_text).runs[0].font.size = Pt(10)

add_heading(doc, '3.4 Experimental Conditions', level=2)
cond_text = (
    'Eight experimental conditions were evaluated to dissociate the contributions of age, '
    'site harmonization, and PSM on classification performance:\n\n'
    '(1) covariates_in_model: Full sample (n=539), harmonized EEG features + age as explicit '
    'classifier input. Establishes the upper bound including age information.\n\n'
    '(2) residualization: Full sample (n=539), age linearly residualized from EEG features '
    'within each fold. Tests EEG signal after removing linear age effects.\n\n'
    '(3) psm_1to1_residualization (PRIMARY): PSM 1:1 matched sample (n=160, 80 per group), '
    'age residualized from EEG. Strongest confound control: age-matched AND age effects removed.\n\n'
    '(4) psm_1to1_covariates (VALIDATION): PSM 1:1 matched sample (n=160), age as feature. '
    'Critical validation: if AUC ~ psm_1to1_residualization, confirms EEG drives classification.\n\n'
    '(5) psm_2to1_residualization: PSM 2:1 (n=147, 49+98), age residualized. '
    'Sensitivity analysis with more controls.\n\n'
    '(6) psm_2to1_covariates: PSM 2:1 (n=147), age as feature.\n\n'
    '(7-8) psm_5to1_residualization / psm_5to1_covariates: PSM 5:1 (n=120, 20 PSEN1). '
    'Exploratory; limited power due to small PSEN1 class.'
)
doc.add_paragraph(cond_text).runs[0].font.size = Pt(10)

add_heading(doc, '3.5 Robustness Analysis', level=2)
rob_text = (
    'Four complementary robustness analyses were conducted to assess model reliability '
    'beyond standard nested CV:\n\n'
    '(A) BOOTSTRAP STABILITY (N=20 resamples):\n'
    'Bootstrap with replacement was applied to the full dataset, training/testing the final '
    'best model on out-of-bag (OOB) samples in each resample. Reports mean +/- SD of OOB AUC. '
    'A large discrepancy between nested CV AUC and bootstrap AUC indicates overfitting '
    'or instability due to small sample size.\n\n'
    '(B) CALIBRATION (OOF predictions):\n'
    'Model calibration was assessed using out-of-fold (OOF) predicted probabilities '
    'accumulated across all 10 outer folds:\n'
    '  - Brier Score: mean squared error of probability predictions (lower is better; 0=perfect, 0.25=chance)\n'
    '  - Expected Calibration Error (ECE): weighted mean absolute difference between '
    'predicted confidence and observed accuracy across 10 equal-width probability bins '
    '(lower is better; 0=perfectly calibrated)\n\n'
    '(C) FEATURE NOISE ROBUSTNESS:\n'
    'Gaussian noise (sigma = 0%, 5%, 10%, 20%, 50% of each feature standard deviation) was '
    'injected into the TEST set only, simulating cross-site EEG variability and sensor noise. '
    'DELTA_AUC_max reports the maximum performance drop across noise levels. '
    'This directly quantifies model resilience to EEG acquisition variability.\n\n'
    '(D) HYPERPARAMETER SENSITIVITY:\n'
    'The key hyperparameter of the best classifier (n_estimators for RF; C for SVM/LR) '
    'was varied by factors [x0.25, x0.5, x1.0, x2.0, x4.0] relative to the '
    'nested CV-optimized value. DELTA_AUC reports the range of AUC across these '
    'perturbations. Low DELTA_AUC confirms that classification performance is not '
    'critically dependent on precise hyperparameter tuning.'
)
doc.add_paragraph(rob_text).runs[0].font.size = Pt(10)

add_heading(doc, '3.6 SAGE Feature Importance (Primary Analysis)', level=2)
sage_text = (
    'SAGE (Shapley Additive Global importancE; Covert, Lundberg & Lee, 2020) was used as '
    'the primary interpretability method. Unlike SHAP, which computes per-sample attributions '
    'that may lack formal informativeness guarantees for deep tree ensembles (Gunther et al., 2025), '
    'SAGE computes global Shapley values defined directly on the model\'s loss function, '
    'providing a statistically coherent measure of each feature\'s contribution to predictive '
    'performance across the entire dataset.\n\n'
    'Implementation (sage-importance package, sklearn-compatible):\n'
    '  - Imputer: MarginalImputer (marginal feature distribution; model-agnostic)\n'
    '  - Estimator: PermutationEstimator with cross-entropy loss\n'
    '  - Convergence: detect_convergence=True (adaptive stopping when estimates stabilize)\n'
    '  - Background sample: n=512 subjects (or all if fewer)\n'
    '  - Random state: 42\n\n'
    'Outputs per condition:\n'
    '  - sage_importance.png: horizontal bar chart with 95% confidence intervals\n'
    '  - sage_importance.xlsx: ranked table with sage_value, sage_std, ci_low, ci_high\n\n'
    'SAGE values represent the reduction in cross-entropy loss attributable to each feature, '
    'averaged over all possible feature coalitions. Positive values indicate features that '
    'reduce prediction error (informative); values indistinguishable from zero (CI crossing 0) '
    'indicate non-informative features. The 95% CI (1.96 x sage_std) reflects the bootstrap '
    'uncertainty in the Shapley estimate.\n\n'
    'Feature naming convention:\n'
    '  - [Channel]_[Band]_coh: inter-channel coherence (e.g., FP1_Alpha-1_coh)\n'
    '  - [Channel]_[Band]: absolute spectral power\n'
    '  - [Channel]_[Band1]/M[Band2]: cross-frequency amplitude ratio\n'
    '  Alpha-1=8-10 Hz; Alpha-2=10-13 Hz; Theta=4-8 Hz; Delta=1-4 Hz; Beta3~25-30 Hz'
)
doc.add_paragraph(sage_text).runs[0].font.size = Pt(10)

add_heading(doc, '3.7 SHAP Analysis (Supplementary)', level=2)
shap_text = (
    'SHapley Additive exPlanations (SHAP; Lundberg & Lee, 2017) were computed as a '
    'supplementary analysis to visualize the DIRECTION of individual feature contributions '
    '(positive vs. negative effect on PSEN1 probability):\n\n'
    '  - Random Forest: TreeExplainer (exact Shapley values on tree structure)\n'
    '  - Logistic Regression: LinearExplainer (exact for linear models)\n'
    '  - SVM: omitted (KernelExplainer is prohibitively slow for high-dimensional data)\n\n'
    'Output: beeswarm plot showing the distribution of SHAP values per subject per feature, '
    'color-coded by feature value (high=red, low=blue). This complements SAGE by showing '
    'how each feature shifts predictions for individual subjects, providing insight into '
    'the heterogeneity of EEG biomarker expression across PSEN1 carriers.\n\n'
    'Note: While SAGE provides the primary global ranking (with formal statistical guarantees), '
    'SHAP beeswarm reveals the directionality that SAGE values alone do not capture '
    '(e.g., whether high C3_Beta3 power is associated with PSEN1 or Control prediction).'
)
doc.add_paragraph(shap_text).runs[0].font.size = Pt(10)

# ── 4. RESULTS ────────────────────────────────────────────────────────────────
add_heading(doc, '4. Results', level=1)

add_heading(doc, '4.1 Main Classification Results', level=2)
results_text = (
    'Table 3 presents the full classification results across all experimental conditions '
    'for the best classifier-feature selection combination. Results are reported as '
    'mean +/- SD across the 10 outer CV folds.'
)
doc.add_paragraph(results_text).runs[0].font.size = Pt(10)

add_paragraph(doc, 'Table 3. Classification performance across experimental conditions (best combo per condition).',
              bold=True, size=10, space_before=6, space_after=3)

make_table(doc,
    headers=['Condition', 'N', 'Model', 'AUC', 'F1', 'Recall', 'Brier', 'Bootstrap AUC'],
    rows=[
        ['covariates_in_model',       '539', 'RF_rfe', '0.974 +/- 0.015', '0.830 +/- 0.054', '0.891 +/- 0.120', '0.051', '0.977 +/- 0.008'],
        ['residualization',           '539', 'RF_rfe', '0.963 +/- 0.015', '0.761 +/- 0.065', '0.847 +/- 0.101', '0.065', '0.926 +/- 0.024'],
        ['psm_1to1_residualization *','160', 'RF_rfe', '0.892 +/- 0.059', '0.809 +/- 0.084', '0.800 +/- 0.115', '0.135', '0.882 +/- 0.055'],
        ['psm_1to1_covariates +',     '160', 'RF_rfe', '0.905 +/- 0.057', '0.831 +/- 0.083', '0.825 +/- 0.127', '0.131', '0.881 +/- 0.039'],
        ['psm_2to1_residualization',  '147', 'LR_rfe', '0.847 +/- 0.152', '0.669 +/- 0.244', '0.640 +/- 0.250', '0.148', '0.808 +/- 0.056'],
        ['psm_2to1_covariates',       '147', 'RF_rfe', '0.880 +/- 0.068', '0.713 +/- 0.102', '0.770 +/- 0.179', '0.146', '0.870 +/- 0.046'],
        ['psm_5to1_residualization',  '120', 'LR_rfe', '0.845 +/- 0.111', '0.519 +/- 0.293', '0.550 +/- 0.269', '0.137', '0.727 +/- 0.144'],
        ['psm_5to1_covariates',       '120', 'SVM_rfe','0.790 +/- 0.158', '0.594 +/- 0.136', '0.600 +/- 0.200', '0.119', '0.794 +/- 0.195'],
    ],
    col_widths=[5.5, 1.2, 2.5, 3.5, 3.5, 3.5, 1.8, 3.8],
    font_size=8
)
add_paragraph(doc,
    '* Primary condition. + Critical validation condition. '
    'AUC: Area Under ROC Curve. F1: harmonic mean of precision and recall. '
    'Brier: Brier Score (0=perfect, 0.25=chance). Bootstrap: OOB AUC over 20 resamples.',
    italic=True, size=9, space_before=3, space_after=6)

add_heading(doc, '4.2 Critical Comparison: PSM + Residualization vs. PSM + Age Feature', level=2)
comp_text = (
    'The near-identical performance of psm_1to1_covariates (AUC=0.905) and '
    'psm_1to1_residualization (AUC=0.892) provides a critical internal validation: '
    'within the age-matched PSM sample, age itself contributes negligibly to '
    'classification (delta AUC = 0.013). This confirms that the observed '
    'discriminative signal originates from EEG features rather than residual '
    'age differences within the matched cohort.\n\n'
    'Comparing residualization (AUC=0.963, unmatched) with '
    'psm_1to1_residualization (AUC=0.892) reveals a 0.071 AUC gap. With '
    'reference-based harmonization, this gap is substantially smaller than in '
    'prior literature, and is attributable primarily to: (a) statistical power '
    '(N=539 vs. N=160), and (b) residual population-level differences. '
    'The high AUC of residualization (0.963) after reference-based harmonization '
    'suggests that, with proper site correction, EEG differences between PSEN1 '
    'and controls are pervasive even in the unmatched sample.\n\n'
    'HARMONIZATION IMPACT: Applying standard ComBat to all subjects yielded '
    'psm_1to1_residualization AUC=0.791; reference-based ComBat yielded 0.892. '
    'The 0.101 AUC difference directly quantifies the biological signal that was '
    'previously absorbed into the Medellin site effect estimate — a systematic '
    'methodological bias affecting all prior analyses with this design.\n\n'
    'The PSM 1:1 condition is considered the most scientifically rigorous, '
    'as it eliminates both age confounding (matched) and minimizes site bias '
    '(by design, controls must have age-compatible matches, which preferentially '
    'selects younger controls from Dortmund and Cuba rather than older Seoul controls).'
)
doc.add_paragraph(comp_text).runs[0].font.size = Pt(10)

add_heading(doc, '4.3 Feature Importance (SAGE Analysis — Primary)', level=2)
sage_interp = (
    'Table 4 presents SAGE-ranked feature importances (global Shapley values on cross-entropy '
    'loss, with 95% CI) for all experimental conditions. SAGE values represent the mean '
    'reduction in cross-entropy attributable to each feature; features with CI crossing 0 '
    'are not statistically distinguishable from noise.\n\n'
    'With reference-based harmonization, SAGE identifies CENTRAL AND TEMPORAL BETA/GAMMA '
    'OSCILLATIONS as the primary discriminative features. C3_Beta3 (central Beta-3 power, '
    '~25-30 Hz; SAGE=0.088-0.095 in PSM 1:1 conditions) and O2_Beta1/MDelta (occipital '
    'cross-frequency ratio Beta1/Delta; SAGE=0.058-0.063) are the most consistent features '
    'across PSM conditions, with all 95% CIs well above zero confirming statistical significance. '
    'O1_Theta_coh (occipital theta coherence; SAGE=0.053-0.064) is a robust secondary feature.\n\n'
    'In the FULL SAMPLE conditions: C3_Gamma (central Gamma, ~30-45 Hz; SAGE=0.058) leads '
    'after residualization, followed by FP1_Alpha-1_coh (frontal alpha-1 coherence; SAGE=0.032) '
    'and T6_Gamma (right temporal Gamma; SAGE=0.030). The co-occurrence of central AND temporal '
    'gamma features in the unmatched residualization condition suggests a widespread gamma '
    'synchronization disruption pattern that PSM may partially attenuate by reducing '
    'population-level heterogeneity.\n\n'
    'SAGE provides a key methodological advantage over SHAP for this analysis: all reported '
    'features have CIs that exclude zero, confirming that they contribute to predictive performance '
    'beyond chance — a guarantee not provided by mean absolute SHAP values on tree ensembles '
    '(Gunther et al., 2025). Convergent validity: SAGE and SHAP agree on Top-3 features for '
    'all primary PSM conditions (Spearman rho > 0.90), validating both approaches.\n\n'
    'These findings are neurobiologically plausible: PSEN1 mutation disrupts gamma oscillations '
    'through GABAergic interneuron dysfunction (Iaccarino et al., 2016); beta desynchronization '
    'in central-temporal regions reflects early cortical network disruption preceding cognitive '
    'symptoms; and cross-frequency coupling alterations (Beta/Delta) are consistent with '
    'disrupted cortical hierarchy in familial AD models. With standard ComBat, these features '
    'were masked by site-effect absorption; their emergence with reference-based harmonization '
    'validates the methodological correction.'
)
doc.add_paragraph(sage_interp).runs[0].font.size = Pt(10)

add_paragraph(doc, 'Table 4. SAGE feature importance — top-3 features per condition (global Shapley values, cross-entropy loss).',
              bold=True, size=10, space_before=6, space_after=3)
make_table(doc,
    headers=['Condition', 'Top Feature (SAGE)', '2nd Feature (SAGE)', '3rd Feature (SAGE)'],
    rows=[
        ['covariates_in_model',
         'age (0.172±0.004)',          'C3_Gamma (0.030±0.001)',         'O2_Beta1/MDelta (0.022±0.001)'],
        ['residualization',
         'C3_Gamma (0.058±0.001)',     'FP1_Alpha-1_coh (0.032±0.001)', 'T6_Gamma (0.030±0.001)'],
        ['psm_1to1_residualization *',
         'C3_Beta3 (0.088±0.002)',     'O2_Beta1/MDelta (0.059±0.002)', 'O1_Theta_coh (0.053±0.001)'],
        ['psm_1to1_covariates +',
         'C3_Beta3 (0.095±0.002)',     'O1_Theta_coh (0.064±0.001)',    'O2_Beta1/MDelta (0.058±0.002)'],
        ['psm_2to1_residualization',
         'FP2_Beta3 (0.050±0.001)',    'O2_Beta1/MDelta (0.049±0.001)', 'C4_Beta3 (0.040±0.001)'],
        ['psm_2to1_covariates',
         'C3_Beta3 (0.092±0.002)',     'O2_Beta1/MDelta (0.063±0.002)', 'C4_Beta3 (0.047±0.001)'],
        ['psm_5to1_residualization',
         'C4_Beta3 (0.016±0.000)',     'T5_Beta3 (0.008±0.000)',        'C3_Beta3/MBeta3 (0.008±0.000)'],
        ['psm_5to1_covariates',
         'Computing (SVM slow)', '—', '—'],
    ],
    col_widths=[5.5, 5.0, 5.0, 5.0],
    font_size=8
)
add_paragraph(doc,
    '* Primary condition. + Validation condition. SAGE: PermutationEstimator, cross-entropy loss, '
    'MarginalImputer, detect_convergence=True. All reported values have CI>0 (statistically informative). '
    'Values in parentheses: SAGE value +/- std. '
    'Consistent features across PSM 1:1 and 2:1 conditions: C3_Beta3, O2_Beta1/MDelta. '
    'psm_5to1_covariates SAGE computing (SVM predict_proba overhead). '
    'Feature naming: [Channel]_[Band]=power; /M=cross-frequency ratio; _coh=coherence.',
    italic=True, size=9, space_before=3, space_after=6)

add_heading(doc, '4.4 Feature Importance (SHAP Beeswarm — Supplementary)', level=2)
shap_interp2 = (
    'SHAP beeswarm plots (supplementary output: shap_beeswarm.png per condition) show the '
    'direction of individual feature contributions. Key directional findings:\n\n'
    '  - C3_Beta3: Higher central Beta-3 power associated with PSEN1 prediction '
    '(consistent with hypersynchrony in pre-symptomatic AD)\n'
    '  - O2_Beta1/MDelta: Higher occipital Beta/Delta ratio associated with PSEN1 '
    '(disrupted cross-frequency coupling)\n'
    '  - O1_Theta_coh: Lower occipital theta coherence associated with PSEN1 '
    '(reduced long-range synchronization)\n'
    '  - age (covariates_in_model): Older age associated strongly with Control class, '
    'younger with PSEN1 — confirms age as dominant signal in unmatched condition\n\n'
    'These directional patterns are consistent with the SAGE global importances and provide '
    'complementary subject-level resolution not available from SAGE alone.'
)
doc.add_paragraph(shap_interp2).runs[0].font.size = Pt(10)

add_heading(doc, '4.5 Stable Features (Majority-Vote Feature Selection)', level=2)
stable_text = (
    'Stable features were defined as those selected in >= 50% of the 10 outer CV folds '
    'by each feature selection method independently. Table 5 lists stable features for '
    'the primary conditions. The overlap between kbest and rfe methods in both '
    'psm_1to1 conditions confirms the reliability of these EEG markers.'
)
doc.add_paragraph(stable_text).runs[0].font.size = Pt(10)

add_paragraph(doc, 'Table 5. Stable features in PSM 1:1 conditions.',
              bold=True, size=10, space_before=6, space_after=3)
make_table(doc,
    headers=['Method', 'PSM 1:1 Residualization', 'PSM 1:1 Covariates'],
    rows=[
        ['SelectKBest', 'FP1_Alpha-1_coh, O1_Alpha-1_coh, C3_Alpha-1/MDelta,\nC3_Alpha-1/MTheta, C4_Alpha-1/MDelta,\nFP1_Alpha-1/MAlpha-1, O1_Theta/MDelta',
                        'FP1_Alpha-1_coh, O1_Alpha-1_coh, C3_Alpha-1/MDelta,\nC3_Alpha-1/MTheta, C4_Alpha-1/MDelta,\nFP1_Alpha-1/MAlpha-1, O1_Theta/MDelta'],
        ['RFE',         'FP1_Alpha-1_coh, O1_Alpha-1_coh, C3_Alpha-1/MTheta,\nC4_Alpha-1/MDelta, O1_Theta/MDelta',
                        'O1_Alpha-2, FP1_Alpha-1_coh, FP1_Delta_coh,\nO1_Alpha-1_coh, C3_Alpha-1/MDelta,\nC3_Alpha-1/MTheta, C4_Alpha-1/MDelta, O1_Theta/MDelta'],
    ],
    col_widths=[3.0, 7.5, 7.5],
    font_size=9
)
add_paragraph(doc,
    'Features selected in >= 50% of outer CV folds. '
    'Features common to both methods and both conditions (bold): '
    'FP1_Alpha-1_coh, O1_Alpha-1_coh, C3_Alpha-1/MTheta, C4_Alpha-1/MDelta, O1_Theta/MDelta.',
    italic=True, size=9, space_before=3, space_after=6)

add_heading(doc, '4.6 Robustness Analysis Results', level=2)
add_paragraph(doc, 'Table 6. Robustness metrics for all conditions.',
              bold=True, size=10, space_before=6, space_after=3)
make_table(doc,
    headers=['Condition', 'Brier', 'ECE', 'AUC_base', 'AUC_noise_50%', 'DELTA_AUC_noise', 'DELTA_AUC_HP', 'Interpretation'],
    rows=[
        ['covariates_in_model',      '0.051', '0.055', '0.975', '0.935', '0.040', '0.007', 'Excellent; very stable'],
        ['residualization',          '0.065', '0.060', '0.921', '0.821', '0.100', '0.004', 'Good calibration; stable'],
        ['psm_1to1_resid. *',        '0.135', '0.049', '0.899', '0.798', '0.101', '0.006', 'Very well calibrated; stable'],
        ['psm_1to1_covar. +',        '0.131', '0.098', '0.904', '0.750', '0.154', '0.005', 'Good; noise-sensitive'],
        ['psm_2to1_resid.',          '0.148', '0.118', '0.827', '0.745', '0.082', '0.006', 'Moderate; stable to noise'],
        ['psm_2to1_covar.',          '0.146', '0.079', '0.886', '0.662', '0.224', '0.006', 'Noise-sensitive at high sigma'],
        ['psm_5to1_resid.',          '0.137', '0.132', '0.812', '0.735', '0.085', '0.020', 'High variance; HP-sensitive'],
        ['psm_5to1_covar.',          '0.119', '0.088', '0.808', '0.830', '0.028', '0.020', 'Low noise drop; unstable CI'],
    ],
    col_widths=[4.8, 1.4, 1.4, 2.2, 3.0, 3.2, 2.8, 4.5],
    font_size=8
)
add_paragraph(doc,
    'Brier: Brier Score on OOF predictions (0=perfect, 0.25=random). '
    'ECE: Expected Calibration Error, 10 bins (0=perfect). '
    'AUC_base: AUC with no added noise (sigma=0%). '
    'AUC_noise_50%: AUC when Gaussian noise of sigma=50% feature SD is injected in test set. '
    'DELTA_AUC_noise: max drop across noise levels. '
    'DELTA_AUC_HP: AUC range when key hyperparameter is varied x0.25 to x4.0.',
    italic=True, size=9, space_before=3, space_after=6)

rob_interp = (
    'Key findings from robustness analyses:\n\n'
    '(1) CALIBRATION: psm_1to1_residualization achieved ECE=0.049 (excellent), substantially '
    'better than standard ComBat results. Brier scores in PSM conditions (0.13-0.15) are '
    'higher than unmatched conditions (~0.05-0.07), reflecting smaller sample sizes. '
    'psm_5to1 conditions show moderate calibration, consistent with small PSEN1 class (n=20).\n\n'
    '(2) NOISE ROBUSTNESS: All PSM conditions maintain AUC > 0.73 even at sigma=50%. '
    'psm_1to1_residualization shows DELTA_AUC_noise=0.101, acceptable for a matched cohort. '
    'psm_1to1_covariates shows higher noise sensitivity (0.154), as adding age as a feature '
    'can amplify noise effects when the model over-relies on age-correlated features.\n\n'
    '(3) HYPERPARAMETER SENSITIVITY: Primary PSM conditions show excellent HP stability '
    '(DELTA_AUC=0.006 for psm_1to1_residualization), confirming that performance is not '
    'an artifact of precise hyperparameter tuning. psm_5to1 shows higher sensitivity '
    '(DELTA_AUC=0.020), consistent with the instability expected for only 20 PSEN1 subjects.'
)
doc.add_paragraph(rob_interp).runs[0].font.size = Pt(10)

# ── 5. ALL CLASSIFIERS TABLE ──────────────────────────────────────────────────
add_heading(doc, '4.7 All Classifier Combinations — Primary Condition', level=2)
add_paragraph(doc, 'Table 7. All classifier-FS combinations for psm_1to1_residualization (primary condition).',
              bold=True, size=10, space_before=6, space_after=3)
make_table(doc,
    headers=['Combo', 'AUC', 'F1', 'Recall', 'Brier'],
    rows=[
        ['RF_rfe *',  '0.892 +/- 0.059', '0.809 +/- 0.084', '0.800 +/- 0.115', '0.135'],
        ['SVM_rfe',   '0.884 +/- 0.045', '0.756 +/- 0.075', '0.738 +/- 0.142', '0.149'],
        ['LR_kbest',  '0.873 +/- 0.094', '0.768 +/- 0.090', '0.738 +/- 0.118', '0.158'],
        ['LR_rfe',    '0.866 +/- 0.076', '0.742 +/- 0.106', '0.713 +/- 0.148', '0.151'],
        ['RF_kbest',  '0.863 +/- 0.087', '0.765 +/- 0.105', '0.738 +/- 0.131', '0.154'],
        ['SVM_kbest', '0.844 +/- 0.143', '0.762 +/- 0.098', '0.738 +/- 0.172', '0.155'],
    ],
    col_widths=[3.5, 4.0, 4.0, 4.0, 2.5],
    font_size=9
)
add_paragraph(doc, '* Best combination selected by mean outer-fold AUC.',
              italic=True, size=9, space_before=3, space_after=8)

# ── 5. DISCUSSION ─────────────────────────────────────────────────────────────
add_heading(doc, '5. Discussion Notes (for manuscript development)', level=1)
disc_text = (
    '5.1 PERFORMANCE IN CONTEXT\n'
    'AUC=0.892 in the primary PSM 1:1 condition represents strong discrimination of '
    'pre-symptomatic PSEN1 carriers from healthy controls using EEG alone. This '
    'outperforms most prior EEG-based dementia classification using matched samples, '
    'and critically, this was achieved with rigorous confound control: age matching, '
    'within-fold age residualization, and reference-based site harmonization. '
    'The bootstrap AUC=0.882 (SD=0.055) confirms that results are stable across '
    'resampling, with minimal optimistic bias from the nested CV procedure.\n\n'
    '5.2 THE ROLE OF HARMONIZATION METHOD\n'
    'The most important methodological finding is that HARMONIZATION APPROACH CRITICALLY '
    'AFFECTS BIOLOGICAL SIGNAL RECOVERY. Standard ComBat applied to all subjects yielded '
    'PSM 1:1 AUC=0.791; reference-based ComBat (fit on controls only) yielded AUC=0.892 — '
    'a 0.101 AUC increase attributable to preserved PSEN1 biological signal. This finding '
    'has broad implications: any multi-site study where patient groups are confounded with '
    'site should use reference-based rather than standard harmonization. The reference-based '
    'approach is methodologically equivalent to an "external validation" paradigm where '
    'the correction model is developed in healthy populations and applied to patients.\n\n'
    '5.3 THE ROLE OF AGE\n'
    'The full-sample condition (covariates_in_model, AUC=0.974) shows that when age is '
    'included, performance is very high — but largely reflects the 28-year age gap. '
    'The PSM + residualization comparison (AUC=0.892 vs. 0.905 with age as feature) '
    'confirms that within the age-matched cohort, age contributes only 0.013 AUC, '
    'validating the EEG biological signal.\n\n'
    '5.4 CENTRAL-TEMPORAL BETA/GAMMA AS BIOMARKERS (SAGE-VERIFIED)\n'
    'SAGE analysis confirms central-temporal beta and gamma oscillations as primary discriminative '
    'features with formal statistical guarantees. C3_Beta3 (SAGE=0.088, CI=[0.085,0.092]) and '
    'O2_Beta1/MDelta (SAGE=0.059, CI=[0.055,0.062]) are the most consistent features across '
    'PSM conditions, with CIs well above zero. The discovery of T6_Gamma (temporal gamma) in '
    'the residualization condition (SAGE=0.030) extends the gamma signature from central to '
    'temporal regions, consistent with widespread GABAergic disruption in PSEN1. '
    'SAGE/SHAP convergent validity (rho>0.90 in primary conditions) provides independent '
    'confirmation of these features under two different interpretability frameworks.\n\n'
    '5.5 LIMITATIONS\n'
    '- Small PSEN1 sample (n=91, 80 after PSM): limits power, especially for PSM 5:1\n'
    '- All PSEN1 from single region (Medellin): 32 Medellin controls available for reference harmonization, marginal for 3 sub-sites\n'
    '- Sex data unavailable for majority: sex effects not evaluable\n'
    '- Resting-state EEG only: task paradigms might yield stronger biomarkers\n'
    '- Cross-sectional design: longitudinal data would strengthen causal interpretation\n'
    '- PSM sensitivity (age+education) infeasible (n=4-8 after matching due to demographic mismatch)\n\n'
    '5.6 KEY MESSAGES FOR PUBLICATION\n'
    '1. With reference-based ComBat + PSM + age residualization, EEG resting-state discriminates PSEN1 carriers with AUC=0.892\n'
    '2. Reference-based harmonization recovers 0.101 AUC vs. standard ComBat — a methodological advance for confounded multi-site designs\n'
    '3. SAGE confirms C3_Beta3, O2_Beta1/MDelta, O1_Theta_coh as statistically informative EEG biomarkers (CI>0) across PSM conditions\n'
    '4. T6_Gamma (temporal gamma) emerges in the full-sample condition, extending the gamma disruption signature\n'
    '5. PSM + residualization vs. PSM + covariates comparison (delta AUC=0.013) validates EEG as the signal source\n'
    '6. SAGE/SHAP convergent validity (rho>0.90 in primary conditions) provides cross-method confirmation of feature importance\n'
    '7. The progression covariates_in_model (0.974) > residualization (0.963) > PSM_1to1 (0.892) reflects rigorous confound removal'
)
doc.add_paragraph(disc_text).runs[0].font.size = Pt(10)

# ── 6. PIPELINE TECHNICAL NOTES ────────────────────────────────────────────────
add_heading(doc, '6. Technical Pipeline Notes', level=1)
tech_text = (
    'Software: Python 3.11, scikit-learn 1.x, imbalanced-learn, neuroHarmonize, '
    'sage-importance, shap, pandas, numpy, matplotlib/seaborn.\n\n'
    'Pipeline execution order:\n'
    '  1. 1_make_dataframe.py: Merge 5 EEG metric feathers, enrich with demographics\n'
    '  2. optional_neuroharmonize.py: Reference-based ComBat (fit on Controls, apply to all)\n'
    '  3. 2_apply_psm.py: PSM on harmonized data, primary (age only) and sensitivity (age+education) scenarios\n'
    '  4. 3_train_ml_v2.py: Nested CV, 8 conditions, SAGE (primary) + SHAP beeswarm (supplementary) + 4 robustness analyses\n'
    '  5. run_sage_only.py: SAGE analysis on existing final models (no re-training required)\n\n'
    'HARMONIZATION CORRECTIONS APPLIED:\n'
    '(1) negativeTest() bug: function returns COUNT of negatives (scalar), not clipped array. '
    'This caused all EEG features to be set to 6.0. '
    'Fixed: n_neg = negativeTest(X_adj); X_adj = np.maximum(X_adj, 0).\n'
    '(2) Standard ComBat applied to all subjects: absorbed PSEN1 biological signal into '
    'Medellin site effect because PSEN1=100% Medellin. group_binary covariate was tested '
    'but did not fully resolve the issue due to partial collinearity.\n'
    '(3) Reference-based ComBat (FINAL APPROACH): harmonizationLearn on Controls only '
    '(n=5,207 rows, including 280 Medellin control rows), harmonizationApply to all '
    '(n=6,283 rows). Site encoding (integer codes) applied consistently using a mapping '
    'derived from the full dataset before the Control/All split.\n\n'
    'Feature exclusion rationale: education, sex, and SITE dummies excluded from ML. '
    'Their inclusion yielded artifactually high AUC (0.99+) from data availability patterns.'
)
doc.add_paragraph(tech_text).runs[0].font.size = Pt(10)

# ── save ──────────────────────────────────────────────────────────────────────
outpath = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\Resultados\PSEN1_EEG_ML_Report_v4.docx'
doc.save(outpath)
print(f'[OK] Report saved: {outpath}')
