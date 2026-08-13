"""
Comprehensive DOCX update — all results, tables, methods, harmonization section.
"""
import pandas as pd
import numpy as np
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from copy import deepcopy

DOC_PATH = 'E:/Academico/Universidad/Posgrado/Tesis/Datos/PORTABLES/Resultados/PSEN1_EEG_ML_Report.docx'
SAGE_BASE = 'E:/Academico/Universidad/Posgrado/Tesis/Datos/PORTABLES/Resultados/graphics/ML_v2/PSEN1_vs_Control'

doc = Document(DOC_PATH)

# ── helpers ───────────────────────────────────────────────────────────────────
def set_para(idx, text):
    p = doc.paragraphs[idx]
    for run in p.runs:
        run.text = ''
    if p.runs:
        p.runs[0].text = text
    else:
        p.add_run(text)

def set_cell(table, row, col, text):
    table.rows[row].cells[col].text = text

def bold_cell(table, row, col):
    for para in table.rows[row].cells[col].paragraphs:
        for run in para.runs:
            run.bold = True

def get_sage(cond, n=3):
    df = pd.read_excel(f'{SAGE_BASE}/{cond}/sage_importance.xlsx')
    rows = []
    for _, r in df.head(n).iterrows():
        feat = r['feature']
        val  = r['sage_value']
        std  = r['sage_std']
        rows.append(f"{feat}\n({val:.3f}\u00b1{std:.3f})")
    return rows

# ── NEW RESULTS DATA ──────────────────────────────────────────────────────────
# (condition, N, model, AUC±SD, F1±SD, Recall±SD, Brier, Bootstrap AUC±SD)
RESULTS = [
    ('covariates_in_model',      '539', 'RF_rfe', '0.968\u00b10.018', '0.811\u00b10.076', '0.858\u00b10.085', '0.052', '0.973\u00b10.009'),
    ('residualization',          '539', 'RF_rfe', '0.956\u00b10.014', '0.759\u00b10.066', '0.846\u00b10.102', '0.066', '0.896\u00b10.032'),
    ('psm_1to1_residualization *','164', 'RF_rfe', '0.878\u00b10.084', '0.798\u00b10.120', '0.790\u00b10.158', '0.141', '0.873\u00b10.050'),
    ('psm_1to1_covariates +',    '164', 'RF_rfe', '0.876\u00b10.095', '0.816\u00b10.132', '0.812\u00b10.170', '0.138', '0.878\u00b10.040'),
    ('psm_2to1_residualization', '147', 'RF_rfe', '0.867\u00b10.119', '0.678\u00b10.161', '0.720\u00b10.204', '0.155', '0.867\u00b10.047'),
    ('psm_2to1_covariates',      '147', 'RF_rfe', '0.841\u00b10.112', '0.666\u00b10.161', '0.720\u00b10.204', '0.158', '0.869\u00b10.053'),
    ('psm_5to1_residualization', '120', 'RF_kbest','0.895\u00b10.105', '0.637\u00b10.231', '0.650\u00b10.320', '0.091', '0.842\u00b10.083'),
    ('psm_5to1_covariates',      '120', 'RF_kbest','0.857\u00b10.176', '0.707\u00b10.143', '0.700\u00b10.245', '0.095', '0.842\u00b10.083'),
]

ROBUSTNESS = [
    # cond, Brier, ECE, AUC_base, AUC_noise50, DELTA_noise, DELTA_HP, note
    ('covariates_in_model', '0.052','0.059','0.972','0.922','0.050','0.003','Excellent; very stable'),
    ('residualization',     '0.066','0.064','0.906','0.817','0.089','0.011','Good calibration; stable'),
    ('psm_1to1_resid. *',   '0.141','0.054','0.891','0.793','0.098','0.002','Well calibrated; very stable'),
    ('psm_1to1_covar. +',   '0.139','0.079','0.888','0.775','0.113','0.005','Good; noise-sensitive with age'),
    ('psm_2to1_resid.',     '0.154','0.096','0.853','0.786','0.067','0.010','Moderate; stable to noise'),
    ('psm_2to1_covar.',     '0.157','0.085','0.860','0.724','0.136','0.012','Noise-sensitive at high sigma'),
    ('psm_5to1_resid.',     '0.091','0.073','0.879','0.825','0.080','0.020','High variance; HP-sensitive'),
    ('psm_5to1_covar.',     '0.095','0.106','0.879','0.850','0.046','0.020','Low noise drop; unstable CI'),
]

ALL_COMBOS_PRIMARY = [
    # combo, AUC, F1, Recall, Brier
    ('RF_rfe *',  '0.878\u00b10.084', '0.798\u00b10.120', '0.790\u00b10.158', '0.140'),
    ('RF_kbest',  '0.876\u00b10.095', '0.797\u00b10.119', '0.768\u00b10.140', '0.141'),
    ('LR_rfe',    '0.865\u00b10.065', '0.758\u00b10.138', '0.704\u00b10.172', '0.151'),
    ('SVM_rfe',   '0.861\u00b10.066', '0.768\u00b10.131', '0.717\u00b10.161', '0.153'),
    ('LR_kbest',  '0.818\u00b10.126', '0.748\u00b10.212', '0.694\u00b10.229', '0.154'),
    ('SVM_kbest', '0.705\u00b10.275', '0.641\u00b10.288', '0.626\u00b10.307', '0.167'),
]

# =============================================================================
# 1. PARAGRAPH UPDATES
# =============================================================================

# [7] Remove "sex as covariates" reference
set_para(7,
    "Given that PSEN1 carriers are exclusively from Medellin while controls span 10 sites, site "
    "harmonization via neuroHarmonize (reference-based ComBat) was applied prior to all analyses to "
    "remove site-related variance from EEG features while preserving biological variance. Age was the "
    "sole biological covariate; sex was excluded because Seoul (n=210 controls, 47%) and Medellin_ld "
    "(n=41 PSEN1, 45%) have 0% sex coverage, making imputation physiologically indefensible. "
    "Education and sex were also excluded from the classifier feature set due to severely imbalanced "
    "data availability, which would create artifactual classification signal unrelated to EEG biology.")

# [34] Confirm per-type harmonization — already correct, minor clarification
set_para(34,
    "Features were log-transformed prior to harmonization (log(0.001+x)) and back-transformed after "
    "(exp(x)\u22120.001). All covariate values were restored from the original data after harmonization. "
    "Harmonization was applied independently per feature type using five separate ComBat models: "
    "spectral power (64 features), synchronization likelihood (64), coherence (64), entropy (64), "
    "and cross-frequency coupling (288). This ensures that site-effect estimation is not dominated by "
    "the largest feature block. Education and sex were excluded from the ML feature set because Seoul "
    "controls (n=210, 47% of controls) lack both variables, which would create artifactual "
    "classification signal.")

# [53] Update feature count after pre-filtering
set_para(53,
    "Result: 63 features (unmatched) / 40\u201354 features (PSM conditions), varying by condition "
    "and feature selection method.")

# All paragraph updates already done in previous session are retained.
# Additional: update para [6] - Control n=448, not 442
# (448 subjects, but 6 ADMCI mixed — check)

print("Paragraph updates done.")

# =============================================================================
# 2. INSERT HARMONIZATION VALIDATION SECTION (after para [34], before ML)
# =============================================================================
# We'll add new paragraphs after index 34 (before index 35 = 3.3 ML)

def insert_paragraph_after(doc, ref_para, text, style=None):
    """Insert a new paragraph after ref_para."""
    new_para = OxmlElement('w:p')
    ref_para._p.addnext(new_para)
    from docx.text.paragraph import Paragraph
    p = Paragraph(new_para, ref_para._parent)
    run = p.add_run(text)
    if style:
        p.style = doc.styles[style]
    return p

# Insert after para 34
ref = doc.paragraphs[34]

p_title = insert_paragraph_after(doc, ref,
    "3.2.1 Harmonization Validation — Quantitative Before/After Assessment")
p_title.runs[0].bold = True

p1 = insert_paragraph_after(doc, p_title,
    "To validate that the reference-based ComBat successfully removed technical site variance without "
    "destroying biological signal, three complementary diagnostics were computed on the full dataset "
    "(n=6,283 rows, 544 EEG features):")

p2 = insert_paragraph_after(doc, p1,
    "QUANTITATIVE SITE VARIANCE REDUCTION (R\u00b2 regression): For each feature, the proportion of "
    "variance explained by site was estimated as R\u00b2 from a linear regression of the feature on "
    "site dummy variables. Before harmonization: mean R\u00b2 = 0.334 (median = 0.274), indicating "
    "that on average 33.4% of each feature's variance was attributable to site differences. After "
    "harmonization: mean R\u00b2 = 0.034 (median = 0.024) \u2014 an 89.7% reduction in site-explained "
    "variance. Nearly all features shifted below the identity line in the before/after scatter plot, "
    "confirming consistent site-effect removal across all five feature types.")

p3 = insert_paragraph_after(doc, p2,
    "PCA VISUALIZATION: Principal component analysis (PCA fit on pre-harmonization data, same "
    "projection applied to post-harmonization data) shows that before harmonization, the first two "
    "PCs are dominated by site clustering \u2014 sites form visually distinct clouds. After "
    "harmonization, site clustering is substantially reduced while the separation between PSEN1 "
    "carriers and Controls is preserved, confirming that biological variance was not destroyed by "
    "the correction. ADMCI subjects (n=985) form an intermediate cluster in the group-colored panels, "
    "consistent with progressive neurodegeneration.")

p4 = insert_paragraph_after(doc, p3,
    "PER-SITE FEATURE DISTRIBUTION: Boxplot distributions of C3_Beta3 (top SAGE discriminative "
    "feature; SAGE=0.096) per site show visually heterogeneous medians and spreads before "
    "harmonization (up to 3-fold differences across sites). After harmonization, site medians "
    "converge substantially while within-group variability is preserved, validating that the "
    "correction appropriately targets technical rather than biological variance.")

p5 = insert_paragraph_after(doc, p4,
    "Diagnostic figures are saved in Resultados/harmonization_models/: "
    "harmonization_pca_before_after.png, harmonization_site_r2_before_after.png, "
    "harmonization_feature_distribution_before_after.png, and combat_params_[type].png "
    "(gamma*/delta* parameter distributions per site for each of the five feature-type models).")

print("Harmonization validation section inserted.")

# =============================================================================
# 3. UPDATE TABLE 1 (PSM Balance) — Table index 1
# =============================================================================
t1 = doc.tables[1]
# Row 2: PSM 1:1 — 80/80 -> 82/82, t=-0.17->-0.21, p=0.869->0.835
set_cell(t1, 2, 0, 'PSM 1:1 (primary)')
set_cell(t1, 2, 1, '82 / 82')
set_cell(t1, 2, 2, '32.1 \u00b1 6.7')
set_cell(t1, 2, 3, '32.4 \u00b1 6.7')
set_cell(t1, 2, 4, '-0.21')
set_cell(t1, 2, 5, '0.835')
# Row 3: PSM 2:1
set_cell(t1, 3, 2, '34.0 \u00b1 7.0')
set_cell(t1, 3, 3, '34.2 \u00b1 7.3')
set_cell(t1, 3, 4, '-0.11')
set_cell(t1, 3, 5, '0.910')
# Row 4: PSM 5:1
set_cell(t1, 4, 1, '20 / 100')
set_cell(t1, 4, 2, '35.0 \u00b1 7.5')
set_cell(t1, 4, 3, '35.8 \u00b1 8.2')
set_cell(t1, 4, 4, '-0.36')
set_cell(t1, 4, 5, '0.717')

print("Table 1 (PSM balance) updated.")

# =============================================================================
# 4. UPDATE TABLE 2 (Main Results) — Table index 2
# =============================================================================
t2 = doc.tables[2]
for i, (cond, n, model, auc, f1, rec, brier, boot) in enumerate(RESULTS):
    row = i + 1  # row 0 is header
    set_cell(t2, row, 0, cond)
    set_cell(t2, row, 1, n)
    set_cell(t2, row, 2, model)
    set_cell(t2, row, 3, auc)
    set_cell(t2, row, 4, f1)
    set_cell(t2, row, 5, rec)
    set_cell(t2, row, 6, brier)
    set_cell(t2, row, 7, boot)

print("Table 2 (Main results) updated.")

# =============================================================================
# 5. UPDATE TABLE 3 (SAGE) — Table index 3
# =============================================================================
t3 = doc.tables[3]

sage_data = {
    'covariates_in_model':       get_sage('covariates_in_model'),
    'residualization':           get_sage('residualization'),
    'psm_1to1_residualization *': get_sage('psm_1to1_residualization'),
    'psm_1to1_covariates +':     get_sage('psm_1to1_covariates'),
    'psm_2to1_residualization':  get_sage('psm_2to1_residualization'),
    'psm_2to1_covariates':       get_sage('psm_2to1_covariates'),
    'psm_5to1_residualization':  get_sage('psm_5to1_residualization'),
    'psm_5to1_covariates':       get_sage('psm_5to1_covariates'),
}

cond_labels = list(sage_data.keys())
models_sage = ['RF_rfe','RF_rfe','RF_rfe','RF_rfe','RF_rfe','RF_rfe','RF_kbest','RF_kbest']

for i, (cond, top3) in enumerate(sage_data.items()):
    row = i + 1
    set_cell(t3, row, 0, cond)
    set_cell(t3, row, 1, top3[0] if len(top3) > 0 else '—')
    set_cell(t3, row, 2, top3[1] if len(top3) > 1 else '—')
    set_cell(t3, row, 3, top3[2] if len(top3) > 2 else '—')
    set_cell(t3, row, 4, models_sage[i])

print("Table 3 (SAGE) updated.")

# =============================================================================
# 6. UPDATE TABLE 5 (Robustness) — Table index 5
# =============================================================================
t5 = doc.tables[5]
for i, (cond, brier, ece, auc_base, auc50, d_noise, d_hp, note) in enumerate(ROBUSTNESS):
    row = i + 1
    set_cell(t5, row, 0, cond)
    set_cell(t5, row, 1, brier)
    set_cell(t5, row, 2, ece)
    set_cell(t5, row, 3, auc_base)
    set_cell(t5, row, 4, auc50)
    set_cell(t5, row, 5, d_noise)
    set_cell(t5, row, 6, d_hp)
    set_cell(t5, row, 7, note)

print("Table 5 (Robustness) updated.")

# =============================================================================
# 7. UPDATE TABLE 6 (All combos primary) — Table index 6
# =============================================================================
t6 = doc.tables[6]
for i, (combo, auc, f1, rec, brier) in enumerate(ALL_COMBOS_PRIMARY):
    row = i + 1
    set_cell(t6, row, 0, combo)
    set_cell(t6, row, 1, auc)
    set_cell(t6, row, 2, f1)
    set_cell(t6, row, 3, rec)
    set_cell(t6, row, 4, brier)

print("Table 6 (All combos) updated.")

# =============================================================================
# 8. SAVE
# =============================================================================
doc.save(DOC_PATH)
print(f"\nDocument saved: {DOC_PATH}")
print("All updates complete.")
