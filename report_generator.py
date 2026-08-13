"""
report_generator.py — Generación de Reportes Estadísticos
===========================================================
Genera documentos Word con:
  - Tabla 1: Características demográficas por grupo (antes y después del PSM)
  - Tabla 2: Rendimiento de modelos ML por condición
  - Pruebas estadísticas con corrección FDR (Benjamini-Hochberg)
  - Resumen de features seleccionadas

Uso:
    python report_generator.py

Requiere: pip install python-docx openpyxl scipy statsmodels
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_PATH   = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES'
RESULTS_DIR = os.path.join(BASE_PATH, 'Resultados')
ML_V2_DIR   = os.path.join(RESULTS_DIR, 'graphics', 'ML_v2', 'PSEN1_vs_Control')
DEMO_EXCEL  = os.path.join(BASE_PATH, 'Portables_pruebas_completa.xlsx')

GROUP1     = 'PSEN1'
GROUP2     = 'Control'
DATA_TYPE  = 'ce'
SPACE      = 'roi'

CONDITIONS = ['no_matching', 'psm_1to1', 'psm_2to1', 'psm_5to1']

# =============================================================================
# PRUEBAS ESTADÍSTICAS
# =============================================================================

def normality_test(values):
    """Shapiro-Wilk. Retorna True si los datos son normales (p >= 0.05)."""
    v = values.dropna()
    if len(v) < 3:
        return False
    if len(v) > 5000:
        v = v.sample(5000, random_state=42)
    _, p = stats.shapiro(v)
    return p >= 0.05


def test_continuous(g1_vals, g2_vals):
    """
    Prueba paramétrica o no paramétrica según normalidad.
    Retorna: p_value, test_name, effect_size, direction (g1 > g2 ? '+' : '-')
    """
    v1 = g1_vals.dropna()
    v2 = g2_vals.dropna()
    if len(v1) < 3 or len(v2) < 3:
        return np.nan, 'n/a', np.nan, ''

    norm1 = normality_test(v1)
    norm2 = normality_test(v2)

    if norm1 and norm2:
        stat, p = stats.ttest_ind(v1, v2)
        pooled  = np.sqrt((v1.std() ** 2 + v2.std() ** 2) / 2)
        d       = (v1.mean() - v2.mean()) / pooled if pooled > 0 else 0
        test    = 't-test'
    else:
        stat, p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        u_max   = len(v1) * len(v2)
        d       = (stat / u_max - 0.5) * 2  # rank-biserial correlation
        test    = 'Mann-Whitney'

    direction = '+' if v1.mean() > v2.mean() else '-'
    return float(p), test, float(d), direction


def test_categorical(g1_vals, g2_vals):
    """
    Chi-cuadrado para variables categóricas (ej. sexo).
    Retorna: p_value, test_name, effect_size (Cramér's V)
    """
    combined = pd.concat([
        pd.Series(g1_vals.values, name='val').assign(group='g1'),
        pd.Series(g2_vals.values, name='val').assign(group='g2'),
    ])
    ct = pd.crosstab(combined['group'], combined['val'])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, 'n/a', np.nan

    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n  = ct.sum().sum()
    v  = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if n > 0 else 0
    return float(p), 'chi-squared', float(v)


def apply_fdr_correction(p_values, alpha=0.05):
    """
    Corrección FDR de Benjamini-Hochberg.
    Retorna p_values corregidos y array booleano de significancia.
    """
    try:
        from statsmodels.stats.multitest import multipletests
        valid_mask = ~np.isnan(p_values)
        p_fdr      = np.full(len(p_values), np.nan)
        rejected   = np.zeros(len(p_values), dtype=bool)
        if valid_mask.sum() > 0:
            rej, p_corr, _, _ = multipletests(
                np.array(p_values)[valid_mask], method='fdr_bh', alpha=alpha
            )
            p_fdr[valid_mask]    = p_corr
            rejected[valid_mask] = rej
        return p_fdr, rejected
    except ImportError:
        # Si statsmodels no está disponible, usar Bonferroni simple
        p_arr  = np.array(p_values)
        n_comp = np.sum(~np.isnan(p_arr))
        p_fdr  = np.where(np.isnan(p_arr), np.nan, np.minimum(p_arr * n_comp, 1.0))
        return p_fdr, p_fdr < alpha


def build_stats_table(data, group1, group2, variables):
    """
    Construye tabla de estadística descriptiva + pruebas para un dataset.
    variables: lista de dicts {name, label, type ('continuous'|'categorical')}
    """
    g1 = data[data['group'] == group1]
    g2 = data[data['group'] == group2]

    rows      = []
    p_values  = []

    for var in variables:
        col   = var['name']
        label = var.get('label', col)
        vtype = var.get('type', 'continuous')

        if col not in data.columns:
            continue

        row = {
            'Variable': label,
            f'N_{group1}': g1[col].notna().sum(),
            f'N_{group2}': g2[col].notna().sum(),
        }

        if vtype == 'continuous':
            for grp_lbl, grp_df in [(group1, g1), (group2, g2)]:
                v = grp_df[col].dropna()
                row[f'{grp_lbl}_mean'] = round(v.mean(), 2) if len(v) > 0 else np.nan
                row[f'{grp_lbl}_std']  = round(v.std(),  2) if len(v) > 0 else np.nan
                row[f'{grp_lbl}_median'] = round(v.median(), 2) if len(v) > 0 else np.nan

            p, test, effect, direction = test_continuous(g1[col], g2[col])
            row.update({'p_raw': p, 'test': test, 'effect_size': round(effect, 3)
                        if not np.isnan(effect) else np.nan, 'direction': direction})

        else:  # categorical
            for grp_lbl, grp_df in [(group1, g1), (group2, g2)]:
                counts = grp_df[col].value_counts()
                row[f'{grp_lbl}_dist'] = ' | '.join([f"{k}:{v}" for k, v in counts.items()])

            p, test, effect = test_categorical(g1[col], g2[col])
            row.update({'p_raw': p, 'test': test, 'effect_size': round(effect, 3)
                        if not np.isnan(effect) else np.nan, 'direction': ''})

        rows.append(row)
        p_values.append(p if not np.isnan(p) else np.nan)

    # FDR correction
    p_fdr, rejected = apply_fdr_correction(p_values)
    for i, row in enumerate(rows):
        row['p_fdr']       = round(p_fdr[i], 4) if not np.isnan(p_fdr[i]) else np.nan
        row['sig_fdr']     = '✓' if rejected[i] else ''
        row['sig_raw']     = '✓' if (not np.isnan(row.get('p_raw', np.nan))
                                     and row['p_raw'] < 0.05) else ''

    return pd.DataFrame(rows)


# =============================================================================
# CARGA DE DATOS Y DEMOGRAFÍA
# =============================================================================

def load_data_with_demographics(data_feather, demo_excel=None, group1=GROUP1, group2=GROUP2):
    """
    Carga datos EEG + opcionalmente los une con demografía del Excel.
    Retorna DataFrame a nivel de sujeto único.
    """
    data = pd.read_feather(data_feather)
    g2   = group2 if isinstance(group2, list) else [group2]
    data = data[data['group'].isin([group1] + g2)].copy()
    data['group'] = data['group'].apply(lambda x: g2[0] if x in g2 else x)

    # Un registro por sujeto
    subj = data.groupby('subject').first().reset_index()

    # Unir con Excel demográfico si está disponible
    if demo_excel and os.path.exists(demo_excel):
        try:
            demo = pd.read_excel(demo_excel)
            demo['subject_norm'] = 'sub-' + demo['subject'].str.replace('_', '', regex=False)
            keep = ['subject_norm', 'Sexo', 'Edad',
                    'Años de escolaridad', 'Nivel socioeconómico', 'MMSE', 'MoCA', 'CDR']
            keep = [c for c in keep if c in demo.columns]
            demo = demo[keep].rename(columns={
                'subject_norm':        'subject',
                'Sexo':                'sex',
                'Edad':                'age_demo',
                'Años de escolaridad': 'education',
                'Nivel socioeconómico':'ses',
                'MMSE':                'mmse',
                'MoCA':                'moca',
                'CDR':                 'cdr',
            })
            subj = subj.merge(demo, on='subject', how='left')
            if 'age' not in subj.columns and 'age_demo' in subj.columns:
                subj['age'] = subj['age_demo']
            elif 'age_demo' in subj.columns:
                subj['age'] = subj['age'].fillna(subj['age_demo'])
        except Exception as e:
            print(f"  Advertencia: no se pudo cargar demografía: {e}")

    return subj


# =============================================================================
# GENERACIÓN DE WORD DOCUMENT
# =============================================================================

def _add_table_to_doc(doc, df, title=''):
    """Agrega una tabla pandas al documento Word."""
    try:
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        pass

    if title:
        doc.add_heading(title, level=2)

    # Tabla
    n_rows, n_cols = df.shape
    table = doc.add_table(rows=1 + n_rows, cols=n_cols)
    table.style = 'Table Grid'

    # Encabezados
    hdr = table.rows[0].cells
    for j, col in enumerate(df.columns):
        hdr[j].text = str(col)
        try:
            run = hdr[j].paragraphs[0].runs[0]
            run.bold = True
            run.font.size = Pt(9)
        except Exception:
            pass

    # Datos
    for i, row_vals in enumerate(df.itertuples(index=False)):
        cells = table.rows[i + 1].cells
        for j, val in enumerate(row_vals):
            txt = '' if pd.isna(val) else str(val)
            cells[j].text = txt
            try:
                run = cells[j].paragraphs[0].runs[0]
                run.font.size = Pt(9)

                # Color verde si significativo
                col_name = df.columns[j]
                if col_name in ('sig_fdr', 'sig_raw') and txt == '✓':
                    run.font.color.rgb = RGBColor(0, 128, 0)
            except Exception:
                pass

    doc.add_paragraph('')  # espacio


def generate_report(output_path):
    """
    Genera el documento Word completo con:
      - Tabla 1: Demografía por grupo (antes y después de PSM)
      - Tabla 2: Resultados ML por condición
      - Tabla 3: Features estables (voto mayoritario)
    """
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        print("ERROR: Instala python-docx:  pip install python-docx")
        return

    doc = Document()

    # Título
    doc.add_heading(f'Reporte Estadístico: {GROUP1} vs {GROUP2}', 0)
    doc.add_paragraph(
        f'Pipeline ML v2 | Data type: {DATA_TYPE.upper()} | Space: {SPACE.upper()}'
    )

    # -------------------------------------------------------------------------
    # TABLA 1: Demografía pre-PSM (todos los sujetos)
    # -------------------------------------------------------------------------
    doc.add_heading('Tabla 1. Características demográficas (todos los sujetos)', level=1)
    doc.add_paragraph(
        'Comparación de variables demográficas antes del PSM. '
        'Pruebas: t-test o Mann-Whitney para variables continuas; '
        'Chi-cuadrado para variables categóricas. '
        'Corrección FDR de Benjamini-Hochberg aplicada.'
    )

    full_data_path = os.path.join(
        RESULTS_DIR, f'Data_complete_{DATA_TYPE}_{SPACE.lower()}.feather'
    )
    if os.path.exists(full_data_path):
        subj_all = load_data_with_demographics(full_data_path, DEMO_EXCEL)
        subj_all = subj_all[subj_all['group'].isin([GROUP1, GROUP2])]

        variables = [
            {'name': 'age',       'label': 'Edad (años)',         'type': 'continuous'},
            {'name': 'education', 'label': 'Escolaridad (años)',   'type': 'continuous'},
            {'name': 'mmse',      'label': 'MMSE',                 'type': 'continuous'},
            {'name': 'moca',      'label': 'MoCA',                 'type': 'continuous'},
            {'name': 'sex',       'label': 'Sexo (F/M)',           'type': 'categorical'},
        ]

        stats_table = build_stats_table(subj_all, GROUP1, GROUP2, variables)
        _add_table_to_doc(doc, stats_table,
                          f'N: {GROUP1}={subj_all[subj_all.group==GROUP1].shape[0]}, '
                          f'{GROUP2}={subj_all[subj_all.group==GROUP2].shape[0]}')

        # Guardar también como Excel
        stats_table.to_excel(
            os.path.join(os.path.dirname(output_path), 'demographic_stats_prePSM.xlsx'),
            index=False
        )
    else:
        doc.add_paragraph('Archivo de datos no encontrado.')

    # -------------------------------------------------------------------------
    # TABLA 2: Demografía post-PSM por ratio
    # -------------------------------------------------------------------------
    doc.add_heading('Tabla 2. Características demográficas tras PSM', level=1)

    for ratio in ['1to1', '2to1', '5to1']:
        psm_path = os.path.join(
            RESULTS_DIR, 'PSM_datasets',
            f'Data_matched_{DATA_TYPE}_{SPACE.lower()}_{GROUP1}_{ratio}.feather'
        )
        if not os.path.exists(psm_path):
            continue

        subj_psm = load_data_with_demographics(psm_path, DEMO_EXCEL)
        subj_psm = subj_psm[subj_psm['group'].isin([GROUP1, GROUP2])]

        doc.add_heading(f'PSM {ratio}', level=3)
        stats_psm = build_stats_table(subj_psm, GROUP1, GROUP2, variables)
        _add_table_to_doc(doc, stats_psm,
                          f'N: {GROUP1}={subj_psm[subj_psm.group==GROUP1].shape[0]}, '
                          f'{GROUP2}={subj_psm[subj_psm.group==GROUP2].shape[0]}')

    # -------------------------------------------------------------------------
    # TABLA 3: Rendimiento ML por condición y combo
    # -------------------------------------------------------------------------
    doc.add_heading('Tabla 3. Rendimiento de clasificación ML (Nested CV)', level=1)
    doc.add_paragraph(
        f'Resultados de nested cross-validation. '
        f'Métricas: media ± desv. estándar sobre los folds externos. '
        f'Clasificadores: RF (Random Forest), SVM (Support Vector Machine), '
        f'LR (Logistic Regression). '
        f'Feature selection: KBest (SelectKBest) y RFE (Recursive Feature Elimination). '
        f'Brier Score: calibración (0=perfecto, 0.25=azar).'
    )

    comparison_path = os.path.join(ML_V2_DIR, 'comparison_all_conditions.xlsx')
    if os.path.exists(comparison_path):
        df_comp = pd.read_excel(comparison_path)

        # Tabla resumida: mejor por condición
        doc.add_heading('Mejor modelo por condición', level=3)
        best_rows = (
            df_comp.sort_values('auc_mean', ascending=False)
            .groupby('condition').first().reset_index()
        )
        cols_show = ['condition', 'combo', 'n_subjects', 'n_pos',
                     'auc_mean', 'auc_std', 'f1_mean', 'f1_std',
                     'recall_mean', 'precision_mean', 'brier_mean']
        cols_show = [c for c in cols_show if c in best_rows.columns]
        best_rows_fmt = best_rows[cols_show].copy()

        # Formatear AUC como "mean±std"
        for metric in ['auc', 'f1', 'recall', 'precision']:
            if f'{metric}_mean' in best_rows_fmt.columns and f'{metric}_std' in best_rows_fmt.columns:
                best_rows_fmt[f'{metric}'] = (
                    best_rows_fmt[f'{metric}_mean'].map('{:.3f}'.format)
                    + '±'
                    + best_rows_fmt[f'{metric}_std'].map('{:.3f}'.format)
                )
                best_rows_fmt.drop([f'{metric}_mean', f'{metric}_std'], axis=1, inplace=True)

        _add_table_to_doc(doc, best_rows_fmt)

        # Tabla completa
        doc.add_heading('Todos los combos por condición', level=3)
        cols_full = ['condition', 'combo', 'auc_mean', 'auc_std',
                     'f1_mean', 'recall_mean', 'brier_mean']
        cols_full = [c for c in cols_full if c in df_comp.columns]
        df_full   = df_comp[cols_full].sort_values(['condition', 'auc_mean'], ascending=[True, False])
        _add_table_to_doc(doc, df_full)
    else:
        doc.add_paragraph(
            f'Tabla de comparación no encontrada. Ejecutar 3_train_ml_v2.py primero.\n'
            f'Esperada en: {comparison_path}'
        )

    # -------------------------------------------------------------------------
    # TABLA 4: Features estables (voto mayoritario)
    # -------------------------------------------------------------------------
    doc.add_heading('Tabla 4. Features seleccionadas (voto mayoritario ≥ 50% de folds)', level=1)
    doc.add_paragraph(
        'Features que fueron seleccionadas en la mayoría de los folds externos '
        'por cada método (SelectKBest y RFE). Indicadas por condición.'
    )

    feat_rows = []
    for cond in CONDITIONS:
        stable_path = os.path.join(ML_V2_DIR, cond, 'stable_features.xlsx')
        if os.path.exists(stable_path):
            sf = pd.read_excel(stable_path)
            for method in ['kbest_stable', 'rfe_stable']:
                if method in sf.columns:
                    feats = sf[method].dropna().tolist()
                    for i, f in enumerate(feats):
                        feat_rows.append({
                            'Condición': cond,
                            'Método':    method.replace('_stable', '').upper(),
                            'Rank':      i + 1,
                            'Feature':   str(f),
                        })

    if feat_rows:
        df_feats = pd.DataFrame(feat_rows)
        _add_table_to_doc(doc, df_feats)
    else:
        doc.add_paragraph('No se encontraron archivos de features estables.')

    # -------------------------------------------------------------------------
    # NOTAS METODOLÓGICAS
    # -------------------------------------------------------------------------
    doc.add_heading('Notas metodológicas', level=1)
    doc.add_paragraph(
        '1. Nested cross-validation: folds externos (K=10) para estimación imparcial; '
        'folds internos (K=5) para ajuste de hiperparámetros (RandomizedSearchCV).\n'
        '2. Todo el preprocessing (imputación KNN, estandarización, SMOTE) se realiza '
        'DENTRO de cada fold para evitar data leakage.\n'
        '3. Pre-filtrado de features (antes de CV): eliminación de varianza cercana a cero '
        'y correlación > 0.85 (no usa etiquetas → sin leakage).\n'
        '4. SMOTE + RandomUnderSampler aplicados solo al conjunto de entrenamiento.\n'
        '5. Brier Score: mide calibración; 0 = predicciones perfectamente calibradas, '
        '0.25 = predicción aleatoria.\n'
        '6. PSM: Nearest Neighbor matching con caliper = 0.2 SD del propensity score.\n'
        '7. Bootstrap stability: AUC calculado sobre 20 remuestras OOB.\n'
    )

    # Guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    print(f"\nReporte guardado: {output_path}")
    return output_path


# =============================================================================
# TABLA COMPARATIVA RÁPIDA (solo Excel, sin Word)
# =============================================================================

def generate_quick_summary_excel(output_path):
    """
    Genera resumen rápido en Excel con demografía + ML por condición.
    No requiere python-docx.
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        # Hoja 1: conteo de sujetos
        counts = []
        full_data_path = os.path.join(
            RESULTS_DIR, f'Data_complete_{DATA_TYPE}_{SPACE.lower()}.feather'
        )
        if os.path.exists(full_data_path):
            data = pd.read_feather(full_data_path)
            subj = data.drop_duplicates('subject')
            row = {'dataset': 'all_data', 'source': 'no_matching'}
            for g in subj['group'].unique():
                row[g] = (subj['group'] == g).sum()
            row['total'] = len(subj)
            counts.append(row)

        for ratio in ['1to1', '2to1', '5to1']:
            psm_path = os.path.join(
                RESULTS_DIR, 'PSM_datasets',
                f'Data_matched_{DATA_TYPE}_{SPACE.lower()}_{GROUP1}_{ratio}.feather'
            )
            if os.path.exists(psm_path):
                data = pd.read_feather(psm_path)
                subj = data.drop_duplicates('subject')
                row  = {'dataset': f'PSM_{ratio}', 'source': psm_path}
                for g in subj['group'].unique():
                    row[g] = (subj['group'] == g).sum()
                row['total'] = len(subj)
                counts.append(row)

        pd.DataFrame(counts).to_excel(writer, sheet_name='Subject_counts', index=False)

        # Hoja 2: ML results (si existe)
        comparison_path = os.path.join(ML_V2_DIR, 'comparison_all_conditions.xlsx')
        if os.path.exists(comparison_path):
            df_comp = pd.read_excel(comparison_path)
            df_comp.to_excel(writer, sheet_name='ML_results', index=False)

            # Hoja 3: Mejor por condición
            best = (df_comp.sort_values('auc_mean', ascending=False)
                    .groupby('condition').first().reset_index())
            best.to_excel(writer, sheet_name='Best_per_condition', index=False)

    print(f"Resumen Excel guardado: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    os.makedirs(ML_V2_DIR, exist_ok=True)

    # 1. Resumen Excel rápido (no requiere python-docx)
    quick_path = os.path.join(ML_V2_DIR, 'quick_summary.xlsx')
    generate_quick_summary_excel(quick_path)

    # 2. Reporte Word completo (requiere python-docx)
    word_path = os.path.join(ML_V2_DIR, f'report_{GROUP1}_vs_{GROUP2}.docx')
    try:
        generate_report(word_path)
    except ImportError as e:
        print(f"\nNo se pudo generar el Word: {e}")
        print("Instala: pip install python-docx statsmodels")
        print(f"El resumen Excel sí fue generado: {quick_path}")

    print("\nGeneración de reportes completada.")
