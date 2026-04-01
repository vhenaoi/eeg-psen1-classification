"""
shap_analysis.py — Módulo de Explicabilidad SHAP
=================================================
Genera visualizaciones SHAP para los modelos finales entrenados.

Uso independiente:
    python shap_analysis.py

O llamado desde 3_train_ml_v2.py después del entrenamiento.
Requiere: pip install shap
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_PATH   = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES'
ML_V2_DIR   = os.path.join(BASE_PATH, 'Resultados', 'graphics', 'ML_v2', 'PSEN1_vs_Control')
N_TOP_FEATS = 15   # features top a mostrar en beeswarm y barras

# Condiciones a procesar (subcarpetas de ML_V2_DIR)
CONDITIONS = ['no_matching', 'psm_1to1', 'psm_2to1', 'psm_5to1']

# =============================================================================
# FUNCIONES CORE
# =============================================================================

_EXCLUDE_META = {
    'subject', 'group', 'Task', 'ses', 'mmse', 'moca',
    'group_sl', 'age_sl', 'SITE_sl',
    'group_coh', 'age_coh', 'SITE_coh',
    'group_ent', 'age_ent', 'SITE_ent',
    'group_cross', 'age_cross', 'SITE_cross',
    'SITE', 'sex', 'age', 'education',
}


def _prepare_X_for_shap(model_info, X_raw):
    """
    Aplica imputer + scaler + feature_selector del modelo guardado
    para obtener X en el espacio de features del modelo.
    """
    X = X_raw.copy().astype(np.float64)
    X = model_info['imputer'].transform(np.nan_to_num(X, nan=0.0))
    X = model_info['scaler'].transform(X)
    X = model_info['feature_selector'].transform(X)
    return X


def compute_shap_values(model_info, X_transformed):
    """
    Calcula SHAP values según el tipo de clasificador.
    - RF/tree:  TreeExplainer (rápido)
    - LR:       LinearExplainer
    - SVM:      KernelExplainer (lento, usa subsample)
    Retorna shap_values (array 2D: n_samples × n_features) para la clase positiva.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Instala shap: pip install shap")

    clf       = model_info['model']
    clf_name  = model_info.get('classifier_name', '')

    if 'RF' in clf_name or hasattr(clf, 'estimators_'):
        explainer   = shap.TreeExplainer(clf)
        shap_output = explainer.shap_values(X_transformed)
        # shap_values puede ser lista [clase0, clase1] o array 3D
        if isinstance(shap_output, list):
            shap_vals = shap_output[1]          # clase positiva
        elif shap_output.ndim == 3:
            shap_vals = shap_output[:, :, 1]    # clase positiva
        else:
            shap_vals = shap_output

    elif 'LR' in clf_name or hasattr(clf, 'coef_'):
        explainer = shap.LinearExplainer(clf, X_transformed, feature_perturbation='interventional')
        shap_vals = explainer.shap_values(X_transformed)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

    else:  # SVM u otro
        # KernelExplainer es lento → usar subsample para background
        sample_size = min(50, len(X_transformed))
        background  = shap.sample(X_transformed, sample_size)
        explainer   = shap.KernelExplainer(clf.predict_proba, background)
        shap_vals_all = explainer.shap_values(X_transformed[:min(100, len(X_transformed))],
                                               nsamples=100, silent=True)
        shap_vals = shap_vals_all[1] if isinstance(shap_vals_all, list) else shap_vals_all

    return shap_vals.astype(np.float32)


def plot_shap_beeswarm(shap_vals, X, feature_names, output_dir, title='', n_top=15):
    """
    Beeswarm plot: cada punto = un sujeto, color = valor de la feature.
    """
    try:
        import shap
    except ImportError:
        return

    # Ordenar por mean |SHAP|
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:n_top]

    shap_top  = shap_vals[:, top_idx]
    X_top     = X[:, top_idx]
    feat_top  = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, n_top * 0.45)))
    shap.summary_plot(
        shap_top, X_top,
        feature_names=feat_top,
        max_display=n_top,
        show=False,
        plot_size=None,
    )
    plt.title(title, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_beeswarm.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Beeswarm guardado: {output_dir}/shap_beeswarm.png")


def plot_shap_bar(shap_vals, feature_names, output_dir, title='', n_top=20):
    """
    Barplot de importancia media |SHAP| con nombre de feature.
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:n_top]
    top_vals = mean_abs[top_idx]
    top_feat = [feature_names[i] for i in top_idx]

    colors = plt.cm.RdBu_r(np.linspace(0.15, 0.85, n_top))[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, n_top * 0.4)))
    bars = ax.barh(range(n_top), top_vals[::-1], color=colors)
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(top_feat[::-1], fontsize=9)
    ax.set_xlabel('Mean |SHAP value|  (impacto promedio)')
    ax.set_title(title or 'Feature Importance — SHAP', fontsize=11)

    # Anotar valores
    for i, (bar, val) in enumerate(zip(bars, top_vals[::-1])):
        ax.text(val + max(top_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_bar.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Bar plot guardado: {output_dir}/shap_bar.png")


def save_shap_table(shap_vals, feature_names, output_dir):
    """Guarda tabla de mean |SHAP| para todas las features."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_raw = shap_vals.mean(axis=0)          # dirección del efecto
    std_abs  = np.abs(shap_vals).std(axis=0)

    df = pd.DataFrame({
        'feature':        feature_names,
        'mean_abs_shap':  mean_abs,
        'mean_shap':      mean_raw,
        'std_shap':       std_abs,
        'rank':           np.argsort(np.argsort(-mean_abs)) + 1,
    }).sort_values('mean_abs_shap', ascending=False)

    df.to_excel(os.path.join(output_dir, 'shap_importance_table.xlsx'), index=False)
    print(f"  Tabla SHAP guardada: {output_dir}/shap_importance_table.xlsx")
    return df


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def run_shap_for_condition(condition_dir, data_path, group1='PSEN1', group2='Control',
                            n_top=N_TOP_FEATS):
    """
    Ejecuta el análisis SHAP completo para una condición.
    condition_dir : carpeta con final_model.pkl
    data_path     : feather con los datos de esa condición
    """
    model_path = os.path.join(condition_dir, 'final_model.pkl')
    if not os.path.exists(model_path):
        print(f"  [SKIP] Modelo no encontrado: {model_path}")
        return

    print(f"\n  SHAP: {os.path.basename(condition_dir)}")
    model_info = joblib.load(model_path)

    # Cargar datos
    data = pd.read_feather(data_path)
    g2   = group2 if isinstance(group2, list) else [group2]
    data = data[data['group'].isin([group1] + g2)].copy()

    # Verificar que las feature_names del modelo coincidan
    feature_names = model_info.get('feature_names', [])
    if not feature_names:
        print("  [SKIP] Modelo sin feature_names")
        return

    # Preparar X: usar all_feature_names → subset → pipeline del modelo
    all_feat = model_info.get('all_feature_names', feature_names)

    # Agregar a sujeto level
    exc = _EXCLUDE_META
    raw_cols = [c for c in data.columns if c not in exc | {'subject', 'group', 'SITE', 'sex', 'age'}]

    if 'SITE' in data.columns:
        dummies = pd.get_dummies(data['SITE'], prefix='SITE', drop_first=True, dtype=float)
        data = pd.concat([data, dummies], axis=1)

    data_agg = data.groupby('subject')[raw_cols].mean().reset_index()
    y = (data.groupby('subject')['group'].first().values == group1).astype(int)

    # Seleccionar solo las features que el modelo espera (all_feature_names)
    available = [f for f in all_feat if f in data_agg.columns]
    if not available:
        print("  [SKIP] No hay features disponibles que coincidan con el modelo")
        return

    X_raw = data_agg[available].values.astype(np.float64)

    # Aplicar preprocessing del modelo
    try:
        X_trans = _prepare_X_for_shap(model_info, X_raw)
    except Exception as e:
        print(f"  [ERROR] Preprocessing para SHAP falló: {e}")
        return

    # Computar SHAP values
    try:
        shap_vals = compute_shap_values(model_info, X_trans)
    except Exception as e:
        print(f"  [ERROR] SHAP falló: {e}")
        return

    shap_dir = os.path.join(condition_dir, 'shap')
    os.makedirs(shap_dir, exist_ok=True)

    clf_name = model_info.get('classifier_name', '')
    title    = f"SHAP — {clf_name} | {os.path.basename(condition_dir)}"

    plot_shap_beeswarm(shap_vals, X_trans, feature_names, shap_dir, title=title, n_top=n_top)
    plot_shap_bar(shap_vals, feature_names, shap_dir, title=title, n_top=n_top)
    save_shap_table(shap_vals, feature_names, shap_dir)

    # Guardar SHAP values raw
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    shap_df.to_csv(os.path.join(shap_dir, 'shap_values_raw.csv'), index=False)

    print(f"  SHAP completo → {shap_dir}")
    return shap_vals, feature_names


def generate_combined_shap_comparison(ml_v2_dir, conditions, data_paths,
                                      group1='PSEN1', group2='Control', n_top=10):
    """
    Genera figura comparativa con un panel SHAP por condición (barra horizontal).
    Útil para el paper: mostrar consistencia de features a través de condiciones.
    """
    try:
        import shap
    except ImportError:
        print("shap no instalado. pip install shap")
        return

    results = {}
    for cond in conditions:
        cond_dir  = os.path.join(ml_v2_dir, cond)
        data_path = data_paths.get(cond)
        if data_path and os.path.exists(cond_dir):
            out = run_shap_for_condition(cond_dir, data_path, group1, group2)
            if out:
                shap_vals, feat_names = out
                mean_abs = np.abs(shap_vals).mean(axis=0)
                results[cond] = {'mean_abs': mean_abs, 'features': feat_names}

    if not results:
        print("Sin resultados para figura comparativa")
        return

    # Figura multi-panel
    n_conds = len(results)
    fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 7), sharey=False)
    if n_conds == 1:
        axes = [axes]

    all_top_feats = set()
    for res in results.values():
        top_idx = np.argsort(res['mean_abs'])[::-1][:n_top]
        all_top_feats.update(np.array(res['features'])[top_idx].tolist())

    for ax, (cond, res) in zip(axes, results.items()):
        feat_arr = np.array(res['features'])
        mean_abs = res['mean_abs']
        top_idx  = np.argsort(mean_abs)[::-1][:n_top]
        ax.barh(range(n_top), mean_abs[top_idx[::-1]],
                color='steelblue', edgecolor='white')
        ax.set_yticks(range(n_top))
        ax.set_yticklabels(feat_arr[top_idx[::-1]], fontsize=8)
        ax.set_title(cond, fontsize=10)
        ax.set_xlabel('Mean |SHAP|')
        if ax != axes[0]:
            ax.set_yticklabels([])

    plt.suptitle(f'Feature Importance SHAP — {group1} vs {group2}', fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(ml_v2_dir, 'shap_comparison_all_conditions.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nFigura comparativa SHAP guardada: {out_path}")


# =============================================================================
# MAIN (uso standalone)
# =============================================================================

if __name__ == '__main__':

    GROUP1 = 'PSEN1'
    GROUP2 = 'Control'
    DATA_TYPE = 'ce'
    SPACE     = 'roi'

    results_base = os.path.join(BASE_PATH, 'Resultados')
    psm_dir      = os.path.join(results_base, 'PSM_datasets')

    # Mapeo condición → archivo de datos
    data_paths = {
        'no_matching': os.path.join(results_base, f'Data_complete_{DATA_TYPE}_{SPACE}.feather'),
        'psm_1to1':    os.path.join(psm_dir, f'Data_matched_{DATA_TYPE}_{SPACE}_{GROUP1}_1to1.feather'),
        'psm_2to1':    os.path.join(psm_dir, f'Data_matched_{DATA_TYPE}_{SPACE}_{GROUP1}_2to1.feather'),
        'psm_5to1':    os.path.join(psm_dir, f'Data_matched_{DATA_TYPE}_{SPACE}_{GROUP1}_5to1.feather'),
    }

    # Ejecutar SHAP para cada condición disponible
    for cond in CONDITIONS:
        data_path = data_paths.get(cond)
        cond_dir  = os.path.join(ML_V2_DIR, cond)
        if data_path and os.path.exists(data_path) and os.path.exists(cond_dir):
            run_shap_for_condition(cond_dir, data_path, GROUP1, GROUP2)

    # Figura comparativa
    generate_combined_shap_comparison(
        ML_V2_DIR, CONDITIONS, data_paths, GROUP1, GROUP2
    )

    print("\nAnálisis SHAP completado.")
