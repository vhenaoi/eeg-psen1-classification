"""
robustness_analysis.py — Análisis de Robustez del Modelo ML
=============================================================
Implementa los 4 análisis de robustez de V2_DLB adaptados a PORTABLES:

1. Noise Injection (σ = 0–50% de la SD de cada feature)
   → Mide degradación del AUC ante perturbaciones (simula variabilidad inter-sitio)

2. Hyperparameter Sensitivity (factor × {0.25, 0.5, 1.0, 2.0, 4.0})
   → Mide sensibilidad del AUC al parámetro principal de cada clasificador

3. Calibration Quality
   → Brier Score + ECE (Expected Calibration Error, 10 bins)
   → Medido via out-of-fold para evitar sesgo optimista

4. Bootstrap Stability
   → AUC sobre N_BOOTSTRAP remuestras OOB
   → Mide consistencia ante resampling

Uso:
    python robustness_analysis.py

O importado desde 3_train_ml_v2.py:
    from robustness_analysis import run_full_robustness
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import roc_auc_score, brier_score_loss
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_PATH  = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES'
ML_V2_DIR  = os.path.join(BASE_PATH, 'Resultados', 'graphics', 'ML_v2', 'PSEN1_vs_Control')
GROUP1     = 'PSEN1'
GROUP2     = 'Control'
DATA_TYPE  = 'ce'
SPACE      = 'roi'

RANDOM_STATE = 42

# Noise injection
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.50]   # fracción de la SD de cada feature

# Hyperparameter sensitivity — factores multiplicadores
SENSITIVITY_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]

# Calibración
N_CALIBRATION_FOLDS = 5
N_BINS_ECE          = 10

# Bootstrap
N_BOOTSTRAP = 20


# =============================================================================
# 1. NOISE INJECTION
# =============================================================================

def _add_gaussian_noise(X, sigma_fraction, rng):
    """Agrega ruido gaussiano: σ = sigma_fraction × SD(feature)."""
    if sigma_fraction == 0.0:
        return X.copy()
    feature_sd = X.std(axis=0, keepdims=True).clip(min=1e-8)
    noise = rng.normal(0, sigma_fraction * feature_sd, size=X.shape)
    return X + noise


def noise_injection_analysis(X, y, model_info, rng=None, noise_levels=NOISE_LEVELS):
    """
    Evalúa la degradación del AUC al añadir ruido gaussiano creciente.

    Parámetros
    ----------
    X           : array pre-procesado (ya filtrado, NO imputed/scaled — usamos el modelo)
    y           : etiquetas
    model_info  : dict con 'model', 'imputer', 'scaler', 'feature_selector'
    rng         : np.random.RandomState
    noise_levels: lista de σ como fracción de la SD

    Retorna
    -------
    dict: {sigma: auc_value}
    """
    if rng is None:
        rng = np.random.RandomState(RANDOM_STATE)

    results = {}
    clf    = model_info['model']
    imp    = model_info['imputer']
    sc     = model_info['scaler']
    fs     = model_info['feature_selector']

    for sigma in noise_levels:
        X_noisy = _add_gaussian_noise(X, sigma, rng)
        try:
            X_proc  = imp.transform(np.nan_to_num(X_noisy, nan=0.0))
            X_proc  = sc.transform(X_proc)
            X_proc  = fs.transform(X_proc)
            y_proba = clf.predict_proba(X_proc)[:, 1]
            auc     = roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else np.nan
        except Exception:
            auc = np.nan
        results[sigma] = float(auc)

    return results


def plot_noise_injection(results_by_condition, output_dir):
    """
    Líneas AUC vs nivel de ruido, una línea por condición.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_condition)))

    for (cond, noise_dict), color in zip(results_by_condition.items(), colors):
        sigmas = sorted(noise_dict.keys())
        aucs   = [noise_dict[s] for s in sigmas]
        ax.plot([s * 100 for s in sigmas], aucs, marker='o',
                label=cond, color=color, lw=2)

    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Azar (AUC=0.5)')
    ax.set_xlabel('Nivel de ruido (% SD de la feature)')
    ax.set_ylabel('AUC')
    ax.set_title('Robustez: Noise Injection')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_noise_injection.png'), dpi=200)
    plt.close()
    print(f"  Guardado: robustness_noise_injection.png")


# =============================================================================
# 2. HYPERPARAMETER SENSITIVITY
# =============================================================================

_MAIN_PARAM = {
    'RF':  ('n_estimators', 200),   # valor base
    'SVM': ('C',            1.0),
    'LR':  ('C',            1.0),
}

_CLF_FACTORIES = {
    'RF':  lambda p, v: RandomForestClassifier(n_estimators=int(max(1, v)), n_jobs=1,
                                                class_weight='balanced',
                                                random_state=RANDOM_STATE),
    'SVM': lambda p, v: SVC(C=v, probability=True, class_weight='balanced',
                             random_state=RANDOM_STATE),
    'LR':  lambda p, v: LogisticRegression(C=v, class_weight='balanced',
                                            max_iter=1000, random_state=RANDOM_STATE),
}


def hyperparameter_sensitivity(X_fs, y, clf_name, factors=SENSITIVITY_FACTORS,
                                n_cv=3):
    """
    Varía el parámetro principal del clasificador por factores multiplicadores.
    Mide AUC (3-fold CV) para cada factor.

    Retorna dict: {factor: auc_value}
    """
    param_name, base_val = _MAIN_PARAM.get(clf_name, ('C', 1.0))
    factory              = _CLF_FACTORIES.get(clf_name)
    if factory is None:
        return {}

    cv      = StratifiedKFold(n_splits=min(n_cv, int(np.bincount(y).min())),
                               shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for factor in factors:
        val = base_val * factor
        clf = factory(param_name, val)
        try:
            y_proba = cross_val_predict(clf, X_fs, y, cv=cv, method='predict_proba')[:, 1]
            auc     = roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else np.nan
        except Exception:
            auc = np.nan
        results[factor] = float(auc)

    return results


def plot_hyperparameter_sensitivity(results_by_clf, output_dir, condition_name=''):
    """
    Una línea por clasificador: AUC vs factor multiplicador del parámetro principal.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for clf_name, factor_dict in results_by_clf.items():
        if not factor_dict:
            continue
        factors = sorted(factor_dict.keys())
        aucs    = [factor_dict[f] for f in factors]
        param   = _MAIN_PARAM.get(clf_name, ('?', None))[0]
        ax.plot(factors, aucs, marker='s', lw=2, label=f'{clf_name} ({param})')

    ax.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Factor multiplicador del parámetro principal')
    ax.set_ylabel('AUC (3-fold CV)')
    ax.set_title(f'Robustez: Sensibilidad a Hiperparámetros — {condition_name}')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_hyperparam_sensitivity.png'), dpi=200)
    plt.close()
    print(f"  Guardado: robustness_hyperparam_sensitivity.png")


# =============================================================================
# 3. CALIBRATION QUALITY (Brier Score + ECE)
# =============================================================================

def expected_calibration_error(y_true, y_proba, n_bins=N_BINS_ECE):
    """
    Expected Calibration Error (ECE): promedio ponderado de |confianza - exactitud|
    dentro de n_bins intervalos de probabilidad.
    ECE = 0 → calibración perfecta; ECE = 1 → peor caso.
    """
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        acc    = y_true[mask].mean()
        conf   = y_proba[mask].mean()
        ece   += (mask.sum() / n) * abs(conf - acc)

    return float(ece)


def calibration_analysis(X_fs, y, clf, n_folds=N_CALIBRATION_FOLDS):
    """
    Calcula Brier Score + ECE via out-of-fold predictions.
    Retorna dict con 'brier', 'ece', 'y_proba_oof', 'y_true_oof'.
    """
    cv      = StratifiedKFold(n_splits=min(n_folds, int(np.bincount(y).min())),
                               shuffle=True, random_state=RANDOM_STATE)
    try:
        y_proba_oof = cross_val_predict(clf, X_fs, y, cv=cv, method='predict_proba')[:, 1]
    except Exception:
        return {'brier': np.nan, 'ece': np.nan}

    return {
        'brier':       float(brier_score_loss(y, y_proba_oof)),
        'ece':         expected_calibration_error(y, y_proba_oof),
        'y_proba_oof': y_proba_oof,
        'y_true_oof':  y,
    }


def plot_calibration_curve(cal_results, output_dir, condition_name=''):
    """
    Reliability diagram: probabilidad media predicha vs fracción real de positivos.
    """
    n_cols = min(3, len(cal_results))
    n_rows = (len(cal_results) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, (clf_name, res) in enumerate(cal_results.items()):
        ax    = axes[idx // n_cols][idx % n_cols]
        yp    = res.get('y_proba_oof')
        yt    = res.get('y_true_oof')

        if yp is None or len(np.unique(yt)) < 2:
            ax.set_visible(False)
            continue

        # Reliability diagram
        bins   = np.linspace(0, 1, N_BINS_ECE + 1)
        mean_p, frac_p = [], []
        for i in range(N_BINS_ECE):
            mask = (yp >= bins[i]) & (yp < bins[i + 1])
            if mask.sum() == 0:
                continue
            mean_p.append(yp[mask].mean())
            frac_p.append(yt[mask].mean())

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectamente calibrado')
        ax.plot(mean_p, frac_p, marker='o', lw=2, color='steelblue', label='Modelo')
        ax.fill_between(mean_p, frac_p, mean_p, alpha=0.15, color='tomato')
        brier = res.get('brier', np.nan)
        ece   = res.get('ece',   np.nan)
        ax.set_title(f'{clf_name}\nBrier={brier:.3f}  ECE={ece:.3f}', fontsize=9)
        ax.set_xlabel('Probabilidad media predicha')
        ax.set_ylabel('Fracción de positivos reales')
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Ocultar ejes vacíos
    for idx in range(len(cal_results), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.suptitle(f'Calibración — {condition_name}', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_calibration.png'), dpi=200)
    plt.close()
    print(f"  Guardado: robustness_calibration.png")


# =============================================================================
# 4. BOOTSTRAP STABILITY (re-exportado desde 3_train_ml_v2 con mejoras)
# =============================================================================

def bootstrap_stability_full(X, y, model_info, n_bootstrap=N_BOOTSTRAP):
    """
    Versión mejorada: además del AUC OOB, calcula la desviación estándar
    de la probabilidad predicha por sujeto (σ_i) a través de los resamples.
    σ_i pequeño → predicción estable para ese sujeto.

    Retorna
    -------
    dict con:
      'auc_mean', 'auc_std'
      'subject_sigma_mean', 'subject_sigma_median'  — estabilidad por sujeto
      'auc_values'  — lista de AUC por resample
    """
    clf    = model_info['model']
    imp    = model_info['imputer']
    sc     = model_info['scaler']
    fs     = model_info['feature_selector']

    rng        = np.random.RandomState(RANDOM_STATE)
    auc_values = []
    all_probas = np.full((len(y), n_bootstrap), np.nan)

    for b in range(n_bootstrap):
        idx_boot = rng.choice(len(y), size=len(y), replace=True)
        oob_mask = np.ones(len(y), dtype=bool)
        oob_mask[idx_boot] = False
        oob_idx  = np.where(oob_mask)[0]

        if len(oob_idx) < 4 or len(np.unique(y[oob_idx])) < 2:
            continue

        try:
            X_b   = imp.transform(np.nan_to_num(X[idx_boot], nan=0.0))
            X_b   = sc.transform(X_b)
            X_b   = fs.transform(X_b)
            X_oob = imp.transform(np.nan_to_num(X[oob_idx], nan=0.0))
            X_oob = sc.transform(X_oob)
            X_oob = fs.transform(X_oob)

            # Re-entrenar con los mismos hiperparámetros
            clf.fit(X_b, y[idx_boot])
            proba = clf.predict_proba(X_oob)[:, 1]
            auc   = roc_auc_score(y[oob_idx], proba)
            auc_values.append(auc)
            all_probas[oob_idx, b] = proba
        except Exception:
            pass

    # Estabilidad por sujeto
    valid_cols   = ~np.all(np.isnan(all_probas), axis=0)
    prob_matrix  = all_probas[:, valid_cols]
    subject_sigmas = np.nanstd(prob_matrix, axis=1)
    subject_sigmas = subject_sigmas[~np.isnan(subject_sigmas)]

    return {
        'auc_mean':            float(np.mean(auc_values)) if auc_values else np.nan,
        'auc_std':             float(np.std(auc_values))  if auc_values else np.nan,
        'auc_values':          auc_values,
        'n_resamples':         len(auc_values),
        'subject_sigma_mean':  float(np.mean(subject_sigmas))   if len(subject_sigmas) > 0 else np.nan,
        'subject_sigma_median':float(np.median(subject_sigmas)) if len(subject_sigmas) > 0 else np.nan,
    }


def plot_bootstrap_distribution(bs_result, output_dir, condition_name=''):
    """Histograma de AUCs bootstrap + estadísticas."""
    auc_vals = bs_result.get('auc_values', [])
    if not auc_vals:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel izquierdo: histograma de AUC
    ax = axes[0]
    ax.hist(auc_vals, bins=min(10, len(auc_vals)), color='steelblue',
            edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(auc_vals), color='red', lw=2,
               label=f'Media={np.mean(auc_vals):.3f}')
    ax.axvline(np.mean(auc_vals) - np.std(auc_vals), color='orange',
               lw=1.5, linestyle='--', label=f'±σ={np.std(auc_vals):.3f}')
    ax.axvline(np.mean(auc_vals) + np.std(auc_vals), color='orange',
               lw=1.5, linestyle='--')
    ax.set_xlabel('AUC (OOB)')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribución Bootstrap AUC\n{condition_name}')
    ax.legend(fontsize=8)

    # Panel derecho: texto con métricas
    ax = axes[1]
    ax.axis('off')
    metrics_text = (
        f"Bootstrap Stability Report\n"
        f"{'─'*30}\n"
        f"N resamples:        {bs_result['n_resamples']}\n"
        f"AUC media:          {bs_result['auc_mean']:.3f}\n"
        f"AUC std:            {bs_result['auc_std']:.3f}\n"
        f"AUC 95% CI:         [{np.percentile(auc_vals, 2.5):.3f} – {np.percentile(auc_vals, 97.5):.3f}]\n"
        f"\nEstabilidad por sujeto:\n"
        f"σ media:            {bs_result['subject_sigma_mean']:.3f}\n"
        f"σ mediana:          {bs_result['subject_sigma_median']:.3f}\n"
        f"\nInterpretación:\n"
        f"  σ < 0.10  → Alta estabilidad ✓\n"
        f"  σ 0.10–0.20 → Estabilidad media\n"
        f"  σ > 0.20  → Baja estabilidad ✗\n"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_bootstrap.png'), dpi=200)
    plt.close()
    print(f"  Guardado: robustness_bootstrap.png")


# =============================================================================
# FUNCIÓN PRINCIPAL: TODOS LOS ANÁLISIS DE ROBUSTEZ
# =============================================================================

def run_full_robustness(X_raw, y, feature_names, model_info, output_dir,
                        condition_name='', save_excel=True):
    """
    Ejecuta los 4 análisis de robustez y genera visualizaciones + tabla Excel.

    Parámetros
    ----------
    X_raw       : array SIN preprocessing (solo filtrado de varianza/correlación)
    y           : etiquetas binarias
    feature_names: nombres de features de X_raw
    model_info  : dict guardado por train_final_model (contiene imputer, scaler, fs, model)
    output_dir  : carpeta donde guardar resultados
    condition_name: nombre de la condición (para títulos)

    Retorna
    -------
    dict con resultados de todos los análisis
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Robustez: {condition_name}")

    results = {}

    # ─────────────────────────────────────────────────────────────────
    # 1. Noise Injection
    # ─────────────────────────────────────────────────────────────────
    print("    1/4 Noise injection...")
    rng    = np.random.RandomState(RANDOM_STATE)
    noise  = noise_injection_analysis(X_raw, y, model_info, rng=rng)
    results['noise'] = noise
    delta_auc = noise.get(0.0, np.nan) - min(v for v in noise.values()
                                              if not np.isnan(v)) \
                if any(not np.isnan(v) for v in noise.values()) else np.nan
    print(f"    AUC(σ=0%) = {noise.get(0.0, np.nan):.3f}  |  ΔAUC = {delta_auc:.3f}")

    # ─────────────────────────────────────────────────────────────────
    # 2. Hyperparameter Sensitivity
    # ─────────────────────────────────────────────────────────────────
    print("    2/4 Hyperparameter sensitivity...")
    # Preparar X en el espacio de features del modelo
    try:
        X_proc = model_info['imputer'].transform(np.nan_to_num(X_raw, nan=0.0))
        X_proc = model_info['scaler'].transform(X_proc)
        X_fs   = model_info['feature_selector'].transform(X_proc)
    except Exception as e:
        print(f"    Preprocessing para sensibilidad falló: {e}")
        X_fs = X_raw

    clf_name = model_info.get('classifier_name', 'RF')
    sens = hyperparameter_sensitivity(X_fs, y, clf_name)
    results['sensitivity'] = {clf_name: sens}

    # Añadir los otros clasificadores si el modelo final es RF
    for other_clf in ['SVM', 'LR']:
        if other_clf != clf_name:
            results['sensitivity'][other_clf] = hyperparameter_sensitivity(
                X_fs, y, other_clf
            )

    plot_hyperparameter_sensitivity(results['sensitivity'], output_dir, condition_name)

    # ─────────────────────────────────────────────────────────────────
    # 3. Calibration (Brier + ECE)
    # ─────────────────────────────────────────────────────────────────
    print("    3/4 Calibration analysis...")
    cal_results = {}
    for clf_name_cal, clf_base, _ in [
        ('RF',  RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                        n_jobs=1, random_state=RANDOM_STATE), None),
        ('SVM', SVC(probability=True, class_weight='balanced',
                    random_state=RANDOM_STATE), None),
        ('LR',  LogisticRegression(class_weight='balanced', max_iter=1000,
                                    random_state=RANDOM_STATE), None),
    ]:
        cal = calibration_analysis(X_fs, y, clf_base)
        cal_results[clf_name_cal] = cal
        print(f"    {clf_name_cal}: Brier={cal['brier']:.3f}  ECE={cal['ece']:.3f}")

    results['calibration'] = cal_results
    plot_calibration_curve(cal_results, output_dir, condition_name)

    # ─────────────────────────────────────────────────────────────────
    # 4. Bootstrap Stability
    # ─────────────────────────────────────────────────────────────────
    print("    4/4 Bootstrap stability...")
    bs = bootstrap_stability_full(X_raw, y, model_info, n_bootstrap=N_BOOTSTRAP)
    results['bootstrap'] = bs
    print(f"    AUC boot = {bs['auc_mean']:.3f} ± {bs['auc_std']:.3f}  "
          f"| σ_sujeto med = {bs['subject_sigma_median']:.3f}")
    plot_bootstrap_distribution(bs, output_dir, condition_name)

    # ─────────────────────────────────────────────────────────────────
    # Guardar tabla de resumen
    # ─────────────────────────────────────────────────────────────────
    if save_excel:
        _save_robustness_excel(results, output_dir, condition_name)

    return results


# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================

def _save_robustness_excel(results, output_dir, condition_name):
    """Guarda todas las métricas de robustez en un Excel con múltiples hojas."""
    path = os.path.join(output_dir, 'robustness_summary.xlsx')

    with pd.ExcelWriter(path, engine='openpyxl') as writer:

        # Hoja 1: Noise injection
        if 'noise' in results:
            noise_df = pd.DataFrame([
                {'sigma_pct': k * 100, 'AUC': v}
                for k, v in sorted(results['noise'].items())
            ])
            auc0 = results['noise'].get(0.0, np.nan)
            noise_df['delta_AUC'] = auc0 - noise_df['AUC']
            noise_df.to_excel(writer, sheet_name='Noise_Injection', index=False)

        # Hoja 2: Hyperparameter sensitivity
        if 'sensitivity' in results:
            rows = []
            for clf_n, fdict in results['sensitivity'].items():
                param = _MAIN_PARAM.get(clf_n, ('?', None))[0]
                for factor, auc in sorted(fdict.items()):
                    rows.append({'classifier': clf_n, 'parameter': param,
                                 'factor': factor, 'AUC': auc})
            pd.DataFrame(rows).to_excel(writer, sheet_name='Hyperparam_Sensitivity', index=False)

        # Hoja 3: Calibración
        if 'calibration' in results:
            rows = []
            for clf_n, cal in results['calibration'].items():
                rows.append({
                    'classifier': clf_n,
                    'brier_score': cal.get('brier', np.nan),
                    'ece':         cal.get('ece',   np.nan),
                    'n_folds':     N_CALIBRATION_FOLDS,
                })
            pd.DataFrame(rows).to_excel(writer, sheet_name='Calibration', index=False)

        # Hoja 4: Bootstrap
        if 'bootstrap' in results:
            bs = results['bootstrap']
            bs_df = pd.DataFrame([{
                'condition':            condition_name,
                'auc_mean':             bs.get('auc_mean', np.nan),
                'auc_std':              bs.get('auc_std',  np.nan),
                'auc_95ci_low':         np.percentile(bs.get('auc_values', [np.nan]), 2.5),
                'auc_95ci_high':        np.percentile(bs.get('auc_values', [np.nan]), 97.5),
                'n_resamples':          bs.get('n_resamples', 0),
                'subject_sigma_mean':   bs.get('subject_sigma_mean',   np.nan),
                'subject_sigma_median': bs.get('subject_sigma_median', np.nan),
            }])
            bs_df.to_excel(writer, sheet_name='Bootstrap', index=False)

    print(f"  Excel robustez guardado: {path}")


# =============================================================================
# FIGURA COMPARATIVA MULTI-CONDICIÓN
# =============================================================================

def generate_robustness_comparison_figure(all_robustness, output_dir):
    """
    Figura de 4 paneles comparando robustez entre condiciones.
    """
    fig = plt.figure(figsize=(14, 10))
    conditions = list(all_robustness.keys())
    colors     = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

    # Panel 1: Noise injection
    ax1 = fig.add_subplot(2, 2, 1)
    for (cond, res), c in zip(all_robustness.items(), colors):
        if 'noise' not in res:
            continue
        sigmas = sorted(res['noise'].keys())
        aucs   = [res['noise'][s] for s in sigmas]
        ax1.plot([s * 100 for s in sigmas], aucs, marker='o', lw=2,
                 label=cond, color=c)
    ax1.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax1.set_xlabel('Ruido (% SD)')
    ax1.set_ylabel('AUC')
    ax1.set_title('Noise Injection')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Hyperparameter sensitivity (solo clasificador principal)
    ax2 = fig.add_subplot(2, 2, 2)
    for (cond, res), c in zip(all_robustness.items(), colors):
        if 'sensitivity' not in res:
            continue
        # Tomar el primer clasificador disponible
        for clf_n, fdict in res['sensitivity'].items():
            if not fdict:
                continue
            factors = sorted(fdict.keys())
            aucs    = [fdict[f] for f in factors]
            ax2.plot(factors, aucs, marker='s', lw=2, label=f'{cond}|{clf_n}', color=c)
            break
    ax2.set_xscale('log')
    ax2.set_xlabel('Factor hiperparámetro')
    ax2.set_ylabel('AUC (3-fold CV)')
    ax2.set_title('Sensibilidad Hiperparámetros')
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.3, which='both')

    # Panel 3: Calibración (Brier + ECE)
    ax3 = fig.add_subplot(2, 2, 3)
    clf_names = ['RF', 'SVM', 'LR']
    x         = np.arange(len(conditions))
    w         = 0.25
    for i, clf_n in enumerate(clf_names):
        briers = []
        for cond in conditions:
            cal = all_robustness.get(cond, {}).get('calibration', {}).get(clf_n, {})
            briers.append(cal.get('brier', np.nan))
        ax3.bar(x + i * w, briers, width=w, label=clf_n, alpha=0.8)
    ax3.axhline(0.25, color='red', lw=1, linestyle='--', label='Azar')
    ax3.set_xticks(x + w)
    ax3.set_xticklabels(conditions, rotation=20, ha='right', fontsize=7)
    ax3.set_ylabel('Brier Score')
    ax3.set_title('Calibración (Brier Score)')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Bootstrap AUC
    ax4 = fig.add_subplot(2, 2, 4)
    boot_means = []
    boot_stds  = []
    labels     = []
    for cond in conditions:
        bs = all_robustness.get(cond, {}).get('bootstrap', {})
        boot_means.append(bs.get('auc_mean', np.nan))
        boot_stds.append(bs.get('auc_std',  np.nan))
        labels.append(cond)
    ax4.bar(labels, boot_means, yerr=boot_stds, color='steelblue',
            alpha=0.8, capsize=5, edgecolor='white')
    ax4.axhline(0.5, color='gray', linestyle='--', lw=1)
    ax4.set_ylabel('AUC (Bootstrap OOB)')
    ax4.set_title('Estabilidad Bootstrap')
    ax4.set_ylim(0, 1.05)
    ax4.tick_params(axis='x', rotation=20, labelsize=7)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Análisis de Robustez — {GROUP1} vs {GROUP2}', fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'robustness_all_conditions.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nFigura comparativa robustez guardada: {path}")


# =============================================================================
# MAIN (uso standalone)
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # Importar funciones de preparación de datos
    # Replicamos la lógica mínima para cargar X, y y el modelo
    def _load_condition(feather_path, model_path, g1=GROUP1, g2=GROUP2):
        """Carga datos + modelo para una condición."""
        data = pd.read_feather(feather_path)
        g2l  = [g2] if isinstance(g2, str) else g2
        data = data[data['group'].isin([g1] + g2l)].copy()
        data['group'] = data['group'].apply(lambda x: g2l[0] if x in g2l else x)

        exc  = {'subject', 'group', 'Task', 'ses', 'mmse', 'moca',
                'group_sl', 'age_sl', 'SITE_sl', 'group_coh', 'age_coh', 'SITE_coh',
                'group_ent', 'age_ent', 'SITE_ent', 'group_cross', 'age_cross', 'SITE_cross',
                'SITE', 'sex', 'age', 'education'}
        feat_cols = [c for c in data.columns if c not in exc]

        if 'SITE' in data.columns:
            dummies = pd.get_dummies(data['SITE'], prefix='SITE', drop_first=True, dtype=float)
            data    = pd.concat([data, dummies], axis=1)
            feat_cols += list(dummies.columns)

        data_agg = data.groupby('subject')[feat_cols].mean().reset_index()
        grp      = data.groupby('subject')['group'].first()
        data_agg['group'] = grp.values

        y            = (data_agg['group'] == g1).astype(int).values
        X_raw        = data_agg[[c for c in feat_cols if c in data_agg.columns]].values.astype(np.float64)
        model_info   = joblib.load(model_path)

        return X_raw, y, model_info

    # ── Ejecutar por condición ──────────────────────────────────────────────
    results_base = os.path.join(BASE_PATH, 'Resultados')
    psm_dir      = os.path.join(results_base, 'PSM_datasets')

    condition_files = {
        'no_matching': os.path.join(results_base, f'Data_complete_{DATA_TYPE}_{SPACE}.feather'),
        'psm_1to1':    os.path.join(psm_dir, f'Data_matched_{DATA_TYPE}_{SPACE}_{GROUP1}_1to1.feather'),
        'psm_2to1':    os.path.join(psm_dir, f'Data_matched_{DATA_TYPE}_{SPACE}_{GROUP1}_2to1.feather'),
        'psm_5to1':    os.path.join(psm_dir, f'Data_matched_{DATA_TYPE}_{SPACE}_{GROUP1}_5to1.feather'),
    }

    all_robustness = {}

    for cond, data_path in condition_files.items():
        cond_dir   = os.path.join(ML_V2_DIR, cond)
        model_path = os.path.join(cond_dir, 'final_model.pkl')
        rob_dir    = os.path.join(cond_dir, 'robustness')

        if not os.path.exists(data_path):
            print(f"[SKIP] Datos no encontrados: {data_path}")
            continue
        if not os.path.exists(model_path):
            print(f"[SKIP] Modelo no encontrado (ejecutar 3_train_ml_v2.py primero): {model_path}")
            continue

        X_raw, y, model_info = _load_condition(data_path, model_path)
        feat_names = model_info.get('all_feature_names', [])

        rob_result = run_full_robustness(
            X_raw, y, feat_names, model_info, rob_dir, condition_name=cond
        )
        all_robustness[cond] = rob_result

    # Figura comparativa multi-condición
    if all_robustness:
        generate_robustness_comparison_figure(
            all_robustness,
            os.path.join(ML_V2_DIR)
        )

    # Guardar Noise data para figura multi-condición con una sola llamada
    if len(all_robustness) > 1:
        noise_only = {c: r for c, r in all_robustness.items() if 'noise' in r}
        if noise_only:
            plot_noise_injection(
                {c: r['noise'] for c, r in noise_only.items()},
                ML_V2_DIR
            )

    print("\nAnálisis de robustez completado.")
