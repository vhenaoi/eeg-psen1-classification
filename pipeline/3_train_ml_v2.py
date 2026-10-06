"""
3_train_ml_v2.py — ML Training Pipeline (Version 2)
====================================================
Mejoras sobre v1 (basadas en V2_DLB pipeline):

1. Nested cross-validation (outer 10-fold, inner 5-fold) → estimaciones sin sesgo optimista
2. Múltiples clasificadores: Random Forest, SVM, Logistic Regression
3. KNN Imputation DENTRO de los folds (sin data leakage)
4. SMOTE + undersampling DENTRO de los folds
5. 3 métodos de feature selection: SelectKBest, RFE (+ voto mayoritario entre folds)
6. Métricas de calibración: Brier Score
7. Confusion matrix agregada sobre todos los folds de CV
8. Análisis de bootstrap para estabilidad
9. Condiciones: no_matching + psm_1to1 + psm_2to1 + psm_5to1
10. Opcional: residualización de covariables (age, SITE) dentro de folds
11. Logging y checkpointing por condición
12. Tabla de comparación global entre condiciones en Excel
"""

import os
import warnings
import logging
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE, VarianceThreshold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, accuracy_score,
    confusion_matrix, roc_curve, brier_score_loss
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone
from sklearn.calibration import calibration_curve

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import sage
    SAGE_AVAILABLE = True
except ImportError:
    SAGE_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

from config import BASE_PATH
DATA_TYPE       = 'ce'
SPACE           = 'roi'
GROUP1          = 'PSEN1'    # clase positiva
GROUP2          = 'Control'  # clase negativa (puede ser lista: ['Control', 'Relative'])

# CV
N_OUTER_FOLDS   = 10
N_INNER_FOLDS   = 5
N_ITER_SEARCH   = 50        # reducir para velocidad; 100+ para publicación

# Feature selection
# k se determina dinámicamente por get_k_candidates() según N y features disponibles.
# Regla: k ≤ N_train/10  (evita overfitting) y k ≤ n_features.
# El inner CV elige el k óptimo dentro de estos candidatos.
K_CANDIDATES   = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]  # valores a explorar

# Ratio máximo features/muestras por tipo de clasificador
# LR: regla EPV (N//10); SVM y RF toleran ratios más altos por su regularización interna
# XGB: regularización propia (colsample_bytree, gamma, reg_alpha) → mismo ratio que SVM
_K_MAX_RATIO = {'RF': 4, 'SVM': 6, 'LR': 10, 'XGB': 6}
VAR_THRESHOLD  = 0.01    # pre-filtro varianza (sin usar y → sin leakage)

# Balanceo de clases
APPLY_SMOTE     = True
SMOTE_RATIO     = 0.8       # oversamplear minoría a 80% de mayoría

# Residualización de covariables (dentro de folds)
APPLY_RESIDUALIZATION = False
COVARIATE_COLS  = ['age']   # columnas a residualizar; 'SITE' se one-hot encode aparte

# Análisis de robustez
N_BOOTSTRAP     = 20        # resamples para estabilidad bootstrap

RANDOM_STATE    = int(os.environ.get('CV_SEED', 42))   # 42 = corridas originales; otras semillas = CV repetida

# ---- v3 (2026-09-21) ---------------------------------------------------------
# 'zero_variance' : drop only constant features (scale-independent). Default.
# 'legacy_var001' : exact behaviour of the original pipeline (VarianceThreshold 0.01
#                   on UNSCALED features) -- kept only for a sensitivity analysis.
PREFILTER_MODE    = 'zero_variance'
SAGE_PERMUTATIONS = 512
if os.environ.get('PIPE_SMOKE') == '1':      # quick end-to-end test only, never for results
    N_OUTER_FOLDS, N_INNER_FOLDS, N_ITER_SEARCH, N_BOOTSTRAP, SAGE_PERMUTATIONS = 3, 2, 3, 2, 16

# Output
OUTPUT_BASE = os.path.join(BASE_PATH, 'Resultados', 'graphics', 'ML_v2')

# Columnas a excluir de features
EXCLUDE_COLS = [
    'subject', 'group', 'Task', 'ses', 'mmse', 'moca',
    'group_sl', 'age_sl', 'SITE_sl',
    'group_coh', 'age_coh', 'SITE_coh',
    'group_ent', 'age_ent', 'SITE_ent',
    'group_cross', 'age_cross', 'SITE_cross',
]

# =============================================================================
# GRIDS DE HIPERPARÁMETROS
# =============================================================================

RF_PARAMS = {
    'n_estimators':      [100, 200, 300, 500],
    'max_features':      ['sqrt', 'log2', 0.3, 0.5],
    'max_depth':         [3, 5, 8, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 5],
    'class_weight':      ['balanced', 'balanced_subsample'],
    'criterion':         ['gini', 'entropy'],
}

SVM_PARAMS = {
    'C':            np.logspace(-3, 3, 20).tolist(),
    'kernel':       ['rbf', 'linear'],
    'gamma':        ['scale', 'auto', 0.001, 0.01],
    'class_weight': ['balanced'],
}

LR_PARAMS = {
    'C':            np.logspace(-4, 4, 20).tolist(),
    'penalty':      ['l1', 'l2'],
    'solver':       ['liblinear', 'saga'],
    'class_weight': ['balanced'],
    'max_iter':     [1000],
}

XGB_PARAMS = {
    'n_estimators':     [100, 200, 300, 500],
    'max_depth':        [3, 4, 5, 6],
    'learning_rate':    [0.01, 0.05, 0.1, 0.2],
    'subsample':        [0.6, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.7, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma':            [0, 0.1, 0.5],
    'reg_alpha':        [0, 0.1, 1.0],
    'scale_pos_weight': [1, 5, 10, 15],
}

CLASSIFIERS = [
    ('RF',  RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1), RF_PARAMS),
    ('SVM', SVC(probability=True, random_state=RANDOM_STATE, max_iter=5000), SVM_PARAMS),
    ('LR',  LogisticRegression(random_state=RANDOM_STATE),               LR_PARAMS),
    ('XGB', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss',
                          verbosity=0, n_jobs=1),                         XGB_PARAMS),
]

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'pipeline_v2.log')

    logger = logging.getLogger('pipeline_v2')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding='utf-8')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =============================================================================
# CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

def load_and_prepare(filepath, group1, group2, logger, include_age=False,
                     case_orig_labels=None, control_orig_labels=None,
                     site_filter=None, case_site_exclude=None,
                     case_site_include=None, control_site_exclude=None,
                     age_range=None, permute_seed=None):
    """
    Carga feather, filtra a 2 grupos, agrega a nivel de sujeto (media).

    include_age          : si True, agrega 'age' como feature explícita del modelo.
    case_orig_labels     : lista de valores de 'orig_group' para el grupo caso.
                           Si None, filtra por group == group1.
    control_orig_labels  : lista de valores de 'orig_group' para el grupo control.
                           Si None, filtra por group in g2_list.
    site_filter          : str o lista — si se especifica, restringe a sujetos con
                           SITE en ese valor/lista antes de cualquier otro filtro.
    case_site_exclude    : str o lista — excluye sujetos CASO cuyo SITE esté en
                           esa lista. Los controles NO se afectan.

    Retorna X, y, feature_names, subjects, ages, data_agg.
    """
    data = pd.read_feather(filepath)

    # Filtro de SITE (aplicado antes del filtro de grupo)
    if site_filter is not None and 'SITE' in data.columns:
        sites = [site_filter] if isinstance(site_filter, str) else list(site_filter)
        data = data[data['SITE'].isin(sites)].copy()
        logger.info(f"  Filtro SITE={sites}: {data['subject'].nunique()} sujetos")

    # Filtro de edad [min, max] (diagnostico de confusion por edad)
    if age_range is not None and 'age' in data.columns:
        data = data[(data['age'] >= age_range[0]) & (data['age'] <= age_range[1])].copy()
        logger.info(f"  Filtro edad {list(age_range)}: {data['subject'].nunique()} sujetos")

    # Filtrar grupos: por orig_group si se especifica, si no por group mapeado
    g2_list  = group2 if isinstance(group2, list) else [group2]
    has_orig = 'orig_group' in data.columns

    needs_orig = case_orig_labels is not None or control_orig_labels is not None
    if needs_orig and not has_orig:
        raise ValueError(
            f"El experimento requiere filtrar por 'orig_group' "
            f"(case_orig_labels={case_orig_labels}), pero la columna 'orig_group' "
            f"no existe en el feather '{os.path.basename(filepath)}'.\n"
            f"Solucion: re-ejecutar 1_make_dataframe.py y luego "
            f"optional_neuroharmonize.py para regenerar el feather con orig_group."
        )

    if needs_orig:
        is_case = (data['orig_group'].isin(case_orig_labels)
                   if case_orig_labels is not None
                   else data['group'] == group1)
        is_ctrl = (data['orig_group'].isin(control_orig_labels)
                   if control_orig_labels is not None
                   else data['group'].isin(g2_list))
        if case_site_exclude is not None and 'SITE' in data.columns:
            excl = [case_site_exclude] if isinstance(case_site_exclude, str) else list(case_site_exclude)
            is_case = is_case & ~data['SITE'].isin(excl)
            logger.info(f"  case_site_exclude={excl}: portadores de esos sitios excluidos")
        if case_site_include is not None and 'SITE' in data.columns:
            inc = [case_site_include] if isinstance(case_site_include, str) else list(case_site_include)
            is_case = is_case & data['SITE'].isin(inc)
        if control_site_exclude is not None and 'SITE' in data.columns:
            exc = [control_site_exclude] if isinstance(control_site_exclude, str) else list(control_site_exclude)
            is_ctrl = is_ctrl & ~data['SITE'].isin(exc)
        combined = is_case | is_ctrl
        data = data[combined].copy()
        data['group'] = np.where(is_case[combined].values, group1, g2_list[0])
    else:
        mask = data['group'].isin([group1] + g2_list)
        data = data[mask].copy()
        data['group'] = data['group'].apply(lambda x: g2_list[0] if x in g2_list else x)
        if case_site_exclude is not None and 'SITE' in data.columns:
            excl = [case_site_exclude] if isinstance(case_site_exclude, str) else list(case_site_exclude)
            drop = (data['group'] == group1) & data['SITE'].isin(excl)
            data = data[~drop].copy()
            logger.info(f"  case_site_exclude={excl}: portadores de esos sitios excluidos")

    logger.info(f"  Rows cargados: {len(data)} | Sujetos únicos: {data['subject'].nunique()}")
    logger.info(f"  Distribución: {data.groupby('group')['subject'].nunique().to_dict()}")
    logger.info(f"  Age como feature: {'Sí' if include_age else 'No'}")

    # Columnas de features (excluir siempre las meta-columnas)
    # age se excluye aquí y se añade explícitamente solo si include_age=True
    exclude = set(EXCLUDE_COLS) | {'subject', 'group', 'SITE', 'sex', 'age',
                                    'education', 'ses', 'mmse', 'moca', 'orig_group'}
    feature_cols = [c for c in data.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(data[c])]

    # Agregar a nivel sujeto (media de todas las filas del sujeto)
    data_feat = data.groupby('subject')[feature_cols].mean().reset_index()

    # Meta-datos (primera ocurrencia por sujeto)
    meta_keep = ['subject', 'group'] + [c for c in ['age', 'sex', 'education', 'SITE'] if c in data.columns]
    meta = data.groupby('subject')[meta_keep[1:]].first().reset_index()
    data_agg = data_feat.merge(meta, on='subject')

    logger.info(f"  Tras agregación: {len(data_agg)} sujetos, {len(feature_cols)} features EEG base")

    # Codificar SITE como dummies
    site_dummies_cols = []
    if 'SITE' in data_agg.columns:
        dummies = pd.get_dummies(data_agg['SITE'], prefix='SITE', drop_first=True, dtype=float)
        data_agg = pd.concat([data_agg.drop('SITE', axis=1), dummies], axis=1)
        site_dummies_cols = list(dummies.columns)

    # Codificar sexo (guardado en data_agg pero NO como feature del clasificador)
    if 'sex' in data_agg.columns:
        sex_map = {'M': 1.0, 'F': 0.0, 'H': 1.0, 'male': 1.0, 'female': 0.0}
        data_agg['sex'] = data_agg['sex'].map(sex_map).fillna(0.5)
        # sex excluido de features: cobertura incompleta (Seoul=0%) introduce artefacto

    # education excluida de features: PSEN1=100% cobertura, controles Seoul=0%
    # → el modelo aprendería disponibilidad de dato, no señal EEG

    # SITE dummies guardadas en data_agg pero NO como features del clasificador:
    # neuroHarmonize ya removió el efecto de sitio de los features EEG;
    # incluir SITE reintroduciría el confound (PSEN1=100% Medellín)

    # Incluir age como feature si se solicita (condición covariates_in_model)
    if include_age and 'age' in data_agg.columns:
        feature_cols = feature_cols + ['age']

    # Feature cols finales: solo EEG features [+ age si aplica]
    all_feature_cols = [c for c in feature_cols if c in data_agg.columns]

    X = data_agg[all_feature_cols].values.astype(np.float64)
    y = (data_agg['group'] == group1).astype(int).values
    if permute_seed is not None:   # control negativo: etiquetas permutadas (misma prevalencia)
        y = np.random.RandomState(int(permute_seed)).permutation(y)
        logger.info(f"  [PERMUTACION] etiquetas permutadas, seed={permute_seed}")
    feature_names = all_feature_cols
    subjects = data_agg['subject'].values
    ages = data_agg['age'].values if 'age' in data_agg.columns else None

    return X, y, feature_names, subjects, ages, data_agg


# =============================================================================
# PRE-FILTRADO DE FEATURES (sin usar y → sin leakage de etiquetas)
# =============================================================================

def prefilter_features(X, feature_names, logger, mode=None):
    """
    Label-free pre-filter applied before cross-validation.

    v3: by default only constant (zero-variance) features are dropped. The original
    filter (variance < 0.01 on unscaled features) removed features because of their
    measurement scale (all synchronization-likelihood and entropy features, most power
    and coherence features), not because they were uninformative. It also swallowed
    errors silently. Feature-count control is done inside the nested CV (k is capped by
    the number of training subjects per classifier), so nothing is lost by not filtering.
    """
    mode = mode or PREFILTER_MODE
    n_orig = X.shape[1]
    Xz = np.nan_to_num(X, nan=0.0)
    if mode == 'zero_variance':
        keep = Xz.var(axis=0) > 0.0
        if not keep.any():
            raise ValueError("prefilter: all features have zero variance")
        X_out = X[:, keep]
        f_out = np.array(feature_names)[keep].tolist()
    elif mode == 'legacy_var001':
        vt = VarianceThreshold(threshold=VAR_THRESHOLD)
        try:
            X_out = vt.fit_transform(Xz)
            f_out = np.array(feature_names)[vt.get_support()].tolist()
        except Exception as exc:
            logger.warning(f"  legacy prefilter found no feature above threshold ({exc}); keeping all")
            X_out, f_out = X, list(feature_names)
    else:
        raise ValueError(f"unknown prefilter mode: {mode}")
    logger.info(f"  Pre-filtro ({mode}): {n_orig} -> {len(f_out)} features")
    return X_out, f_out


def get_k_candidates(n_samples, n_features_avail, clf_name='LR'):
    """
    Candidatos de k para SelectKBest/RFE, con cap específico por clasificador.
    LR: N//10 (regla EPV para regresión logística).
    SVM: N//6 (kernel SVM tolera más features por regularización implícita).
    RF: N//4 (árboles de decisión toleran ratios feature/muestra altos).
    """
    ratio = _K_MAX_RATIO.get(clf_name, 10)
    max_k = min(n_features_avail, max(5, n_samples // ratio))
    return [k for k in K_CANDIDATES if k <= max_k] or [5]


# =============================================================================
# RESIDUALIZACIÓN DE COVARIABLES (dentro de fold → sin leakage)
# =============================================================================

def residualize_covariates(X_train, X_test, cov_train, cov_test):
    """Elimina el efecto lineal de covariables de cada feature (dentro del fold)."""
    cov_tr = np.nan_to_num(np.array(cov_train, dtype=float), nan=0.0)
    cov_mean = cov_tr.mean(axis=0, keepdims=True)
    cov_te = np.where(np.isnan(cov_test), cov_mean, np.array(cov_test, dtype=float))

    X_tr_r = X_train.copy()
    X_te_r = X_test.copy()

    for i in range(X_train.shape[1]):
        reg = LinearRegression()
        reg.fit(cov_tr, X_train[:, i])
        X_tr_r[:, i] -= reg.predict(cov_tr)
        X_te_r[:, i] -= reg.predict(cov_te)

    return X_tr_r, X_te_r


# =============================================================================
# MÉTRICAS
# =============================================================================

def compute_metrics(y_true, y_pred, y_proba):
    """Calcula todas las métricas de clasificación."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    auc_val = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'auc':       auc_val,
        'brier':     brier_score_loss(y_true, y_proba),
        'cm':        cm,
    }


# =============================================================================
# NESTED CROSS-VALIDATION (NÚCLEO PRINCIPAL)
# =============================================================================

def _apply_smote_to_train(X_tr, y_tr, random_state=RANDOM_STATE):
    """SMOTE + undersampling sobre train. Retorna (X_bal, y_bal)."""
    counts = np.bincount(y_tr)
    ratio  = counts.min() / counts.max() if counts.max() > 0 else 1.0
    if ratio >= 0.8:
        return X_tr, y_tr
    k_nb = min(5, int(counts.min()) - 1)
    if k_nb < 1:
        return X_tr, y_tr
    try:
        sm  = SMOTE(sampling_strategy=SMOTE_RATIO, random_state=random_state, k_neighbors=k_nb)
        rus = RandomUnderSampler(sampling_strategy=1.0, random_state=random_state)
        X_s, y_s = sm.fit_resample(X_tr, y_tr)
        X_s, y_s = rus.fit_resample(X_s, y_s)
        return X_s, y_s
    except Exception:
        return X_tr, y_tr


def _build_covariate_matrix(ages, sexes, idx, apply_resid, apply_sex_resid):
    """Construye la matriz de covariables para residualización dentro de un fold."""
    cols = []
    if apply_resid and ages is not None:
        cols.append(ages[idx].reshape(-1, 1))
    if apply_sex_resid and sexes is not None:
        cols.append(sexes[idx].reshape(-1, 1))
    return np.hstack(cols) if cols else None


def _preprocess_fold(X_tr_raw, X_te_raw, y_tr, ages, train_idx, test_idx,
                     apply_resid, apply_smote, sexes=None, apply_sex_resid=False):
    """
    Aplica la cadena completa de preprocesamiento dentro de un fold sin data leakage:
      KNNImputer → [residualización edad/sexo] → StandardScaler → [SMOTE]

    Retorna (X_tr_proc, y_tr_proc, X_te_proc)
    donde X_te_proc NO tiene SMOTE (solo impute+resid+scale).
    """
    # 1. Imputation
    imp = KNNImputer(n_neighbors=min(5, len(train_idx) - 1))
    X_tr = imp.fit_transform(X_tr_raw)
    X_te = imp.transform(X_te_raw)

    # 2. Residualización de covariables (edad y/o sexo)
    cov_tr = _build_covariate_matrix(ages, sexes, train_idx, apply_resid, apply_sex_resid)
    cov_te = _build_covariate_matrix(ages, sexes, test_idx,  apply_resid, apply_sex_resid)
    if cov_tr is not None:
        X_tr, X_te = residualize_covariates(X_tr, X_te, cov_tr, cov_te)

    # 3. Escalado
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    # 4. SMOTE (solo en train)
    if apply_smote:
        X_tr, y_tr = _apply_smote_to_train(X_tr, y_tr)

    return X_tr, y_tr, X_te


def _build_selector_pipeline(clf_base, fs_name, n_features_avail):
    """Construye Pipeline(selector, clf) y su grid con k como hiperparámetro."""
    from sklearn.pipeline import Pipeline as SkPipeline
    if fs_name == 'kbest':
        selector  = SelectKBest(f_classif)
        fs_key    = 'selector__k'
        step_name = 'selector'
    else:
        rfe_est  = RandomForestClassifier(n_estimators=30, max_depth=5,
                                          n_jobs=1, random_state=RANDOM_STATE)
        step_size = max(1, n_features_avail // 8)
        selector  = RFE(rfe_est, step=step_size)
        fs_key    = 'selector__n_features_to_select'
        step_name = 'selector'

    pipe = SkPipeline([(step_name, selector), ('clf', clone(clf_base))])
    return pipe, fs_key


def run_nested_cv(X, y, feature_names, ages, logger,
                  n_outer=10, n_inner=5, n_iter=50,
                  apply_smote=True, apply_resid=False,
                  sexes=None, apply_sex_resid=False):
    """
    Nested CV con k (número de features) como hiperparámetro del inner CV.

      Outer loop  : n_outer folds estratificados → estimación sin sesgo
      Inner loop  : RandomizedSearchCV sobre [clf params + k] → selección óptima

    Preprocesamiento dentro de cada fold (sin data leakage):
      KNNImputer → [residualización edad] → StandardScaler → [SMOTE]
      → Pipeline(SelectKBest/RFE, Classifier) optimizado por inner CV

    k se elige adaptativamente: candidatos en K_CANDIDATES ∩ [5, N_train/10].

    Retorna summary, y_true_all, stable_features, feature_votes.
    """
    n_splits = max(3, min(n_outer, int(np.bincount(y).min())))
    if n_splits < n_outer:
        logger.warning(f"  Clase minoritaria = {np.bincount(y).min()} → {n_splits} folds")

    outer_cv    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    combo_names = [f"{clf}_{fs}" for clf, _, _ in CLASSIFIERS for fs in ['kbest', 'rfe']]
    combo_names += ['ENS_soft']
    all_results = {k: [] for k in combo_names}
    all_cm      = {k: np.zeros((2, 2), dtype=int) for k in combo_names}
    all_y_proba = {k: [] for k in combo_names}
    all_oof_idx = {k: [] for k in combo_names}   # indices de test, alineados con all_y_proba
    y_true_all  = []
    feature_votes = {
        'kbest': np.zeros(X.shape[1], dtype=int),
        'rfe':   np.zeros(X.shape[1], dtype=int),
    }

    logger.info(f"  Iniciando nested CV: {n_splits}-fold outer × {n_inner}-fold inner")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        t0 = time.time()
        X_tr_raw = X[train_idx].copy()
        X_te_raw = X[test_idx].copy()
        y_tr     = y[train_idx]
        y_te     = y[test_idx]

        X_tr, y_tr_proc, X_te = _preprocess_fold(
            X_tr_raw, X_te_raw, y_tr, ages, train_idx, test_idx,
            apply_resid, apply_smote, sexes=sexes, apply_sex_resid=apply_sex_resid,
        )

        inner_cv = StratifiedKFold(
            n_splits=min(n_inner, int(np.bincount(y_tr_proc).min())),
            shuffle=True, random_state=RANDOM_STATE,
        )

        y_true_all.extend(y_te.tolist())
        fold_probas = {}   # acumula y_proba por combo para el ensemble

        for clf_name, clf_base, clf_params in CLASSIFIERS:
            # k candidates específicos por clasificador
            k_cands = get_k_candidates(len(y_tr_proc), X_tr.shape[1], clf_name)
            for fs_name in ['kbest', 'rfe']:
                key  = f"{clf_name}_{fs_name}"
                pipe, fs_key = _build_selector_pipeline(clf_base, fs_name, X_tr.shape[1])
                grid = {f'clf__{p}': v for p, v in clf_params.items()}
                grid[fs_key] = k_cands

                try:
                    search = RandomizedSearchCV(
                        pipe, grid,
                        n_iter=n_iter, cv=inner_cv,
                        scoring='roc_auc', n_jobs=1,
                        random_state=RANDOM_STATE, error_score=0.5, refit=True,
                    )
                    search.fit(X_tr, y_tr_proc)

                    # Registrar features seleccionadas por el mejor modelo
                    sel = search.best_estimator_.named_steps['selector']
                    feat_idx = (sel.get_support(indices=True)
                                if fs_name == 'kbest'
                                else np.where(sel.support_)[0])
                    feature_votes[fs_name][feat_idx] += 1

                    y_pred  = search.predict(X_te)
                    y_proba = search.predict_proba(X_te)[:, 1]
                    fold_probas[key] = (y_proba, search.best_score_)

                    m = compute_metrics(y_te, y_pred, y_proba)
                    all_results[key].append(m)
                    all_cm[key]     += m['cm']
                    all_y_proba[key].extend(y_proba.tolist())
                    all_oof_idx[key].extend(test_idx.tolist())

                except Exception as e:
                    logger.debug(f"    {key} fold {fold_idx+1} error: {e}")

        # Ensemble suave: promedio ponderado por AUC del inner CV
        if len(fold_probas) >= 2:
            try:
                total_w  = sum(w for _, w in fold_probas.values())
                ens_prob = sum(p * (w / total_w)
                               for p, w in fold_probas.values())
                ens_pred = (ens_prob >= 0.5).astype(int)
                m_ens    = compute_metrics(y_te, ens_pred, ens_prob)
                all_results['ENS_soft'].append(m_ens)
                all_cm['ENS_soft']     += m_ens['cm']
                all_y_proba['ENS_soft'].extend(ens_prob.tolist())
                all_oof_idx['ENS_soft'].extend(test_idx.tolist())
            except Exception as e:
                logger.debug(f"    ENS_soft fold {fold_idx+1} error: {e}")

        elapsed = time.time() - t0
        logger.info(f"  Fold {fold_idx+1}/{n_splits} completado ({elapsed:.1f}s) | "
                    f"Test n={len(y_te)} | pos={y_te.sum()}")

    # Agregar por fold
    summary = {}
    for key, fold_metrics in all_results.items():
        if not fold_metrics:
            continue
        agg = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'brier']:
            vals = [m[metric] for m in fold_metrics
                    if not np.isnan(m.get(metric, np.nan))]
            if vals:
                agg[f'{metric}_mean'] = float(np.mean(vals))
                agg[f'{metric}_std']  = float(np.std(vals))
        agg['n_folds']     = len(fold_metrics)
        agg['cm']          = all_cm[key]
        agg['y_proba_all'] = np.array(all_y_proba[key])
        summary[key] = agg

    # Exponer predicciones OOF (indice de fila de X + probabilidad) para analisis posteriores
    global _OOF_STORE
    _OOF_STORE = {k: (np.array(all_oof_idx[k], dtype=int), np.array(all_y_proba[k], dtype=float))
                  for k in combo_names if len(all_y_proba[k]) > 0}

    majority_thr    = n_splits // 2
    feat_arr        = np.array(feature_names)
    stable_features = {
        'kbest': feat_arr[feature_votes['kbest'] >= majority_thr].tolist(),
        'rfe':   feat_arr[feature_votes['rfe']   >= majority_thr].tolist(),
    }

    return summary, np.array(y_true_all), stable_features, feature_votes


# =============================================================================
# ANÁLISIS DE BOOTSTRAP (estabilidad de predicciones)
# =============================================================================

def bootstrap_stability(X, y, feature_names, ages, best_combo_name, n_bootstrap=20,
                        apply_smote=False, apply_resid=False, logger=None,
                        sexes=None, apply_sex_resid=False):
    """
    Estabilidad bootstrap: reentrena el pipeline completo (preprocessing + selector +
    clasificador) sobre N_BOOTSTRAP remuestras OOB. k se optimiza en cada resample
    igual que en el CV, garantizando consistencia metodológica.
    """
    clf_name, fs_name = best_combo_name.split('_', 1)
    clf_base = param_grid = None
    for name, clf, params in CLASSIFIERS:
        if name == clf_name:
            clf_base, param_grid = clf, params
            break

    auc_vals = []
    rng = np.random.RandomState(RANDOM_STATE)

    for b in range(n_bootstrap):
        idx_boot = rng.choice(len(y), size=len(y), replace=True)
        oob_mask = np.ones(len(y), dtype=bool)
        oob_mask[idx_boot] = False
        oob_idx  = np.where(oob_mask)[0]

        if len(oob_idx) < 4 or len(np.unique(y[oob_idx])) < 2:
            continue

        X_boot, y_boot = X[idx_boot].copy(), y[idx_boot].copy()
        X_oob,  y_oob  = X[oob_idx].copy(),  y[oob_idx]

        # Preprocessing (fit en boot, apply en oob)
        imp = KNNImputer(n_neighbors=min(5, len(idx_boot) - 1))
        X_boot = imp.fit_transform(X_boot)
        X_oob  = imp.transform(X_oob)

        cov_boot = _build_covariate_matrix(ages, sexes, idx_boot, apply_resid, apply_sex_resid)
        cov_oob  = _build_covariate_matrix(ages, sexes, oob_idx,  apply_resid, apply_sex_resid)
        if cov_boot is not None:
            X_boot, X_oob = residualize_covariates(X_boot, X_oob, cov_boot, cov_oob)

        sc = StandardScaler()
        X_boot = sc.fit_transform(X_boot)
        X_oob  = sc.transform(X_oob)

        if apply_smote:
            X_boot, y_boot = _apply_smote_to_train(X_boot, y_boot, random_state=b)

        # Pipeline selector + clf con k como hiperparámetro
        k_cands   = get_k_candidates(len(y_boot), X_boot.shape[1], clf_name)
        pipe, fs_key = _build_selector_pipeline(clf_base, fs_name, X_boot.shape[1])
        grid = {f'clf__{p}': v for p, v in param_grid.items()}
        grid[fs_key] = k_cands

        try:
            inner = StratifiedKFold(
                n_splits=min(3, int(np.bincount(y_boot).min())),
                shuffle=True, random_state=b,
            )
            s = RandomizedSearchCV(pipe, grid, n_iter=20, cv=inner,
                                   scoring='roc_auc', n_jobs=1,
                                   random_state=b, error_score=0.5)
            s.fit(X_boot, y_boot)
            proba = s.predict_proba(X_oob)[:, 1]
            auc_vals.append(roc_auc_score(y_oob, proba))
        except Exception:
            pass

    result = {
        'auc_mean':    float(np.mean(auc_vals)) if auc_vals else np.nan,
        'auc_std':     float(np.std(auc_vals))  if auc_vals else np.nan,
        'n_resamples': len(auc_vals),
    }
    if logger:
        logger.info(f"  Bootstrap ({len(auc_vals)} resamples): "
                    f"AUC = {result['auc_mean']:.3f} ± {result['auc_std']:.3f}")
    return result


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def shap_analysis(final_model, X_fs, feature_names_fs, clf_name,
                  output_dir, label='', logger=None):
    """
    SHAP: TreeExplainer (RF) o LinearExplainer (LR).
    SVM omitido por overhead de KernelExplainer.
    Genera: beeswarm plot + bar plot + Excel con mean |SHAP| por feature.
    """
    if not SHAP_AVAILABLE:
        if logger:
            logger.warning("  SHAP no disponible. Instalar con: pip install shap")
        return None

    if clf_name == 'SVM':
        if logger:
            logger.info("  SHAP: SVM omitido (KernelExplainer es muy lento para datos de alta dimensión)")
        return None

    try:
        if clf_name in ('RF', 'XGB'):
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_fs)
            # RF/XGB binario: puede ser lista [clase0, clase1] o array 3D
            if isinstance(shap_values, list) and len(shap_values) == 2:
                sv = shap_values[1]          # clase positiva
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv = shap_values[:, :, 1]    # clase positiva
            else:
                sv = shap_values

        elif clf_name == 'LR':
            explainer = shap.LinearExplainer(final_model, X_fs)
            shap_values = explainer.shap_values(X_fs)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                sv = shap_values[1]
            else:
                sv = shap_values
        else:
            return None

        sv = np.array(sv)
        n_display = min(15, len(feature_names_fs))

        # --- Beeswarm plot (dirección + magnitud por sujeto) ---
        plt.figure(figsize=(8, max(4, n_display * 0.45)))
        shap.summary_plot(
            sv, X_fs,
            feature_names=feature_names_fs,
            max_display=n_display,
            show=False,
            plot_size=None,
        )
        plt.title(f'SHAP Beeswarm — {label} ({clf_name})', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_beeswarm.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

        # --- Bar plot (mean |SHAP|) ---
        plt.figure(figsize=(7, max(4, n_display * 0.45)))
        shap.summary_plot(
            sv, X_fs,
            feature_names=feature_names_fs,
            max_display=n_display,
            plot_type='bar',
            show=False,
            plot_size=None,
        )
        plt.title(f'SHAP Importancia Media |SHAP| — {label} ({clf_name})', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_bar.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

        # Guardar tabla de importancias
        mean_abs = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature':        np.array(feature_names_fs),
            'mean_abs_shap':  mean_abs,
        }).sort_values('mean_abs_shap', ascending=False)
        shap_df.to_excel(os.path.join(output_dir, 'shap_importance.xlsx'), index=False)

        if logger:
            top3 = shap_df.head(3)['feature'].tolist()
            logger.info(f"  SHAP completado ({clf_name}). Top-3: {top3}")
        return shap_df

    except Exception as e:
        if logger:
            logger.warning(f"  SHAP análisis falló: {e}")
        return None


# =============================================================================
# SAGE ANALYSIS (primary interpretability)
# =============================================================================

def sage_analysis(final_model, X_fs, y, feature_names_fs,
                  output_dir, condition_name, clf_name, logger=None):
    """
    SAGE (Shapley Additive Global importancE) — análisis de importancia primario.
    Calcula valores globales con IC 95% bootstrap vía PermutationEstimator.
    Modelo-agnóstico: funciona para RF, SVM y LR.

    Genera:
      - sage_importance.png  : bar chart horizontal con IC 95%
      - sage_importance.xlsx : tabla con sage_value, sage_std, ci_low, ci_high
    """
    if not SAGE_AVAILABLE:
        if logger:
            logger.warning("  SAGE no disponible. Instalar con: pip install sage-importance")
        return None

    if not hasattr(final_model, 'predict_proba'):
        if logger:
            logger.info(f"  SAGE: {clf_name} no tiene predict_proba — omitido")
        return None

    try:
        # Subsample para eficiencia: máximo 256 sujetos de fondo
        # detect_convergence=True puede no terminar nunca con datos EEG ruidosos;
        # n_permutations fijo garantiza tiempo de ejecución acotado.
        n_bg = min(256, len(X_fs))
        N_SAGE_PERMUTATIONS = SAGE_PERMUTATIONS   # 512 in real runs
        rng = np.random.default_rng(RANDOM_STATE)
        bg_idx = rng.choice(len(X_fs), size=n_bg, replace=False)
        X_bg = X_fs[bg_idx]
        y_bg = np.asarray(y)[bg_idx]
        # Workaround: SAGE library accesses index n_bg (off-by-one bug).
        # Añadir una fila duplicada hace que ese acceso sea válido.
        X_bg_s = np.vstack([X_bg, X_bg[-1:]])
        y_bg_s = np.append(y_bg, y_bg[-1])

        if logger:
            logger.info(f"  SAGE iniciando ({clf_name}, n={n_bg}, "
                        f"permutations={N_SAGE_PERMUTATIONS}, loss=cross_entropy)...")

        imputer   = sage.MarginalImputer(final_model, X_bg_s)
        estimator = sage.PermutationEstimator(imputer, 'cross entropy')
        sage_values = estimator(X_bg_s, y_bg_s,
                                detect_convergence=False,
                                n_permutations=N_SAGE_PERMUTATIONS,
                                verbose=False, bar=False)

        vals = np.array(sage_values.values)   # (n_features,)
        stds = np.array(sage_values.std)       # (n_features,)
        ci_half = 1.96 * stds
        ci_low  = vals - ci_half
        ci_high = vals + ci_half

        feature_names_arr = np.array(feature_names_fs)
        order      = np.argsort(vals)[::-1]
        n_display  = min(20, len(feature_names_arr))
        top_idx    = order[:n_display]

        top_feats = feature_names_arr[top_idx]
        top_vals  = vals[top_idx]
        top_ci    = ci_half[top_idx]

        # Bar chart horizontal con IC 95%
        fig, ax = plt.subplots(figsize=(8, max(4, n_display * 0.48)))
        colors = ['#d62728' if v > 0 else '#1f77b4' for v in top_vals]
        ax.barh(range(n_display), top_vals[::-1], xerr=top_ci[::-1],
                color=colors[::-1], edgecolor='white', capsize=3, height=0.7,
                error_kw={'elinewidth': 1.2, 'ecolor': 'black', 'capthick': 1.2})
        ax.set_yticks(range(n_display))
        ax.set_yticklabels(top_feats[::-1], fontsize=9)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('SAGE Value (contribución a reducción de cross-entropy)', fontsize=9)
        ax.set_title(f'SAGE Feature Importance — {condition_name} ({clf_name})\n'
                     f'Top-{n_display} features | barras = IC 95%', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sage_importance.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

        # Excel con todos los features
        sage_df = pd.DataFrame({
            'feature':    feature_names_arr[order],
            'sage_value': vals[order],
            'sage_std':   stds[order],
            'ci_low':     ci_low[order],
            'ci_high':    ci_high[order],
        })
        sage_df.to_excel(os.path.join(output_dir, 'sage_importance.xlsx'), index=False)

        if logger:
            top3 = sage_df.head(3)['feature'].tolist()
            logger.info(f"  SAGE completado ({clf_name}). Top-3: {top3}")

        return sage_df

    except Exception as e:
        if logger:
            logger.warning(f"  SAGE análisis falló: {e}")
        return None


# =============================================================================
# CALIBRACIÓN + ECE
# =============================================================================

def compute_ece(y_true, y_proba, n_bins=10):
    """Expected Calibration Error con n_bins uniformes en [0, 1]."""
    y_true  = np.asarray(y_true,  dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() > 0:
            bin_acc  = float(y_true[mask].mean())
            bin_conf = float(y_proba[mask].mean())
            ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)


def plot_calibration(y_true, y_proba, output_dir, label='', n_bins=10):
    """
    Curva de calibración OOF + Brier Score + ECE.
    Usa las predicciones out-of-fold del nested CV (sin data leakage).
    """
    y_true  = np.asarray(y_true,  dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    brier = float(brier_score_loss(y_true, y_proba))
    ece   = compute_ece(y_true, y_proba, n_bins=n_bins)

    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy='uniform'
        )
    except Exception:
        frac_pos  = np.array([float(y_true.mean())])
        mean_pred = np.array([float(y_proba.mean())])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Calibración perfecta')
    ax.plot(mean_pred, frac_pos, 's-', color='steelblue', lw=2,
            label=f'Brier = {brier:.3f}  |  ECE = {ece:.3f}')
    ax.set_xlabel('Probabilidad predicha media')
    ax.set_ylabel('Fracción de positivos reales')
    ax.set_title(f'Curva de Calibración (OOF) — {label}', fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'), dpi=200)
    plt.close()

    return {'brier': brier, 'ece': ece}


# =============================================================================
# ROBUSTEZ: RUIDO EN FEATURES
# =============================================================================

def feature_noise_robustness(X, y, best_combo_name,
                              noise_levels=None, output_dir=None, logger=None):
    """
    Evalúa degradación de AUC al añadir ruido Gaussiano al TEST.
    Simula variabilidad entre equipos EEG de distintos sitios (multisite).

    noise_levels : fracciones de la std de cada feature
                   [0.0 = sin ruido, 0.05 = 5%, ..., 0.50 = 50%]
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.20, 0.50]

    clf_name, fs_name = best_combo_name.split('_', 1)
    clf_base = None
    for name, clf, _ in CLASSIFIERS:
        if name == clf_name:
            clf_base = clf
            break

    feat_stds = np.nanstd(X, axis=0)
    feat_stds = np.where(feat_stds == 0, 1.0, feat_stds)

    n_cv = max(3, min(5, int(np.bincount(y).min())))
    cv   = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=RANDOM_STATE)
    rng  = np.random.RandomState(RANDOM_STATE + 1)

    auc_by_level = []
    for sigma in noise_levels:
        fold_aucs = []
        for train_idx, test_idx in cv.split(X, y):
            X_tr = np.nan_to_num(X[train_idx].copy())
            X_te = np.nan_to_num(X[test_idx].copy())
            y_tr, y_te = y[train_idx], y[test_idx]

            # Ruido solo en TEST (simula nuevo sitio de adquisición)
            if sigma > 0:
                X_te = X_te + rng.normal(0, sigma * feat_stds, X_te.shape)

            # Preprocessing idéntico al pipeline principal
            imp = KNNImputer(n_neighbors=min(5, len(train_idx) - 1))
            X_tr = imp.fit_transform(X_tr)
            X_te = imp.transform(X_te)
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            k_opts = get_k_candidates(len(X_tr), X_tr.shape[1], clf_name)
            n_feat = k_opts[-1] if k_opts else 5
            if fs_name == 'kbest':
                fs = SelectKBest(f_classif, k=n_feat)
                X_tr_fs = fs.fit_transform(X_tr, y_tr)
                X_te_fs = fs.transform(X_te)
            else:
                rf_rfe = RandomForestClassifier(
                    n_estimators=30, max_depth=5,
                    n_jobs=-1, random_state=RANDOM_STATE)
                fs_obj = RFE(rf_rfe, n_features_to_select=n_feat,
                             step=max(1, n_feat))
                X_tr_fs = fs_obj.fit_transform(X_tr, y_tr)
                X_te_fs = fs_obj.transform(X_te)

            try:
                m = clone(clf_base)
                if hasattr(m, 'class_weight'):
                    m.set_params(class_weight='balanced')
                m.fit(X_tr_fs, y_tr)
                proba = m.predict_proba(X_te_fs)[:, 1]
                if len(np.unique(y_te)) > 1:
                    fold_aucs.append(roc_auc_score(y_te, proba))
            except Exception:
                pass

        auc_by_level.append(float(np.mean(fold_aucs)) if fold_aucs else np.nan)

    valid = [v for v in auc_by_level if not np.isnan(v)]
    base_auc  = auc_by_level[0]
    delta_auc = float(base_auc - min(valid)) if valid and not np.isnan(base_auc) else np.nan

    if logger:
        level_str = '  |  '.join(
            f"σ={s:.0%}→{a:.3f}" for s, a in zip(noise_levels, auc_by_level)
        )
        logger.info(f"  Ruido features: {level_str}  |  ΔAUC_max={delta_auc:.3f}")

    if output_dir:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([s * 100 for s in noise_levels], auc_by_level,
                'o-', color='steelblue', lw=2, markersize=6)
        ax.axhline(y=0.5, color='gray', linestyle='--', lw=1, label='Azar (AUC=0.5)')
        ax.set_xlabel('Nivel de ruido (% std feature)')
        ax.set_ylabel('AUC-ROC')
        ax.set_title('Robustez: Ruido en features (multisite)', fontsize=10)
        valid_v = [v for v in auc_by_level if not np.isnan(v)]
        ax.set_ylim(max(0, (min(valid_v) - 0.1) if valid_v else 0), 1.02)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'robustness_noise.png'), dpi=200)
        plt.close()

    return {'noise_aucs': auc_by_level, 'levels': noise_levels, 'delta_auc': delta_auc}


# =============================================================================
# ROBUSTEZ: SENSIBILIDAD A HIPERPARÁMETROS
# =============================================================================

def hyperparameter_sensitivity(X, y, best_combo_name, best_params=None,
                                output_dir=None, logger=None):
    """
    AUC al perturbar el hiperparámetro clave del mejor modelo.
    HP clave: C (LR/SVM) o n_estimators (RF).
    Factores: [0.25, 0.5, 1.0, 2.0, 4.0] × valor base del CV.
    """
    clf_name, fs_name = best_combo_name.split('_', 1)
    clf_base = None
    for name, clf, _ in CLASSIFIERS:
        if name == clf_name:
            clf_base = clf
            break

    factors = [0.25, 0.5, 1.0, 2.0, 4.0]
    if clf_name == 'RF':
        param_name = 'n_estimators'
        base_val   = int(best_params.get('n_estimators', 200)) if best_params else 200
        hp_vals    = [max(10, int(base_val * f)) for f in factors]
    else:  # LR o SVM
        param_name = 'C'
        base_val   = float(best_params.get('C', 1.0)) if best_params else 1.0
        hp_vals    = [base_val * f for f in factors]

    n_cv = max(3, min(5, int(np.bincount(y).min())))
    cv   = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=RANDOM_STATE)

    auc_by_hp = []
    for hp_val in hp_vals:
        fold_aucs = []
        for train_idx, test_idx in cv.split(X, y):
            X_tr = np.nan_to_num(X[train_idx].copy())
            X_te = np.nan_to_num(X[test_idx].copy())
            y_tr, y_te = y[train_idx], y[test_idx]

            imp = KNNImputer(n_neighbors=min(5, len(train_idx) - 1))
            X_tr = imp.fit_transform(X_tr)
            X_te = imp.transform(X_te)
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            k_opts = get_k_candidates(len(X_tr), X_tr.shape[1], clf_name)
            n_feat = k_opts[-1] if k_opts else 5
            if fs_name == 'kbest':
                fs = SelectKBest(f_classif, k=n_feat)
                X_tr_fs = fs.fit_transform(X_tr, y_tr)
                X_te_fs = fs.transform(X_te)
            else:
                rf_rfe = RandomForestClassifier(
                    n_estimators=30, max_depth=5,
                    n_jobs=-1, random_state=RANDOM_STATE)
                fs_obj = RFE(rf_rfe, n_features_to_select=n_feat,
                             step=max(1, n_feat))
                X_tr_fs = fs_obj.fit_transform(X_tr, y_tr)
                X_te_fs = fs_obj.transform(X_te)

            try:
                m = clone(clf_base)
                params_upd = {param_name: hp_val}
                if hasattr(m, 'class_weight'):
                    params_upd['class_weight'] = 'balanced'
                m.set_params(**params_upd)
                m.fit(X_tr_fs, y_tr)
                proba = m.predict_proba(X_te_fs)[:, 1]
                if len(np.unique(y_te)) > 1:
                    fold_aucs.append(roc_auc_score(y_te, proba))
            except Exception:
                pass

        auc_by_hp.append(float(np.mean(fold_aucs)) if fold_aucs else np.nan)

    valid = [v for v in auc_by_hp if not np.isnan(v)]
    delta_auc = float(max(valid) - min(valid)) if valid else np.nan

    if logger:
        hp_str = '  |  '.join(
            f"×{f}→{a:.3f}" for f, a in zip(factors, auc_by_hp)
        )
        logger.info(
            f"  HP sensitivity ({param_name}, base={base_val:.3g}): "
            f"{hp_str}  |  ΔAUC={delta_auc:.3f}"
        )

    if output_dir:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(factors, auc_by_hp, 'o-', color='darkorange', lw=2, markersize=6)
        ax.axvline(x=1.0, color='gray', linestyle='--', lw=1, label='Valor base CV')
        ax.set_xlabel(f'Factor × {param_name} (base = {base_val:.3g})')
        ax.set_ylabel('AUC-ROC')
        ax.set_title(f'Sensibilidad Hiperparámetro — {clf_name}', fontsize=10)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'robustness_hp_sensitivity.png'), dpi=200)
        plt.close()

    return {
        'hp_aucs':    auc_by_hp,
        'factors':    factors,
        'hp_vals':    hp_vals,
        'param_name': param_name,
        'base_val':   base_val,
        'delta_auc':  delta_auc,
    }


# =============================================================================
# ENTRENAMIENTO DE MODELO FINAL (sobre TODOS los datos)
# =============================================================================

def train_final_model(X, y, feature_names, best_combo_name, output_dir, logger,
                      group1=None, group2=None, ages=None, apply_resid=False,
                      apply_smote=False, sexes=None, apply_sex_resid=False):
    """
    Entrena el modelo final sobre TODOS los datos.
    Replica el mismo pipeline del CV: impute → resid → scale → SMOTE →
    Pipeline(selector, clf) con k optimizado por inner CV.
    """
    group1 = group1 or GROUP1
    group2 = group2 or GROUP2
    clf_name, fs_name = best_combo_name.split('_', 1)
    clf_base = param_grid = None
    for name, clf, params in CLASSIFIERS:
        if name == clf_name:
            clf_base, param_grid = clf, params
            break

    # Preprocessing sobre todos los datos (sin leakage: no hay test aquí)
    imp = KNNImputer(n_neighbors=5)
    X_p = imp.fit_transform(X)

    all_idx = np.arange(len(y))
    cov_all = _build_covariate_matrix(ages, sexes, all_idx, apply_resid, apply_sex_resid)
    if cov_all is not None:
        dummy = np.zeros_like(X_p)
        X_p, _ = residualize_covariates(X_p, dummy, cov_all, cov_all)

    sc  = StandardScaler()
    X_p = sc.fit_transform(X_p)

    y_p = y.copy()
    X_real = X_p.copy()      # v3: real participants (no synthetic SMOTE rows) for SAGE / SHAP
    if apply_smote:
        X_p, y_p = _apply_smote_to_train(X_p, y_p)

    # Pipeline selector + clf con k como hiperparámetro
    k_cands   = get_k_candidates(len(y_p), X_p.shape[1], clf_name)
    pipe, fs_key = _build_selector_pipeline(clf_base, fs_name, X_p.shape[1])
    grid = {f'clf__{p}': v for p, v in param_grid.items()}
    grid[fs_key] = k_cands

    _min_class    = int(np.bincount(y_p).min())
    _n_cv_final   = max(2, min(5, _min_class))
    inner_cv      = StratifiedKFold(n_splits=_n_cv_final, shuffle=True, random_state=RANDOM_STATE)
    search        = RandomizedSearchCV(
        pipe, grid, n_iter=N_ITER_SEARCH, cv=inner_cv,
        scoring='roc_auc', n_jobs=1, random_state=RANDOM_STATE,
    )
    search.fit(X_p, y_p)

    best_pipe    = search.best_estimator_
    sel          = best_pipe.named_steps['selector']
    selected_idx = (sel.get_support(indices=True)
                    if fs_name == 'kbest' else np.where(sel.support_)[0])
    selected_features = [feature_names[i] for i in selected_idx]

    # X en espacio de features seleccionadas (para SAGE/SHAP)
    X_fs = best_pipe.named_steps['selector'].transform(X_real)   # v3: aligned with the original y

    # Guardar artefactos
    model_info = {
        'model':             best_pipe.named_steps['clf'],
        'pipeline':          best_pipe,
        'imputer':           imp,
        'scaler':            sc,
        'feature_selector':  sel,
        'feature_names':     selected_features,
        'all_feature_names': feature_names,
        'classifier_name':   clf_name,
        'feature_selection': fs_name,
        'best_params':       search.best_params_,
        'group_mapping':     {group2: 0, group1: 1},
        'class_names':       [group2, group1],
        'n_features_used':   len(selected_features),
        'n_subjects_train':  len(y),
        'pos_class':         group1,
    }
    joblib.dump(model_info, os.path.join(output_dir, 'final_model.pkl'))
    logger.info(f"  Modelo final guardado: {clf_name} | {fs_name} | "
                f"{len(selected_features)} features (k_opt={len(selected_features)})")

    return best_pipe.named_steps['clf'], X_fs, selected_features, search.best_params_


# =============================================================================
# VISUALIZACIONES
# =============================================================================

def plot_aggregated_cm(cm, class_names, output_dir, title):
    """Confusion matrix sumada sobre todos los folds de CV."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar=False)
    for i in range(2):
        for j in range(2):
            ax.texts[i * 2 + j].set_text(
                f"{cm[i,j]}\n({cm_norm[i,j]:.0%})"
            )
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Etiqueta real')
    ax.set_xlabel('Etiqueta predicha')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cm_cv_aggregated.png'), dpi=200)
    plt.close()


def plot_roc_cv(y_true_all, y_proba_all, output_dir, label=''):
    """Curva ROC de predicciones concatenadas de todos los folds."""
    if len(np.unique(y_true_all)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true_all, y_proba_all)
    auc_val = roc_auc_score(y_true_all, y_proba_all)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color='steelblue', label=f'AUC = {auc_val:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color='steelblue')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve CV — {label}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_cv_{label}.png'), dpi=200)
    plt.close()


def plot_feature_votes(feature_votes, feature_names, output_dir, method='kbest', top_n=30):
    """Barplot de frecuencia de selección de features a través de los folds."""
    votes    = feature_votes[method]
    feat_arr = np.array(feature_names)
    order    = np.argsort(votes)[::-1][:top_n]
    top_feats = feat_arr[order]
    top_votes = votes[order]
    n = len(top_votes)  # puede ser < top_n si hay pocas features disponibles

    if n == 0:
        return

    colors = ['steelblue' if v >= len(feature_votes[method]) // 2 else 'lightgray'
              for v in top_votes]

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.35)))
    ax.barh(range(n), top_votes[::-1], color=colors[::-1])
    ax.set_yticks(range(n))
    ax.set_yticklabels(top_feats[::-1], fontsize=8)
    ax.axvline(x=max(top_votes) / 2, color='red', linestyle='--', lw=1,
               label='Umbral mayoría (50%)')
    ax.set_xlabel('N° folds seleccionado')
    ax.set_title(f'Estabilidad de Feature Selection ({method.upper()}) a través de folds')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_votes_{method}.png'), dpi=200)
    plt.close()


def print_results_table(summary, condition_name):
    """Imprime tabla de resultados ordenada por AUC."""
    print(f"\n{'='*72}")
    print(f"  CONDICIÓN: {condition_name}")
    print(f"{'='*72}")
    print(f"  {'Combo':<18} {'AUC':>13} {'F1':>13} {'Recall':>13} {'Brier':>8}")
    print(f"  {'-'*64}")

    sorted_combos = sorted(
        [(k, v) for k, v in summary.items() if 'auc_mean' in v],
        key=lambda x: x[1]['auc_mean'],
        reverse=True
    )
    for combo, m in sorted_combos:
        auc_s   = f"{m['auc_mean']:.3f}±{m['auc_std']:.3f}"
        f1_s    = f"{m['f1_mean']:.3f}±{m['f1_std']:.3f}"
        rec_s   = f"{m['recall_mean']:.3f}±{m['recall_std']:.3f}"
        brier_s = f"{m['brier_mean']:.3f}"
        marker  = ' <-best' if combo == sorted_combos[0][0] else ''
        print(f"  {combo:<18} {auc_s:>13} {f1_s:>13} {rec_s:>13} {brier_s:>8}{marker}")


def save_comparison_table(all_condition_results, output_path):
    """Tabla plana con todos los resultados (condición × combo × métricas)."""
    rows = []
    for condition, results_dict in all_condition_results.items():
        summary     = results_dict['summary']
        n_subjects  = results_dict.get('n_subjects', '?')
        n_pos       = results_dict.get('n_pos', '?')
        bootstrap   = results_dict.get('bootstrap', {})

        for combo, m in summary.items():
            row = {
                'condition':      condition,
                'combo':          combo,
                'n_subjects':     n_subjects,
                'n_pos':          n_pos,
            }
            for k, v in m.items():
                if k not in ('cm', 'y_proba_all'):
                    row[k] = round(v, 4) if isinstance(v, float) else v

            # Bootstrap si corresponde al mejor combo
            if bootstrap and combo == results_dict.get('best_combo'):
                row['bootstrap_auc_mean'] = bootstrap.get('auc_mean', np.nan)
                row['bootstrap_auc_std']  = bootstrap.get('auc_std', np.nan)

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    return df


# =============================================================================
# FILTRADO POR FAMILIA DE FEATURES
# =============================================================================

_FAMILY_PREDICATES = {
    'power':      lambda c: '/' not in c and not any(c.endswith(s) for s in ('_sl', '_coh', '_ent')),
    'sl':         lambda c: c.endswith('_sl'),
    'coh':        lambda c: c.endswith('_coh'),
    'entropy':    lambda c: c.endswith('_ent'),
    'crossfreq':  lambda c: '/' in c,
    'node_level': lambda c: '/' not in c,   # power + sl + coh + entropy (sin ratios)
    'all':        lambda c: True,
}

def _filter_features_by_family(X, feature_names, family, logger):
    """Filtra columnas de X y feature_names a la familia especificada."""
    if not family or family == 'all':
        return X, feature_names
    if family not in _FAMILY_PREDICATES:
        raise ValueError(f"feature_family='{family}' desconocida. Opciones: {list(_FAMILY_PREDICATES)}")
    mask = np.array([_FAMILY_PREDICATES[family](c) for c in feature_names])
    X_fam = X[:, mask]
    names_fam = [n for n, m in zip(feature_names, mask) if m]
    logger.info(f"  Familia features '{family}': {mask.sum()}/{len(feature_names)} features")
    return X_fam, names_fam


# =============================================================================
# PIPELINE POR CONDICIÓN
# =============================================================================

def run_condition(condition_name, filepath, output_dir, logger,
                  apply_resid=False, include_age=False, exp_config=None,
                  feature_family=None, residualize_sex=False):
    """
    Ejecuta el pipeline completo para una condición (feather file).

    apply_resid  : elimina el efecto lineal de la edad dentro de cada fold.
    include_age  : incluye 'age' como feature explícita del modelo.
    exp_config   : dict del experiments_registry. Si se provee, sus valores
                   tienen prioridad sobre los globals GROUP1/GROUP2/APPLY_SMOTE
                   y se guarda metadata del experimento.

    Retorna dict con summary, bootstrap, best_combo, n_subjects, n_pos.
    """
    # Extraer overrides del exp_config (si existe)
    if residualize_sex:
        raise ValueError("v3: sex residualization is disabled; age is the only covariate.")
    _group1      = exp_config.get('case_label',    GROUP1)    if exp_config else GROUP1
    _group2      = exp_config.get('control_label', GROUP2)    if exp_config else GROUP2
    _apply_smote = exp_config.get('smote',         APPLY_SMOTE) if exp_config else APPLY_SMOTE
    _case_orig         = exp_config.get('case_orig_labels')       if exp_config else None
    _ctrl_orig         = exp_config.get('control_orig_labels')    if exp_config else None
    _site_filter       = exp_config.get('site_filter')             if exp_config else None
    _case_site_exclude = exp_config.get('case_site_exclude')       if exp_config else None
    _case_site_include = exp_config.get('case_site_include')       if exp_config else None
    _ctrl_site_exclude = exp_config.get('control_site_exclude')    if exp_config else None
    _age_range         = exp_config.get('age_range')               if exp_config else None
    _permute_seed      = exp_config.get('permute_seed')            if exp_config else None

    tags = []
    if include_age:
        tags.append('age como feature')
    if apply_resid:
        tags.append('residualizacion edad')
    if residualize_sex:
        tags.append('residualizacion sexo')
    if feature_family and feature_family != 'all':
        tags.append(f'features={feature_family}')
    tag_str = f" [{', '.join(tags)}]" if tags else ''

    logger.info(f"\n{'='*60}")
    logger.info(f"  CONDICIÓN: {condition_name}{tag_str}")
    logger.info(f"  Archivo: {os.path.basename(filepath)}")
    logger.info(f"{'='*60}")
    os.makedirs(output_dir, exist_ok=True)

    # Checkpointing: saltar si ya está completo
    done_flag = os.path.join(output_dir, 'DONE.flag')
    if os.path.exists(done_flag):
        logger.info(f"  [SKIP] {condition_name} ya completado (borrar DONE.flag para re-ejecutar)")
        return None

    t_start = time.time()

    # 1. Cargar y preparar datos
    X, y, feature_names, subjects, ages, data_agg = load_and_prepare(
        filepath, _group1, _group2, logger, include_age=include_age,
        case_orig_labels=_case_orig, control_orig_labels=_ctrl_orig,
        site_filter=_site_filter, case_site_exclude=_case_site_exclude,
        case_site_include=_case_site_include, control_site_exclude=_ctrl_site_exclude,
        age_range=_age_range, permute_seed=_permute_seed,
    )

    # Extraer sexes para residualización (ya codificado M=1/F=0 en load_and_prepare)
    sexes = data_agg['sex'].values.astype(float) if (residualize_sex and 'sex' in data_agg.columns) else None

    # Filtrar por familia de features (antes del prefilter de varianza)
    X, feature_names = _filter_features_by_family(X, feature_names, feature_family, logger)
    n_subjects = len(y)
    n_pos      = int(y.sum())

    # Guardar metadata del experimento (sujetos, sitios, demografía)
    if exp_config is not None:
        from experiment_metadata import save_experiment_metadata, print_metadata_summary
        meta = save_experiment_metadata(data_agg, exp_config, output_dir)
        print_metadata_summary(meta)

    # Guard: skip conditions with too few subjects to run stratified CV
    # Need at least 6 per class (≥ 2 folds × 3 minimum usable)
    MIN_PER_CLASS = 6
    if n_pos < MIN_PER_CLASS or (n_subjects - n_pos) < MIN_PER_CLASS:
        logger.warning(
            f"  [SKIP] {condition_name}: muestra insuficiente para CV "
            f"(PSEN1={n_pos}, Control={n_subjects-n_pos}). "
            f"Mínimo requerido: {MIN_PER_CLASS} por clase. "
            f"Verifica que 2_apply_psm.py generó archivos válidos."
        )
        return None

    # 2. Pre-filtrado de features (sin y)
    X_filt, feat_filt = prefilter_features(X, feature_names, logger,
                                           mode=(exp_config or {}).get('prefilter'))

    # 3. Nested CV
    if apply_resid and ages is None:
        logger.warning("  apply_resid=True pero no hay columna 'age' en los datos → "
                       "residualización omitida.")
        apply_resid = False

    summary, y_true_all, stable_features, feature_votes = run_nested_cv(
        X_filt, y, feat_filt, ages, logger,
        n_outer=N_OUTER_FOLDS, n_inner=N_INNER_FOLDS, n_iter=N_ITER_SEARCH,
        apply_smote=_apply_smote, apply_resid=apply_resid,
        sexes=sexes, apply_sex_resid=residualize_sex,
    )

    print_results_table(summary, condition_name)

    # Guardar predicciones out-of-fold con ID de sujeto (para partial confounder test, etc.)
    try:
        oof = pd.DataFrame({'subject': np.asarray(subjects), 'y': y})
        for _c in ('age', 'sex', 'SITE', 'orig_group'):
            if _c in data_agg.columns:
                oof[_c] = data_agg[_c].values
        for _k, (_i, _p) in _OOF_STORE.items():
            _col = np.full(len(y), np.nan); _col[_i] = _p
            oof[f'proba_{_k}'] = _col
        oof['best_combo'] = summary and max(summary, key=lambda c: summary[c].get('auc_mean', -1))
        oof.to_csv(os.path.join(output_dir, 'oof_predictions.csv'), index=False)
    except Exception as _e:
        logger.warning(f'  No se pudo guardar oof_predictions.csv: {_e}')

    # 4. Mejor combo (por AUC)
    valid = [(k, v) for k, v in summary.items() if 'auc_mean' in v]
    if not valid:
        logger.error(f"  Sin resultados válidos para {condition_name}")
        return None
    best_combo = max(valid, key=lambda x: x[1]['auc_mean'])[0]
    logger.info(f"  Mejor combo: {best_combo} | "
                f"AUC = {summary[best_combo]['auc_mean']:.3f} ± {summary[best_combo]['auc_std']:.3f}")

    # ENS_soft no tiene un clasificador sklearn real → para bootstrap y modelo final
    # se usa el mejor combo INDIVIDUAL (mayor AUC excluyendo ENS_soft)
    single_valid = [(k, v) for k, v in valid if k != 'ENS_soft']
    model_combo  = (max(single_valid, key=lambda x: x[1]['auc_mean'])[0]
                    if single_valid else best_combo)
    if model_combo != best_combo:
        logger.info(f"  Combo para modelo final / bootstrap: {model_combo} "
                    f"(ENS_soft no es entrenable como modelo único)")

    # 5. Visualizaciones del mejor combo
    best_m = summary[best_combo]
    plot_aggregated_cm(
        best_m['cm'], [_group2, _group1], output_dir,
        title=f'CV Aggregated CM — {condition_name}\n({best_combo})'
    )
    if len(y_true_all) > 0 and len(np.unique(y_true_all)) > 1:
        plot_roc_cv(y_true_all, best_m['y_proba_all'], output_dir,
                    label=condition_name.replace(':', '').replace(' ', '_'))

    for method in ['kbest', 'rfe']:
        plot_feature_votes(feature_votes, feat_filt, output_dir, method=method)

    # 6. Features estables (voto mayoritario)
    stable_df = pd.DataFrame({
        'kbest_stable': pd.Series(stable_features.get('kbest', [])),
        'rfe_stable':   pd.Series(stable_features.get('rfe', [])),
    })
    stable_df.to_excel(os.path.join(output_dir, 'stable_features.xlsx'), index=False)

    # 7. Bootstrap stability (usa model_combo: mejor combo INDIVIDUAL)
    bs = bootstrap_stability(X_filt, y, feat_filt, ages, model_combo,
                             n_bootstrap=N_BOOTSTRAP,
                             apply_smote=_apply_smote, apply_resid=apply_resid,
                             logger=logger,
                             sexes=sexes, apply_sex_resid=residualize_sex)

    # 8. Modelo final (usa model_combo)
    final_model, X_fs, selected_features, best_params = train_final_model(
        X_filt, y, feat_filt, model_combo, output_dir, logger,
        group1=_group1, group2=_group2,
        ages=ages, apply_resid=apply_resid, apply_smote=_apply_smote,
        sexes=sexes, apply_sex_resid=residualize_sex,
    )

    # 9. SAGE (primario) + SHAP beeswarm (suplementario)
    clf_name_best = model_combo.split('_')[0]
    sage_analysis(
        final_model, X_fs, y, selected_features,
        output_dir, condition_name, clf_name_best, logger=logger
    )
    shap_analysis(
        final_model, X_fs, selected_features,
        clf_name_best, output_dir,
        label=condition_name + ' [suplementario]', logger=logger
    )

    # 10. Calibración OOF (Brier + ECE + curva)
    calib = plot_calibration(
        y_true_all,
        summary[best_combo]['y_proba_all'],
        output_dir,
        label=condition_name,
    )
    logger.info(
        f"  Calibración OOF — Brier={calib['brier']:.3f} | ECE={calib['ece']:.3f}"
    )

    # 11. Robustez: ruido en features (multisite)
    noise_rob = feature_noise_robustness(
        X_filt, y, best_combo,
        output_dir=output_dir, logger=logger
    )

    # 12. Robustez: sensibilidad a hiperparámetros
    hp_rob = hyperparameter_sensitivity(
        X_filt, y, best_combo,
        best_params=best_params,
        output_dir=output_dir, logger=logger
    )

    # 13. Resumen estructurado (JSON) — para compare_experiments.py
    import json as _json
    elapsed = time.time() - t_start
    results_json = {
        'experiment_id':   exp_config.get('id', condition_name) if exp_config else condition_name,
        'condition_name':  condition_name,
        'best_combo':      best_combo,
        'n_subjects':      n_subjects,
        'n_case':          n_pos,
        'n_control':       n_subjects - n_pos,
        'group1':          _group1,
        'group2':          _group2,
        'apply_resid':     apply_resid,
        'include_age':     include_age,
        'apply_smote':     _apply_smote,
        'elapsed_min':     round(elapsed / 60, 2),
        'combos': {
            k: {m: round(v, 4) if isinstance(v, float) else v
                for m, v in metrics.items() if m not in ('cm', 'y_proba_all')}
            for k, metrics in summary.items()
        },
        'bootstrap': {
            'auc_mean': round(bs.get('auc_mean', float('nan')), 4),
            'auc_std':  round(bs.get('auc_std',  float('nan')), 4),
            'n_resamples': bs.get('n_resamples', 0),
        },
        'calibration': {
            'brier': round(calib.get('brier', float('nan')), 4),
            'ece':   round(calib.get('ece',   float('nan')), 4),
        },
    }
    with open(os.path.join(output_dir, 'results_summary.json'), 'w', encoding='utf-8') as _f:
        _json.dump(results_json, _f, indent=2, default=str)

    # Resumen en texto
    summary_txt = os.path.join(output_dir, 'results_summary.txt')
    with open(summary_txt, 'w', encoding='utf-8') as f:
        f.write(f"Condición: {condition_name}\n")
        f.write(f"Residualización edad: {'Sí' if apply_resid else 'No'}\n")
        f.write(f"Archivo:   {filepath}\n")
        f.write(f"N sujetos: {n_subjects} ({_group1}={n_pos}, {_group2}={n_subjects-n_pos})\n")
        f.write(f"Features tras pre-filtro: {len(feat_filt)}\n")
        f.write(f"Mejor combo: {best_combo}\n")
        f.write(f"Tiempo total: {elapsed/60:.1f} min\n\n")
        f.write(f"{'Combo':<20} {'AUC':>14} {'F1':>14} {'Recall':>14} {'Brier':>8}\n")
        f.write('-' * 72 + '\n')
        for combo, m in sorted(valid, key=lambda x: x[1]['auc_mean'], reverse=True):
            f.write(f"{combo:<20} "
                    f"{m['auc_mean']:.3f}±{m['auc_std']:.3f}  "
                    f"{m['f1_mean']:.3f}±{m['f1_std']:.3f}  "
                    f"{m['recall_mean']:.3f}±{m['recall_std']:.3f}  "
                    f"{m['brier_mean']:.3f}\n")
        f.write(f"\nBootstrap ({N_BOOTSTRAP} resamples): "
                f"AUC = {bs['auc_mean']:.3f} ± {bs['auc_std']:.3f}\n")
        f.write(f"\nCalibracion OOF: Brier = {calib['brier']:.3f} | "
                f"ECE = {calib['ece']:.3f}\n")
        if noise_rob.get('delta_auc') is not None and not np.isnan(noise_rob['delta_auc']):
            noise_str = '  '.join(
                f"sigma={s:.0%}->{a:.3f}"
                for s, a in zip(noise_rob['levels'], noise_rob['noise_aucs'])
            )
            f.write(f"\nRobustez ruido: {noise_str}\n")
            f.write(f"  DELTA_AUC_max = {noise_rob['delta_auc']:.3f}\n")
        if hp_rob.get('delta_auc') is not None and not np.isnan(hp_rob['delta_auc']):
            hp_str = '  '.join(
                f"x{f}->{a:.3f}"
                for f, a in zip(hp_rob['factors'], hp_rob['hp_aucs'])
            )
            f.write(f"\nSensibilidad HP ({hp_rob['param_name']} base={hp_rob['base_val']:.3g}):\n")
            f.write(f"  {hp_str}\n")
            f.write(f"  DELTA_AUC = {hp_rob['delta_auc']:.3f}\n")

    # Marcar como completado
    open(done_flag, 'w').close()
    logger.info(f"  Condición completada en {elapsed/60:.1f} min → {output_dir}")

    return {
        'summary':     summary,
        'best_combo':  best_combo,
        'bootstrap':   bs,
        'calibration': calib,
        'noise_rob':   noise_rob,
        'hp_rob':      hp_rob,
        'n_subjects':  n_subjects,
        'n_pos':       n_pos,
        'resid':       apply_resid,
        'include_age': include_age,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    results_base = os.path.join(BASE_PATH, 'Resultados')
    global_output = os.path.join(OUTPUT_BASE, f'{GROUP1}_vs_{GROUP2}')
    logger = setup_logging(global_output)

    logger.info("=" * 70)
    logger.info("  ML TRAINING PIPELINE v2")
    logger.info(f"  Comparación: {GROUP1} vs {GROUP2}")
    logger.info(f"  Data type: {DATA_TYPE.upper()} | Space: {SPACE.upper()}")
    logger.info(f"  Nested CV: {N_OUTER_FOLDS}-fold outer × {N_INNER_FOLDS}-fold inner")
    logger.info(f"  Clasificadores: {[c[0] for c in CLASSIFIERS]}")
    logger.info(f"  SMOTE: {APPLY_SMOTE}")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Localizar archivos de datos
    # -------------------------------------------------------------------------
    psm_dir   = os.path.join(results_base, 'PSM_datasets')
    # Preferir archivo HARMONIZED (igual que 2_apply_psm.py) para consistencia:
    # los datasets PSM ya provienen del archivo harmonizado → usar mismo input
    _harmonized = os.path.join(results_base, f'Data_complete_{DATA_TYPE}_{SPACE.lower()}_HARMONIZED.feather')
    _baseline   = os.path.join(results_base, f'Data_complete_{DATA_TYPE}_{SPACE.lower()}.feather')
    no_match    = _harmonized if os.path.exists(_harmonized) else _baseline

    def psm_file(ratio):
        return os.path.join(
            psm_dir,
            f'Data_matched_{DATA_TYPE}_{SPACE.lower()}_{GROUP1}_{ratio}.feather'
        )

    # -------------------------------------------------------------------------
    # Definir las 3 condiciones (+ variantes PSM para psm_residualization)
    #
    #  Condición              │ Datos          │ Age feature │ Residualización
    #  ──────────────────────────────────────────────────────────────────────
    #  covariates_in_model    │ Sin PSM        │  Sí         │  No
    #  residualization        │ Sin PSM        │  No         │  Sí (edad)
    #  psm_1to1_residualization│ PSM 1:1       │  No         │  Sí (edad) ← PRIMARIA
    #  psm_2to1_residualization│ PSM 2:1       │  No         │  Sí (edad)
    #  psm_5to1_residualization│ PSM 5:1       │  No         │  Sí (edad)
    #
    # Cada tupla: (nombre, filepath, apply_resid, include_age)
    # -------------------------------------------------------------------------
    conditions = []

    # 1. covariates_in_model: edad entra como feature directa al modelo
    if os.path.exists(no_match):
        conditions.append(('covariates_in_model', no_match, False, True))
    else:
        logger.warning(f"  covariates_in_model: archivo no encontrado: {no_match}")

    # 2. residualization: edad removida de features EEG dentro de cada fold
    if os.path.exists(no_match):
        conditions.append(('residualization', no_match, True, False))

    # 3. psm_residualization: PSM primary (age) + residualización (uno por ratio)
    for ratio in ['1to1', '2to1', '5to1']:
        pf = psm_file(ratio)
        if os.path.exists(pf):
            conditions.append((f'psm_{ratio}_residualization', pf, True, False))
        else:
            logger.warning(f"  psm_{ratio}_residualization: archivo no encontrado: {pf}")

    # 3b. psm_covariates: PSM primary + age como feature (sin residualizar)
    # Permite comparar si age aporta señal adicional sobre EEG en sample pareado
    for ratio in ['1to1', '2to1', '5to1']:
        pf = psm_file(ratio)
        if os.path.exists(pf):
            conditions.append((f'psm_{ratio}_covariates', pf, False, True))
        else:
            logger.warning(f"  psm_{ratio}_covariates: archivo no encontrado: {pf}")

    # 4. psm_sensitivity: PSM sensitivity (age+education) — datasets de tamaño reducido
    def psm_sensitivity_file(ratio):
        return os.path.join(
            psm_dir,
            f'Data_matched_{DATA_TYPE}_{SPACE.lower()}_{GROUP1}_{ratio}_sensitivity.feather'
        )
    for ratio in ['1to1', '2to1']:   # solo 1:1 y 2:1 son viables con n~31 controles
        pf = psm_sensitivity_file(ratio)
        if os.path.exists(pf):
            conditions.append((f'psm_{ratio}_sensitivity_residualization', pf, True, False))
        else:
            logger.warning(f"  psm_{ratio}_sensitivity: archivo no encontrado: {pf}")

    if not conditions:
        logger.error("No se encontraron archivos de datos. "
                     "Ejecuta 1_make_dataframe.py y 2_apply_psm.py primero.")
        raise SystemExit(1)

    logger.info(f"\nCondiciones a ejecutar ({len(conditions)} total):")
    logger.info(f"  {'Condición':<35} {'Age feature':>12} {'Residualización':>16}")
    logger.info(f"  {'-'*65}")
    for name, _, resid, incl_age in conditions:
        logger.info(f"  {name:<35} {'Sí' if incl_age else 'No':>12} {'Sí' if resid else 'No':>16}")
    logger.info("")

    # -------------------------------------------------------------------------
    # Ejecutar todas las condiciones
    # -------------------------------------------------------------------------
    all_results = {}
    t_total = time.time()

    for condition_name, filepath, apply_resid, include_age in conditions:
        out_dir = os.path.join(global_output, condition_name)
        result  = run_condition(condition_name, filepath, out_dir, logger,
                                apply_resid=apply_resid, include_age=include_age)
        if result is not None:
            all_results[condition_name] = result

    # -------------------------------------------------------------------------
    # Tabla de comparación global
    # -------------------------------------------------------------------------
    comparison_path = os.path.join(global_output, 'comparison_all_conditions.xlsx')
    if all_results:
        save_comparison_table(all_results, comparison_path)
        logger.info(f"\nTabla comparativa guardada: {comparison_path}")

    # -------------------------------------------------------------------------
    # Resumen final
    # -------------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  RESUMEN FINAL — {GROUP1} vs {GROUP2}")
    print(f"{'='*72}")
    print(f"  {'Condición':<18} {'Mejor combo':<18} {'AUC (CV)':>14} {'Bootstrap AUC':>14}")
    print(f"  {'-'*66}")
    for cond, res in all_results.items():
        best = res['best_combo']
        m    = res['summary'][best]
        bs   = res['bootstrap']
        auc_cv  = f"{m['auc_mean']:.3f}±{m['auc_std']:.3f}"
        auc_bs  = f"{bs['auc_mean']:.3f}±{bs['auc_std']:.3f}" \
                  if not np.isnan(bs.get('auc_mean', np.nan)) else 'N/A'
        print(f"  {cond:<18} {best:<18} {auc_cv:>14} {auc_bs:>14}")

    total_min = (time.time() - t_total) / 60
    logger.info(f"\nPipeline v2 completado en {total_min:.1f} min")
    logger.info(f"Todos los resultados en: {global_output}")
