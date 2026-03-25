"""
run_sage_only.py — SAGE analysis on existing final models
==========================================================
Applies SAGE (Shapley Additive Global importancE) to pre-trained models
without re-running the expensive nested CV pipeline.

Output per condition: sage_importance.png + sage_importance.xlsx
"""

import os
import sys
import logging
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

try:
    import sage
except ImportError:
    print("ERROR: sage-importance no instalado. Ejecutar: pip install sage-importance")
    sys.exit(1)

# =============================================================================
# Import config and utilities from 3_train_ml_v2.py via importlib
# (direct import impossible because filename starts with a digit)
# =============================================================================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'train_ml_v2',
    os.path.join(CODE_DIR, '3_train_ml_v2.py')
)
ml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml)

BASE_PATH    = ml.BASE_PATH
DATA_TYPE    = ml.DATA_TYPE
SPACE        = ml.SPACE
GROUP1       = ml.GROUP1
GROUP2       = ml.GROUP2
ML_DIR       = os.path.join(ml.OUTPUT_BASE, f'{GROUP1}_vs_{GROUP2}')
PSM_DIR      = os.path.join(BASE_PATH, 'Resultados', 'PSM_datasets')
RESULTS_DIR  = os.path.join(BASE_PATH, 'Resultados')
RANDOM_STATE = ml.RANDOM_STATE


# =============================================================================
# HELPERS
# =============================================================================

def make_logger():
    log = logging.getLogger('sage_runner')
    log.setLevel(logging.INFO)
    if not log.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        log.addHandler(ch)
    return log


def get_conditions():
    """Return list of (condition_name, filepath, include_age) tuples."""
    harmonized = os.path.join(RESULTS_DIR, f'Data_complete_{DATA_TYPE}_{SPACE}_HARMONIZED.feather')
    baseline   = os.path.join(RESULTS_DIR, f'Data_complete_{DATA_TYPE}_{SPACE}.feather')
    no_match   = harmonized if os.path.exists(harmonized) else baseline

    def psm_file(ratio):
        return os.path.join(PSM_DIR,
                            f'Data_matched_{DATA_TYPE}_{SPACE}_{GROUP1}_{ratio}.feather')

    def psm_sens_file(ratio):
        return os.path.join(PSM_DIR,
                            f'Data_matched_{DATA_TYPE}_{SPACE}_{GROUP1}_{ratio}_sensitivity.feather')

    out = []
    if os.path.exists(no_match):
        out.append(('covariates_in_model', no_match, True))
        out.append(('residualization',     no_match, False))

    for ratio in ['1to1', '2to1', '5to1']:
        pf = psm_file(ratio)
        if os.path.exists(pf):
            out.append((f'psm_{ratio}_residualization', pf, False))
            out.append((f'psm_{ratio}_covariates',      pf, True))

    for ratio in ['1to1', '2to1']:
        pf = psm_sens_file(ratio)
        if os.path.exists(pf):
            out.append((f'psm_{ratio}_sensitivity_residualization', pf, False))

    return out


def prepare_data_for_sage(filepath, model_info, include_age, logger):
    """
    Reconstruct X_fs for a condition using the saved model's preprocessing pipeline.

    Strategy:
      1. Load feather and run load_and_prepare (same logic as training)
      2. Run prefilter_features (same logic as training)
      3. Apply saved imputer → scaler → feature_selector → X_fs
    """
    X_raw, y, feature_names, _, _, _ = ml.load_and_prepare(
        filepath, GROUP1, GROUP2, logger, include_age=include_age
    )
    X_filt, feat_filt = ml.prefilter_features(X_raw, feature_names, logger)

    # Align columns to model's expected feature order
    model_feat = list(model_info['all_feature_names'])
    if list(feat_filt) != model_feat:
        feat_filt_idx = {f: i for i, f in enumerate(feat_filt)}
        missing = [f for f in model_feat if f not in feat_filt_idx]
        if missing:
            print(f"  Warning: {len(missing)} model features not in current prefilter — filling NaN")
        X_aligned = np.full((X_filt.shape[0], len(model_feat)), np.nan)
        for j, fname in enumerate(model_feat):
            idx = feat_filt_idx.get(fname)
            if idx is not None:
                X_aligned[:, j] = X_filt[:, idx]
        X_filt = X_aligned

    # Apply saved pipeline
    X_imp = model_info['imputer'].transform(X_filt)
    X_sc  = model_info['scaler'].transform(X_imp)
    X_fs  = model_info['feature_selector'].transform(X_sc)

    return X_fs, y


def run_sage(model, X_fs, y, feature_names_fs, out_dir, condition_name, clf_name):
    """
    Compute SAGE values, save bar chart (sage_importance.png) and
    Excel table (sage_importance.xlsx). Returns sage_df or None.
    """
    if not hasattr(model, 'predict_proba'):
        print(f"  SAGE: {clf_name} sin predict_proba — omitido")
        return None

    n_bg   = min(512, len(X_fs))
    rng    = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(len(X_fs), size=n_bg, replace=False)
    X_bg   = X_fs[bg_idx]
    y_bg   = np.asarray(y)[bg_idx]

    print(f"  SAGE — {clf_name} | n={n_bg} | detect_convergence=True ...")
    try:
        imputer_s = sage.MarginalImputer(model, X_bg)
        estimator = sage.PermutationEstimator(imputer_s, 'cross entropy')
        sv        = estimator(X_bg, y_bg, detect_convergence=True,
                              verbose=False, bar=False)

        vals    = np.array(sv.values)
        stds    = np.array(sv.std)
        ci_half = 1.96 * stds

        feat_arr = np.array(feature_names_fs)
        order    = np.argsort(vals)[::-1]
        n_disp   = min(20, len(feat_arr))
        top_idx  = order[:n_disp]

        top_feats = feat_arr[top_idx]
        top_vals  = vals[top_idx]
        top_ci    = ci_half[top_idx]

        # Horizontal bar chart with 95% CI
        fig, ax = plt.subplots(figsize=(8, max(4, n_disp * 0.48)))
        colors = ['#d62728' if v > 0 else '#1f77b4' for v in top_vals]
        ax.barh(range(n_disp), top_vals[::-1], xerr=top_ci[::-1],
                color=colors[::-1], edgecolor='white', capsize=3, height=0.7,
                error_kw={'elinewidth': 1.2, 'ecolor': 'black', 'capthick': 1.2})
        ax.set_yticks(range(n_disp))
        ax.set_yticklabels(top_feats[::-1], fontsize=9)
        ax.axvline(0, color='black', lw=0.8, linestyle='--')
        ax.set_xlabel('SAGE Value (contribución a reducción de cross-entropy)', fontsize=9)
        ax.set_title(f'SAGE Feature Importance — {condition_name} ({clf_name})\n'
                     f'Top-{n_disp} features | barras = IC 95%', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'sage_importance.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

        # Excel table with all features
        sage_df = pd.DataFrame({
            'feature':    feat_arr[order],
            'sage_value': vals[order],
            'sage_std':   stds[order],
            'ci_low':     (vals - ci_half)[order],
            'ci_high':    (vals + ci_half)[order],
        })
        sage_df.to_excel(os.path.join(out_dir, 'sage_importance.xlsx'), index=False)

        top3 = sage_df.head(3)['feature'].tolist()
        top3_vals = sage_df.head(3)['sage_value'].values
        print(f"  OK — Top-3: {list(zip(top3, [f'{v:.4f}' for v in top3_vals]))}")
        return sage_df

    except Exception as e:
        import traceback
        print(f"  SAGE falló: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger = make_logger()

    print("=" * 70)
    print("  SAGE ANALYSIS ON EXISTING FINAL MODELS")
    print("=" * 70)

    conditions = get_conditions()
    print(f"\nCondiciones a analizar: {len(conditions)}")
    for cname, fp, ia in conditions:
        model_path = os.path.join(ML_DIR, cname, 'final_model.pkl')
        status = 'OK' if os.path.exists(model_path) else 'MISSING'
        print(f"  [{status}] {cname}")

    summary = {}

    for condition_name, filepath, include_age in conditions:
        print(f"\n{'-'*70}")
        print(f"[{condition_name}]  (include_age={include_age})")

        out_dir    = os.path.join(ML_DIR, condition_name)
        model_path = os.path.join(out_dir, 'final_model.pkl')
        sage_path  = os.path.join(out_dir, 'sage_importance.xlsx')

        if not os.path.exists(model_path):
            print(f"  Sin final_model.pkl — omitido")
            continue

        if os.path.exists(sage_path):
            print(f"  sage_importance.xlsx ya existe — cargando (borrar para re-ejecutar)")
            summary[condition_name] = pd.read_excel(sage_path)
            continue

        model_info = joblib.load(model_path)
        clf_name   = model_info['classifier_name']
        print(f"  Modelo: {clf_name} | {model_info['n_features_used']} features seleccionadas")

        try:
            X_fs, y = prepare_data_for_sage(filepath, model_info, include_age, logger)
            print(f"  X_fs: {X_fs.shape} | {GROUP1}={int(y.sum())} | {GROUP2}={int(len(y)-y.sum())}")
        except Exception as e:
            import traceback
            print(f"  Error preparando datos: {e}")
            traceback.print_exc()
            continue

        sage_df = run_sage(model_info['model'], X_fs, y,
                           model_info['feature_names'],
                           out_dir, condition_name, clf_name)
        if sage_df is not None:
            summary[condition_name] = sage_df

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESUMEN SAGE — TOP-3 POR CONDICIÓN")
    print("=" * 70)
    for cond, df in summary.items():
        top3   = df.head(3)['feature'].tolist()
        vals   = df.head(3)['sage_value'].values
        stds   = df.head(3)['sage_std'].values
        pairs  = [f"{f} ({v:.4f}±{s:.4f})" for f, v, s in zip(top3, vals, stds)]
        print(f"  {cond:<45}  {pairs}")

    print(f"\nSAGE completado. Resultados guardados en:\n  {ML_DIR}")
    return summary


if __name__ == '__main__':
    main()
