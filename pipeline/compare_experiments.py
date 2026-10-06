"""
compare_experiments.py
======================
Lee todos los resultados guardados en Resultados/experiments/ y genera
una tabla maestra de comparación en Excel.

Uso:
    python compare_experiments.py                    # usa directorio por defecto
    python compare_experiments.py --dir /otro/path   # directorio custom

Archivos que lee por experimento:
    experiment_metadata.json   → N, sitios, edad, sexo
    results_summary.json       → AUC, F1, Brier, bootstrap

Salida:
    Resultados/experiments/master_comparison.xlsx
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from config import BASE_PATH
EXPERIMENTS_OUT = os.path.join(BASE_PATH, 'Resultados', 'experiments')


# ---------------------------------------------------------------------------

def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _sites_str(site_dict, top_n=4):
    """Convierte dict de sitios en string compacto, ej. 'Seoul=286|Dortmund=195'."""
    if not site_dict:
        return ''
    items = sorted(site_dict.items(), key=lambda x: -x[1])[:top_n]
    return '|'.join(f"{s}={n}" for s, n in items)


def _best_combo_metrics(results_json):
    """Extrae métricas del mejor combo del results_summary.json."""
    best = results_json.get('best_combo', '')
    combos = results_json.get('combos', {})
    if not best or best not in combos:
        return {}
    return combos[best]


def build_master_comparison(experiments_dir=None):
    """
    Lee todos los subdirectorios de experiments_dir, carga metadata y resultados,
    y genera master_comparison.xlsx con una fila por experimento.

    Retorna el DataFrame generado.
    """
    experiments_dir = experiments_dir or EXPERIMENTS_OUT

    if not os.path.isdir(experiments_dir):
        print(f"[AVISO] Directorio no encontrado: {experiments_dir}")
        return pd.DataFrame()

    rows = []

    for entry in sorted(os.listdir(experiments_dir)):
        exp_dir = os.path.join(experiments_dir, entry)
        if not os.path.isdir(exp_dir):
            continue

        meta_path    = os.path.join(exp_dir, 'experiment_metadata.json')
        results_path = os.path.join(exp_dir, 'results_summary.json')

        # Necesita al menos uno de los dos
        if not os.path.exists(meta_path) and not os.path.exists(results_path):
            continue

        meta    = _load_json(meta_path)
        results = _load_json(results_path)

        # ── Identidad del experimento ──────────────────────────────────────
        row = {
            'exp_id':       meta.get('experiment_id', results.get('experiment_id', entry)),
            'exp_name':     meta.get('experiment_name', results.get('condition_name', entry)),
            'stage':        meta.get('stage', ''),
            'comparison':   meta.get('comparison', ''),
            'description':  meta.get('description', '')[:80],
        }

        # ── Configuración ──────────────────────────────────────────────────
        row.update({
            'balancing':    meta.get('balancing',    results.get('apply_smote', '')),
            'smote':        meta.get('smote',        results.get('apply_smote', '')),
            'age_strategy': meta.get('age_strategy', ''),
            'data_file':    meta.get('data_file',    ''),
            'case_label':   meta.get('case_label',   results.get('group1', '')),
            'ctrl_label':   meta.get('control_label',results.get('group2', '')),
        })

        # ── Tamaños de muestra ─────────────────────────────────────────────
        row.update({
            'n_case':    meta.get('n_case',    results.get('n_case',    '')),
            'n_control': meta.get('n_control', results.get('n_control', '')),
            'n_total':   meta.get('n_total',   results.get('n_subjects','')),
        })

        # ── Grupos originales ──────────────────────────────────────────────
        orig_case = meta.get('case_orig_groups', {})
        orig_ctrl = meta.get('control_orig_groups', {})
        row['case_orig_groups'] = ', '.join(f"{k}={v}" for k, v in orig_case.items())
        row['ctrl_orig_groups'] = ', '.join(f"{k}={v}" for k, v in orig_ctrl.items())

        # ── Sitios ────────────────────────────────────────────────────────
        row['case_sites']    = _sites_str(meta.get('case_by_site', {}))
        row['control_sites'] = _sites_str(meta.get('control_by_site', {}))
        row['n_sites_ctrl']  = len(meta.get('control_by_site', {}))

        # ── Edad ──────────────────────────────────────────────────────────
        age_c = meta.get('case_age', {})
        age_k = meta.get('control_age', {})
        row['age_case_mean']    = age_c.get('mean', '')
        row['age_case_std']     = age_c.get('std',  '')
        row['age_ctrl_mean']    = age_k.get('mean', '')
        row['age_ctrl_std']     = age_k.get('std',  '')

        # ── Sexo ──────────────────────────────────────────────────────────
        sex_c = meta.get('case_sex', {})
        sex_k = meta.get('control_sex', {})
        row['pct_female_case'] = sex_c.get('pct_female', '')
        row['pct_female_ctrl'] = sex_k.get('pct_female', '')

        # ── Rendimiento del mejor combo ────────────────────────────────────
        best_metrics = _best_combo_metrics(results)
        row['best_combo']      = results.get('best_combo', '')
        row['AUC_mean']        = best_metrics.get('auc_mean',       '')
        row['AUC_std']         = best_metrics.get('auc_std',        '')
        row['F1_mean']         = best_metrics.get('f1_mean',        '')
        row['F1_std']          = best_metrics.get('f1_std',         '')
        row['Recall_mean']     = best_metrics.get('recall_mean',    '')
        row['Precision_mean']  = best_metrics.get('precision_mean', '')
        row['Brier_mean']      = best_metrics.get('brier_mean',     '')

        # ── Bootstrap ─────────────────────────────────────────────────────
        bs = results.get('bootstrap', {})
        row['bootstrap_AUC_mean'] = bs.get('auc_mean', '')
        row['bootstrap_AUC_std']  = bs.get('auc_std',  '')

        # ── Calibración ───────────────────────────────────────────────────
        calib = results.get('calibration', {})
        row['Brier_OOF'] = calib.get('brier', '')
        row['ECE_OOF']   = calib.get('ece',   '')

        # ── Tiempo ────────────────────────────────────────────────────────
        row['elapsed_min'] = results.get('elapsed_min', '')

        rows.append(row)

    if not rows:
        print("  [AVISO] No se encontraron resultados en el directorio.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Convertir columnas numéricas (pueden llegar como '' cuando no hay resultados aún)
    numeric_cols = [
        'n_case', 'n_control', 'n_total',
        'age_case_mean', 'age_case_std', 'age_ctrl_mean', 'age_ctrl_std',
        'pct_female_case', 'pct_female_ctrl', 'n_sites_ctrl',
        'AUC_mean', 'AUC_std', 'F1_mean', 'F1_std',
        'Recall_mean', 'Precision_mean', 'Brier_mean',
        'bootstrap_AUC_mean', 'bootstrap_AUC_std',
        'Brier_OOF', 'ECE_OOF', 'elapsed_min',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ordenar: por AUC descendente si disponible
    if 'AUC_mean' in df.columns:
        df = df.sort_values('AUC_mean', ascending=False, na_position='last')

    # Guardar
    out_path = os.path.join(experiments_dir, 'master_comparison.xlsx')
    df.to_excel(out_path, index=False)
    print(f"\n  [OK] Tabla maestra guardada: {out_path}")
    print(f"       {len(df)} experimentos, {len(df.columns)} columnas\n")

    # Resumen en consola
    cols_show = ['exp_id', 'exp_name', 'n_case', 'n_control',
                 'balancing', 'AUC_mean', 'AUC_std', 'bootstrap_AUC_mean']
    cols_avail = [c for c in cols_show if c in df.columns]
    print(df[cols_avail].to_string(index=False))

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Genera tabla maestra de comparación entre experimentos.'
    )
    parser.add_argument(
        '--dir', default=EXPERIMENTS_OUT,
        help=f'Directorio con resultados de experimentos. Default: {EXPERIMENTS_OUT}'
    )
    args = parser.parse_args()
    build_master_comparison(args.dir)
