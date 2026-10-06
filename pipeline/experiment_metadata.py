"""
experiment_metadata.py
======================
Guarda una caracterización completa de los datos usados en cada experimento:
  - experiment_metadata.json : N por grupo, distribución por sitio, edad, sexo,
                               lista completa de sujetos usados.
  - subjects_used.csv        : una fila por sujeto, fácil de inspeccionar.

Sin dependencias del pipeline principal → importable desde cualquier script.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _site_counts(df):
    if 'SITE' in df.columns:
        return df['SITE'].value_counts().to_dict()
    return {}


def _age_stats(df):
    if 'age' not in df.columns:
        return {}
    ages = pd.to_numeric(df['age'], errors='coerce').dropna()
    if len(ages) == 0:
        return {}
    return {
        'mean': round(float(ages.mean()), 2),
        'std':  round(float(ages.std()),  2),
        'min':  round(float(ages.min()),  2),
        'max':  round(float(ages.max()),  2),
    }


def _sex_stats(df):
    if 'sex' not in df.columns:
        return {}
    total = len(df)
    if total == 0:
        return {}
    female = df['sex'].astype(str).str.upper().isin(['F', 'FEMALE', '0', '0.0'])
    n_f = int(female.sum())
    return {
        'n_female':   n_f,
        'n_male':     total - n_f,
        'pct_female': round(n_f / total, 3),
    }


def _orig_groups(df):
    if 'orig_group' in df.columns:
        return df['orig_group'].value_counts().to_dict()
    return {}


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def compute_experiment_metadata(data_agg, exp_config):
    """
    Calcula el diccionario de metadata a partir del DataFrame agregado
    (una fila por sujeto) y la config del experimento.

    data_agg  : DataFrame con columnas 'group', 'subject', opcionalmente
                'SITE', 'age', 'sex', 'orig_group'.
    exp_config: dict del registry (id, name, case_label, control_label, …).

    Retorna dict listo para serializar como JSON.
    """
    case_label = exp_config['case_label']
    ctrl_label = exp_config['control_label']

    df_case = data_agg[data_agg['group'] == case_label].copy()
    df_ctrl = data_agg[data_agg['group'] == ctrl_label].copy()

    metadata = {
        # --- Identificación del experimento ---
        'experiment_id':   exp_config.get('id', 'unknown'),
        'experiment_name': exp_config.get('name', ''),
        'comparison':      exp_config.get('comparison', f"{case_label}_vs_{ctrl_label}"),
        'stage':           exp_config.get('stage', ''),
        'description':     exp_config.get('description', ''),
        'run_timestamp':   datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),

        # --- Configuración del experimento ---
        'data_file':    os.path.basename(exp_config.get('data_file', '')),
        'balancing':    exp_config.get('balancing', 'unknown'),
        'smote':        exp_config.get('smote', False),
        'age_strategy': exp_config.get('age_strategy', 'unknown'),
        'case_orig_labels_filter':    exp_config.get('case_orig_labels'),
        'control_orig_labels_filter': exp_config.get('control_orig_labels'),

        # --- Tamaños de muestra ---
        'n_total':   len(data_agg),
        'n_case':    len(df_case),
        'n_control': len(df_ctrl),

        # --- Grupos originales (antes del mapeo) ---
        'case_label':          case_label,
        'control_label':       ctrl_label,
        'case_orig_groups':    _orig_groups(df_case),
        'control_orig_groups': _orig_groups(df_ctrl),

        # --- Distribución por sitio ---
        'case_by_site':    _site_counts(df_case),
        'control_by_site': _site_counts(df_ctrl),

        # --- Estadísticas de edad ---
        'case_age':    _age_stats(df_case),
        'control_age': _age_stats(df_ctrl),

        # --- Estadísticas de sexo ---
        'case_sex':    _sex_stats(df_case),
        'control_sex': _sex_stats(df_ctrl),

        # --- Lista completa de sujetos (reproducibilidad) ---
        'subjects_case':    sorted(df_case['subject'].tolist()),
        'subjects_control': sorted(df_ctrl['subject'].tolist()),
    }
    return metadata


def save_experiment_metadata(data_agg, exp_config, output_dir):
    """
    Llama a compute_experiment_metadata() y guarda:
      - experiment_metadata.json
      - subjects_used.csv

    Retorna el dict de metadata calculado.
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata = compute_experiment_metadata(data_agg, exp_config)

    # --- JSON ---
    json_path = os.path.join(output_dir, 'experiment_metadata.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    # --- CSV de sujetos ---
    case_label = exp_config['case_label']
    ctrl_label = exp_config['control_label']
    df_case = data_agg[data_agg['group'] == case_label]
    df_ctrl = data_agg[data_agg['group'] == ctrl_label]

    rows = []
    for role, df_part in [('case', df_case), ('control', df_ctrl)]:
        for _, row in df_part.iterrows():
            rows.append({
                'subject':    row['subject'],
                'role':       role,
                'group':      row.get('group', ''),
                'orig_group': row.get('orig_group', ''),
                'SITE':       row.get('SITE', ''),
                'age':        row.get('age', ''),
                'sex':        row.get('sex', ''),
            })

    csv_path = os.path.join(output_dir, 'subjects_used.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return metadata


def print_metadata_summary(metadata):
    """Imprime un resumen compacto de la metadata (para logging)."""
    n_case = metadata['n_case']
    n_ctrl = metadata['n_control']
    sites_c = metadata.get('case_by_site', {})
    sites_k = metadata.get('control_by_site', {})
    age_c   = metadata.get('case_age', {})
    age_k   = metadata.get('control_age', {})

    print(f"  Sujetos: {n_case} caso / {n_ctrl} control  (total={n_case+n_ctrl})")

    if sites_c:
        top = sorted(sites_c.items(), key=lambda x: -x[1])
        print(f"  Sitios caso   : " + " | ".join(f"{s}={n}" for s, n in top))
    if sites_k:
        top = sorted(sites_k.items(), key=lambda x: -x[1])
        print(f"  Sitios control: " + " | ".join(f"{s}={n}" for s, n in top))
    if age_c:
        print(f"  Edad caso   : {age_c.get('mean','?')} ± {age_c.get('std','?')}"
              f"  [{age_c.get('min','?')} – {age_c.get('max','?')}]")
    if age_k:
        print(f"  Edad control: {age_k.get('mean','?')} ± {age_k.get('std','?')}"
              f"  [{age_k.get('min','?')} – {age_k.get('max','?')}]")

    orig_c = metadata.get('case_orig_groups', {})
    orig_k = metadata.get('control_orig_groups', {})
    if orig_c:
        print(f"  Orig grupos caso   : {orig_c}")
    if orig_k:
        print(f"  Orig grupos control: {orig_k}")
