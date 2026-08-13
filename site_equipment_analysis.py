"""
site_equipment_analysis.py
==========================
Analiza el impacto del equipo de grabación en el espacio de features EEG
de los portadores PSEN1 E280A (ACr+SCr) de Medellín.

Sites de Medellín:
  Medellín_hd   → alta resolución (equipo principal)
  Medellin_ld   → baja resolución (equipo portátil)
  Medellin_duque → alta resolución, época diferente

Genera en Resultados/experiments/site_equipment_analysis/:
  fig1_pca_by_site.png       — PCA de portadores, coloreado por site
  fig2_features_by_site.png  — distribución de features discriminantes por site
  site_composition.xlsx      — tabla N por site y grupo

Uso:
    python site_equipment_analysis.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import BASE_PATH

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RESULTS_DIR    = os.path.join(BASE_PATH, 'Resultados')
DATA_HARM      = os.path.join(RESULTS_DIR, 'Data_complete_ce_roi_HARMONIZED.feather')
DATA_RAW       = os.path.join(RESULTS_DIR, 'Data_complete_ce_roi.feather')
EXPERIMENTS_DIR = os.path.join(RESULTS_DIR, 'experiments')
OUTPUT_DIR     = os.path.join(EXPERIMENTS_DIR, 'site_equipment_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAGE_PRIORITY = ['E05', 'E32', 'E28', 'E03']  # busca SAGE en este orden

# ---------------------------------------------------------------------------
# Paleta de sites de Medellín
# ---------------------------------------------------------------------------
SITE_COLOR = {
    'Medellín_hd':    '#1565C0',   # azul oscuro
    'Medellin_hd':    '#1565C0',
    'Medellin_ld':    '#E53935',   # rojo
    'Medellin_duque': '#7B1FA2',   # morado
}
SITE_LABEL = {
    'Medellín_hd':    'Medellín HD',
    'Medellin_hd':    'Medellín HD',
    'Medellin_ld':    'Medellín LD (portátil)',
    'Medellin_duque': 'Medellín Duque',
}
SITE_MARKER = {
    'Medellín_hd':    'o',
    'Medellin_hd':    'o',
    'Medellin_ld':    's',
    'Medellin_duque': '^',
}
MEDELLIN_SITES = list(SITE_COLOR.keys())

GROUP_COLOR = {'ACr': '#FF8F00', 'SCr': '#C62828'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data():
    path = DATA_HARM if os.path.exists(DATA_HARM) else DATA_RAW
    print(f"Cargando: {os.path.basename(path)}")
    return pd.read_feather(path)


def get_feature_cols(data):
    exclude = {
        'subject', 'group', 'Task', 'ses', 'mmse', 'moca', 'SITE', 'sex',
        'age', 'education', 'orig_group',
        'group_sl', 'age_sl', 'SITE_sl',
        'group_coh', 'age_coh', 'SITE_coh',
        'group_ent', 'age_ent', 'SITE_ent',
    }
    return [c for c in data.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(data[c])]


def aggregate_subjects(data, feature_cols):
    """Agrega a nivel sujeto (media). Preserva SITE, group, orig_group, age, sex."""
    feats = data.groupby('subject')[feature_cols].mean().reset_index()
    meta_cols = ['subject', 'group', 'SITE']
    for c in ['orig_group', 'age', 'sex']:
        if c in data.columns:
            meta_cols.append(c)
    meta = data.groupby('subject')[meta_cols[1:]].first().reset_index()
    return feats.merge(meta, on='subject')


def load_sage_features(n=10):
    """Carga las top N features de SAGE del mejor experimento disponible."""
    for exp_id in SAGE_PRIORITY:
        sage_path = os.path.join(EXPERIMENTS_DIR, exp_id, 'sage_importance.xlsx')
        if os.path.exists(sage_path):
            df = pd.read_excel(sage_path)
            print(f"  SAGE cargado de {exp_id}: {len(df)} features")
            return df.head(n)['feature'].tolist(), exp_id
    print("  SAGE no disponible — se usarán loadings de PCA")
    return None, None


def normalize_site(site):
    """Normaliza variaciones de nombre de site."""
    if pd.isna(site):
        return 'other'
    s = str(site).strip()
    if s in ('Medellín_hd', 'Medellin_hd'):
        return 'Medellin_hd'
    return s


# ---------------------------------------------------------------------------
# Figura 1: PCA de portadores coloreado por site
# ---------------------------------------------------------------------------

def fig1_pca_carriers(agg, feature_cols, output_dir):
    carriers = agg[agg['orig_group'].isin(['ACr', 'GG', 'G1', 'SCr'])].copy()
    if carriers.empty:
        carriers = agg[agg['group'].isin(['PSEN1', 'SCr', 'AllCarriers'])].copy()

    carriers['site_norm'] = carriers['SITE'].apply(normalize_site)
    carriers['orig_label'] = carriers['orig_group'].apply(
        lambda x: 'SCr' if x == 'SCr' else 'ACr'
    )

    X = carriers[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    sc = StandardScaler()
    X_s = sc.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_s)
    carriers = carriers.copy()
    carriers['PC1'] = coords[:, 0]
    carriers['PC2'] = coords[:, 1]

    var1, var2 = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Espacio PCA de portadores PSEN1 E280A por equipo de grabación',
                 fontsize=13, fontweight='bold')

    # Panel izquierdo: coloreado por site
    ax = axes[0]
    ax.set_title('Coloreado por site (equipo)', fontsize=11)
    medellin_sites_present = [s for s in ['Medellin_hd', 'Medellin_ld', 'Medellin_duque']
                               if s in carriers['site_norm'].values]

    for site in medellin_sites_present:
        sub = carriers[carriers['site_norm'] == site]
        ax.scatter(sub['PC1'], sub['PC2'],
                   c=SITE_COLOR.get(site, '#888'),
                   marker=SITE_MARKER.get(site, 'o'),
                   s=70, alpha=0.75, edgecolors='white', linewidth=0.4,
                   label=SITE_LABEL.get(site, site))

    # Agregar centroide por site
    for site in medellin_sites_present:
        sub = carriers[carriers['site_norm'] == site]
        cx, cy = sub['PC1'].mean(), sub['PC2'].mean()
        ax.scatter(cx, cy, c=SITE_COLOR.get(site, '#888'),
                   marker='*', s=300, edgecolors='black', linewidth=0.8, zorder=5)

    ax.set_xlabel(f'PC1 ({var1:.1f}% var)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var2:.1f}% var)', fontsize=10)
    ax.legend(fontsize=9, title='Site', title_fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel derecho: coloreado por grupo (ACr vs SCr), marcador por site
    ax = axes[1]
    ax.set_title('Coloreado por grupo (ACr/SCr)', fontsize=11)
    for orig_lbl, color in GROUP_COLOR.items():
        for site in medellin_sites_present:
            sub = carriers[(carriers['orig_label'] == orig_lbl) &
                           (carriers['site_norm'] == site)]
            if sub.empty:
                continue
            ax.scatter(sub['PC1'], sub['PC2'],
                       c=color, marker=SITE_MARKER.get(site, 'o'),
                       s=70, alpha=0.75, edgecolors='white', linewidth=0.4)

    # Leyenda compuesta
    group_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLOR.items()]
    site_handles = [
        plt.Line2D([0], [0], marker=SITE_MARKER.get(s, 'o'), color='grey',
                   linestyle='None', markersize=8,
                   label=SITE_LABEL.get(s, s))
        for s in medellin_sites_present
    ]
    ax.legend(handles=group_patches + site_handles, fontsize=8,
              title='Grupo / Site', title_fontsize=9)
    ax.set_xlabel(f'PC1 ({var1:.1f}% var)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var2:.1f}% var)', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, 'fig1_pca_by_site.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {os.path.basename(out)}")

    # Tabla de varianza explicada
    loadings = pd.DataFrame(pca.components_.T, index=feature_cols,
                            columns=['PC1', 'PC2'])
    top_load = loadings.abs().max(axis=1).nlargest(15).index.tolist()
    loadings.loc[top_load].to_excel(
        os.path.join(output_dir, 'pca_top_loadings.xlsx'))


# ---------------------------------------------------------------------------
# Figura 2: Distribución de features discriminantes por site (SCr)
# ---------------------------------------------------------------------------

def fig2_feature_distributions(agg, feature_cols, output_dir):
    top_features, sage_exp = load_sage_features(n=8)

    if top_features is None:
        # Fallback: usar las 8 features con mayor varianza en portadores
        carriers = agg[agg['orig_group'].isin(['ACr', 'GG', 'G1', 'SCr'])]
        if carriers.empty:
            print("  No hay portadores con orig_group — saltando fig2")
            return
        var = carriers[feature_cols].var().nlargest(8)
        top_features = var.index.tolist()
        sage_exp = 'varianza'

    # Filtrar solo SCr de Medellín para comparar equipos
    scr = agg[agg['orig_group'] == 'SCr'].copy()
    if scr.empty:
        print("  No hay SCr con orig_group — saltando fig2")
        return

    scr['site_norm'] = scr['SITE'].apply(normalize_site)
    medellin_scr = scr[scr['site_norm'].isin(['Medellin_hd', 'Medellin_ld', 'Medellin_duque'])]

    if medellin_scr.empty:
        print("  No hay SCr de Medellín — saltando fig2")
        return

    # Filtrar features que existen en el dataframe
    top_features = [f for f in top_features if f in agg.columns]
    if not top_features:
        print("  Features SAGE no encontradas en el feather — saltando fig2")
        return

    n_feats = min(8, len(top_features))
    ncols = 4
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = axes.flatten() if n_feats > 1 else [axes]

    site_order = [s for s in ['Medellin_hd', 'Medellin_ld', 'Medellin_duque']
                  if s in medellin_scr['site_norm'].values]
    palette = {s: SITE_COLOR[s] for s in site_order}

    for i, feat in enumerate(top_features[:n_feats]):
        ax = axes[i]
        sns.boxplot(data=medellin_scr, x='site_norm', y=feat, hue='site_norm',
                    order=site_order, palette=palette, ax=ax,
                    width=0.5, fliersize=3, legend=False)
        sns.stripplot(data=medellin_scr, x='site_norm', y=feat, hue='site_norm',
                      order=site_order, palette=palette, ax=ax,
                      size=4, alpha=0.6, jitter=True, legend=False)
        short_name = feat.split('_')[0] + '_' + '_'.join(feat.split('_')[1:3])
        ax.set_title(short_name, fontsize=8, pad=3)
        ax.set_xlabel('')
        ax.set_ylabel('')
        tick_labels = [SITE_LABEL.get(s, s).replace(' (portátil)', '\n(portátil)')
                       for s in site_order]
        ax.set_xticks(range(len(site_order)))
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=15, ha='right')

    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)

    src_str = f'SAGE de {sage_exp}' if sage_exp not in ('varianza', None) else sage_exp
    fig.suptitle(f'Top features discriminantes por equipo — SCr Medellín\n(fuente: {src_str})',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(output_dir, 'fig2_features_by_site.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardado: {os.path.basename(out)}")


# ---------------------------------------------------------------------------
# Tabla de composición por site
# ---------------------------------------------------------------------------

def save_composition_table(agg, output_dir):
    rows = []
    orig_col = 'orig_group' if 'orig_group' in agg.columns else 'group'

    for site, grp in agg.groupby('SITE'):
        site_norm = normalize_site(site)
        for og, cnt in grp[orig_col].value_counts().items():
            rows.append({
                'SITE': site,
                'site_normalizado': site_norm,
                'orig_group': og,
                'N': cnt,
            })

    df = pd.DataFrame(rows).sort_values(['SITE', 'orig_group'])
    out = os.path.join(output_dir, 'site_composition.xlsx')
    df.to_excel(out, index=False)
    print(f"  Guardado: {os.path.basename(out)}")

    # Resumen Medellín para pantalla
    print("\n  === Composición Medellín (sujetos por site y grupo) ===")
    medd = df[df['site_normalizado'].isin(['Medellin_hd', 'Medellin_ld', 'Medellin_duque'])]
    pivot = medd.pivot_table(index='site_normalizado', columns='orig_group',
                              values='N', aggfunc='sum', fill_value=0)
    print(pivot.to_string())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== site_equipment_analysis.py ===\n")

    data = load_data()
    feature_cols = get_feature_cols(data)
    print(f"  Features EEG disponibles: {len(feature_cols)}")
    print(f"  Sujetos únicos (filas): {data['subject'].nunique()}")

    agg = aggregate_subjects(data, feature_cols)
    print(f"  Sujetos tras agregación: {len(agg)}")

    print("\n[1/3] Tabla de composición por site...")
    save_composition_table(agg, OUTPUT_DIR)

    print("[2/3] Figura 1: PCA portadores por site...")
    fig1_pca_carriers(agg, feature_cols, OUTPUT_DIR)

    print("[3/3] Figura 2: Distribución features discriminantes...")
    fig2_feature_distributions(agg, feature_cols, OUTPUT_DIR)

    print(f"\nListo. Resultados en:\n  {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
