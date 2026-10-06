"""
Global EEG Harmonization — Reference-Based Two-Step Approach
============================================================
Step 1: Learn site effects from CONTROLS ONLY (harmonizationLearn)
Step 2: Apply the learned correction to ALL subjects (harmonizationApply)

This ensures that biological signal from PSEN1 carriers does NOT
contaminate the estimation of the Medellin site effect, which would
happen with standard ComBat applied to all subjects jointly.

Input  : One complete dataframe
Output : One complete harmonized dataframe (all metrics)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from neuroHarmonize import harmonizationLearn, harmonizationApply
import pickle

from utils import (
    mapsDrop, covars, select, extract_components_interes,
    negativeTest
)

# ============================================================================
# CONFIGURATION
# ============================================================================

from config import BASE_PATH as _BASE
PATH = os.path.join(_BASE, 'Resultados')
SPACE = 'roi'          # 'roi' or 'ic'
DATA_TYPE = 'ce'
ICA = '54x10'

INPUT_FILE = os.path.join(PATH, f'Data_complete_{DATA_TYPE}_{SPACE}.feather')
OUTPUT_FILE = os.path.join(
    PATH,
    f'Data_complete_{DATA_TYPE}_{SPACE}_HARMONIZED.feather'
)
MODEL_PATH = os.path.join(PATH, 'harmonization_models')
os.makedirs(MODEL_PATH, exist_ok=True)

METRICS = ['power', 'sl', 'cohfreq', 'entropy', 'crossfreq']

COMPONENTS = {
    '54x10': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
    '58x25': ['C14', 'C15', 'C18', 'C20', 'C22', 'C23', 'C24', 'C25']
}

# Columns that are covariates/metadata — must not be harmonized
COVARIATE_COLS = ['age', 'sex', 'gender', 'SITE', 'education', 'education_cross',
                  'diagnosis', 'subject_id', 'subject', 'session', 'database', 'group']

# Metadata duplicates produced by each metric's dataframe (group_sl, age_coh, etc.)
# These must be excluded from harmonization — they are not EEG features.
METRIC_METADATA = {
    'power':    [],
    'sl':       ['group_sl',    'age_sl',    'SITE_sl'],
    'cohfreq':  ['group_coh',   'age_coh',   'SITE_coh'],
    'entropy':  ['group_ent',   'age_ent',   'SITE_ent'],
    'crossfreq': ['group_cross', 'age_cross', 'SITE_cross'],
}


def get_metric_feature_cols(data_cols, metric):
    """
    Return only the true feature columns for a given metric type.
    Column naming conventions:
      power    — channel_band format, no suffix, no '/'  (e.g. C3_Alpha-1)
      sl       — ends with '_sl'                          (e.g. C3_Alpha-1_sl)
      cohfreq  — ends with '_coh'                         (e.g. C3_Alpha-1_coh)
      entropy  — ends with '_ent'                         (e.g. C3_Alpha-1_ent)
      crossfreq— contains '/'                             (e.g. C3_Alpha-1/MAlpha-1)
    """
    exclude = set(COVARIATE_COLS) | set(METRIC_METADATA.get(metric, []))
    # also exclude ALL metric-metadata variants regardless of current metric
    for meta_list in METRIC_METADATA.values():
        exclude.update(meta_list)

    all_suffixes = ('_sl', '_coh', '_ent', '_cross')

    if metric == 'power':
        return [c for c in data_cols
                if c not in exclude
                and '/' not in c
                and not any(c.endswith(s) for s in all_suffixes)]
    elif metric == 'sl':
        return [c for c in data_cols if c not in exclude and c.endswith('_sl')]
    elif metric == 'cohfreq':
        return [c for c in data_cols if c not in exclude and c.endswith('_coh')]
    elif metric == 'entropy':
        return [c for c in data_cols if c not in exclude and c.endswith('_ent')]
    elif metric == 'crossfreq':
        return [c for c in data_cols if c not in exclude and '/' in c]
    return []


# ============================================================================
# FUNCTIONS
# ============================================================================

def encode_covars(covars_df, site_mapping):
    """
    Encode covariate DataFrame for neuroHarmonize:
    - SITE: integer codes (consistent mapping across reference and full data)
    - age: float
    - sex: float (already encoded by utils.covars)
    """
    df = covars_df.copy()

    if 'SITE' in df.columns:
        df['SITE'] = df['SITE'].map(site_mapping)

    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

    if 'education' in df.columns:
        df['education'] = pd.to_numeric(df['education'], errors='coerce')

    return df


def harmonize_block_reference(
    block_ref, covars_ref_df,
    block_all, covars_all_df,
    covariate_cols=COVARIATE_COLS
):
    """
    Reference-based harmonization using two steps:
      1. harmonizationLearn on controls only  -> learns site effects
      2. harmonizationApply on all subjects   -> applies correction

    Parameters
    ----------
    block_ref : pd.DataFrame
        Feature block for reference population (Controls only).
    covars_ref_df : pd.DataFrame
        Covariates for reference population (encoded).
    block_all : pd.DataFrame
        Feature block for all subjects.
    covars_all_df : pd.DataFrame
        Covariates for all subjects (same encoding).

    Returns
    -------
    df_adj : pd.DataFrame
        Harmonized features for ALL subjects (index matches block_all).
    model : dict
        neuroHarmonize model learned from controls.
    """
    # Identify columns to harmonize from reference block
    numeric_cols = block_ref.select_dtypes(include=[np.number]).columns
    cols_to_harmonize = [c for c in numeric_cols if c not in covariate_cols]

    if len(cols_to_harmonize) == 0:
        print("  Warning: No columns to harmonize after excluding covariates")
        return pd.DataFrame(index=block_all.index), None

    print(f"  Harmonizing {len(cols_to_harmonize)} features "
          f"(reference n={len(block_ref)}, apply n={len(block_all)})...")

    # Extract numeric data
    X_ref = block_ref[cols_to_harmonize].values
    X_all = block_all[cols_to_harmonize].values

    # Log transform (add small constant to avoid log(0))
    X_ref_log = np.log(0.001 + X_ref)
    X_all_log = np.log(0.001 + X_all)

    # Replace any non-finite values
    if np.any(~np.isfinite(X_ref_log)):
        print("  Warning: Non-finite values in reference — replacing with 0")
        X_ref_log = np.nan_to_num(X_ref_log, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(~np.isfinite(X_all_log)):
        print("  Warning: Non-finite values in full data — replacing with 0")
        X_all_log = np.nan_to_num(X_all_log, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 1: Learn site effects from controls only
    try:
        model, _ = harmonizationLearn(X_ref_log, covars_ref_df)
    except Exception as e:
        print(f"  Error during harmonizationLearn: {e}")
        return pd.DataFrame(index=block_all.index), None

    # Step 2: Apply learned correction to all subjects
    try:
        X_all_adj = harmonizationApply(X_all_log, covars_all_df, model)
    except Exception as e:
        print(f"  Error during harmonizationApply: {e}")
        return pd.DataFrame(index=block_all.index), None

    # Back-transform (inverse of log)
    X_all_adj = np.exp(X_all_adj) - 0.001

    # Clip any residual negatives
    n_neg = negativeTest(X_all_adj)
    if n_neg > 0:
        print(f"  Warning: {n_neg} negative values clipped to 0 after back-transform")
    X_all_adj = np.maximum(X_all_adj, 0)

    df_adj = pd.DataFrame(X_all_adj, columns=cols_to_harmonize, index=block_all.index)
    return df_adj, model


def plot_combat_parameters(models, site_mapping, output_path):
    """
    Plot distributions of ComBat gamma (additive) and delta (multiplicative)
    parameters across features, one curve per site.

    gamma_star: additive site offset — values near 0 mean no additive correction.
    delta_star: multiplicative site scaling of variance — values near 1 mean no
                scaling correction. Plotted in log scale (log(delta)) so the
                symmetric reference is 0.

    One figure is produced per metric type and saved to output_path.

    Parameters
    ----------
    models : dict  {metric_name: neuroHarmonize_model_dict}
    site_mapping : dict  {site_name: integer_code}
    output_path : str
    """
    # Reverse mapping: integer code -> site name
    code_to_site = {v: k for k, v in site_mapping.items()}

    for metric, model in models.items():
        if model is None:
            continue

        gamma = model.get('gamma_star')   # shape: (n_sites, n_features)
        delta = model.get('delta_star')   # shape: (n_sites, n_features)
        batch_levels = model.get('info_dict', {}).get('batch_levels', [])

        if gamma is None or delta is None:
            continue

        n_sites = gamma.shape[0]
        site_labels = [code_to_site.get(int(b), f'Site {b}') for b in batch_levels]

        # Color palette — one color per site
        cmap = plt.cm.get_cmap('tab10', n_sites)
        colors = [cmap(i) for i in range(n_sites)]

        label = 'All EEG features' if metric == 'all_features' else metric.upper()
        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(
            f'ComBat site-effect parameters — {label}\n'
            f'(estimated from Controls only, reference-based harmonization)',
            fontsize=12, fontweight='bold', y=0.98
        )
        gs = gridspec.GridSpec(1, 2, figure=fig)

        # --- Left panel: gamma (additive effects) ---
        ax_g = fig.add_subplot(gs[0])
        for idx in range(n_sites):
            vals = gamma[idx, :]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 3:
                continue
            xs = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 300)
            try:
                kde = gaussian_kde(vals, bw_method='scott')
                ax_g.plot(xs, kde(xs), color=colors[idx],
                          label=site_labels[idx], linewidth=1.8)
            except Exception:
                pass

        ax_g.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax_g.set_xlabel('gamma* (additive site effect)', fontsize=11)
        ax_g.set_ylabel('Density', fontsize=11)
        ax_g.set_title('Additive effects (gamma*)', fontsize=11)
        ax_g.legend(fontsize=7, loc='upper right', framealpha=0.7)
        ax_g.grid(True, alpha=0.3)

        # --- Right panel: log(delta) (multiplicative / variance scaling) ---
        ax_d = fig.add_subplot(gs[1])
        for idx in range(n_sites):
            vals = delta[idx, :]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) < 3:
                continue
            log_vals = np.log(vals)
            xs = np.linspace(log_vals.min() - 0.5, log_vals.max() + 0.5, 300)
            try:
                kde = gaussian_kde(log_vals, bw_method='scott')
                ax_d.plot(xs, kde(xs), color=colors[idx],
                          label=site_labels[idx], linewidth=1.8)
            except Exception:
                pass

        ax_d.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax_d.set_xlabel('log(delta*) [multiplicative site effect]', fontsize=11)
        ax_d.set_ylabel('Density', fontsize=11)
        ax_d.set_title('Multiplicative effects — log(delta*)', fontsize=11)
        ax_d.legend(fontsize=7, loc='upper right', framealpha=0.7)
        ax_d.grid(True, alpha=0.3)

        fig.subplots_adjust(top=0.88, wspace=0.35)
        out_file = os.path.join(output_path, f'combat_params_{metric}.png')
        fig.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] Gamma/delta plot saved: {out_file}")


def save_harmonization_report(data_original, data_harmonized, output_path,
                               n_ref, n_total):
    """Generate a report comparing original and harmonized data."""
    report = []
    report.append("=" * 80)
    report.append("HARMONIZATION REPORT — Reference-Based Two-Step ComBat")
    report.append("=" * 80)
    report.append(f"\nMethod: harmonizationLearn on Controls (n={n_ref} rows), "
                  f"harmonizationApply on all (n={n_total} rows)")
    report.append(f"Original dataset shape: {data_original.shape}")
    report.append(f"Harmonized dataset shape: {data_harmonized.shape}")

    report.append("\n" + "-" * 80)
    report.append("COVARIATE PRESERVATION CHECK")
    report.append("-" * 80)

    for col in COVARIATE_COLS:
        if col in data_original.columns and col in data_harmonized.columns:
            if data_original[col].equals(data_harmonized[col]):
                report.append(f"[OK] {col}: PRESERVED")
            elif pd.api.types.is_numeric_dtype(data_original[col]):
                if np.allclose(data_original[col].fillna(0),
                               data_harmonized[col].fillna(0),
                               rtol=1e-5, atol=1e-8):
                    report.append(f"[OK] {col}: PRESERVED (within tolerance)")
                else:
                    report.append(f"[!] {col}: MODIFIED")
            else:
                report.append(f"[!] {col}: MODIFIED")

    report_file = os.path.join(output_path, 'harmonization_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))
    print(f"\nReport saved to: {report_file}")


# ============================================================================
# PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("GLOBAL EEG HARMONIZATION PIPELINE (Reference-Based)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # [1] Load
    # ------------------------------------------------------------------
    print("\n[1/6] Loading complete dataset...")
    data = pd.read_feather(INPUT_FILE)
    print(f"  Loaded {data.shape[0]} rows, {data.shape[1]} features")

    data_harmonized = data.copy()

    # ------------------------------------------------------------------
    # [2] Split: reference (Controls) vs full dataset
    # ------------------------------------------------------------------
    print("\n[2/6] Splitting into reference (Controls) and full dataset...")

    controls_mask = data['group'] == 'Control'
    n_ref   = controls_mask.sum()
    n_total = len(data)
    print(f"  Controls (reference): {n_ref} rows")
    print(f"  All subjects (apply): {n_total} rows")

    # Medellin controls check — critical for reference-based approach
    if 'SITE' in data.columns:
        mdl_col = data['SITE'].str.contains('edel', na=False, case=False)
        mdl_ctrl = (controls_mask & mdl_col).sum()
        mdl_psen = (data['group'] == 'PSEN1').sum()
        print(f"\n  Medellin site check:")
        print(f"    PSEN1 carriers:       {mdl_psen}")
        print(f"    Medellin Controls:    {mdl_ctrl} "
              f"({'OK — sufficient for estimation' if mdl_ctrl >= 5 else 'WARNING: very few'})")
        by_site = data[controls_mask & mdl_col].groupby('SITE').size()
        for site, n in by_site.items():
            print(f"      {site}: {n} controls")

    # ------------------------------------------------------------------
    # [3] Prepare covariates
    # ------------------------------------------------------------------
    print("\n[3/6] Preparing covariates...")

    # Reference: controls only
    data_ref_clean = mapsDrop(data[controls_mask].reset_index(drop=True))
    _, cvars_ref   = covars(data_ref_clean)
    covars_ref_df  = pd.DataFrame(cvars_ref)

    # Full dataset
    data_all_clean = mapsDrop(data)
    _, cvars_all   = covars(data_all_clean)
    covars_all_df  = pd.DataFrame(cvars_all)

    # Save original covariate columns for restoration after harmonization
    original_covariates = {}
    for col in COVARIATE_COLS:
        if col in data.columns:
            original_covariates[col] = data[col].copy()

    # Build SITE mapping from ALL sites in the full dataset (ensures consistency)
    all_sites   = sorted(data['SITE'].dropna().unique()) if 'SITE' in data.columns else []
    site_mapping = {site: idx for idx, site in enumerate(all_sites)}
    print(f"  SITE mapping ({len(site_mapping)} sites):")
    for site, idx in site_mapping.items():
        grps = data[data['SITE'] == site]['group'].value_counts().to_dict()
        print(f"    [{idx}] {site}: {grps}")

    # Encode covariates with the SAME site mapping
    covars_ref_df = encode_covars(covars_ref_df, site_mapping)
    covars_all_df = encode_covars(covars_all_df, site_mapping)

    print(f"\n  Reference covariates: {list(covars_ref_df.columns)}")
    print(f"  Reference age: mean={covars_ref_df['age'].mean():.1f}, "
          f"range=[{covars_ref_df['age'].min():.0f}, {covars_ref_df['age'].max():.0f}]")
    print(f"  Full data age: mean={covars_all_df['age'].mean():.1f}, "
          f"range=[{covars_all_df['age'].min():.0f}, {covars_all_df['age'].max():.0f}]")

    # ------------------------------------------------------------------
    # [4] Reference-based harmonization — independently per feature type
    # ------------------------------------------------------------------
    print("\n[4/6] Reference-based harmonization (learn on Controls, apply to all)...")
    models = {}

    for metric in METRICS:
        print(f"\n  Processing metric: {metric.upper()}")
        print("  " + "-" * 60)

        feat_cols = get_metric_feature_cols(data_ref_clean.columns, metric)
        if not feat_cols:
            print(f"  [!] No feature columns found for '{metric}'. Skipping...")
            continue
        print(f"  Found {len(feat_cols)} feature columns for {metric}")

        block_ref = data_ref_clean[feat_cols].copy()
        block_all = data_all_clean[feat_cols].copy()

        # Two-step harmonization
        metric_harmonized, model = harmonize_block_reference(
            block_ref, covars_ref_df,
            block_all, covars_all_df
        )

        if metric_harmonized.empty or model is None:
            print(f"  [!] Skipping metric '{metric}' — harmonization failed")
            continue

        # Update harmonized dataframe
        for col in metric_harmonized.columns:
            if col in data_harmonized.columns:
                data_harmonized.loc[:, col] = metric_harmonized[col].values

        models[metric] = model
        print(f"  [OK] Harmonized {len(metric_harmonized.columns)} features for {metric}")

    # ------------------------------------------------------------------
    # [5] Restore original covariates
    # ------------------------------------------------------------------
    print("\n[5/6] Restoring original covariate values...")
    for col, values in original_covariates.items():
        if col in data_harmonized.columns:
            data_harmonized[col] = values
            print(f"  [OK] Restored: {col}")

    # ------------------------------------------------------------------
    # [6] Save
    # ------------------------------------------------------------------
    print("\n[6/6] Saving results...")

    data_harmonized.to_feather(OUTPUT_FILE)
    print(f"  [OK] Harmonized dataset saved: {OUTPUT_FILE}")

    model_file = os.path.join(MODEL_PATH, 'neuroHarmonize_models.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(models, f)
    print(f"  [OK] Models saved: {model_file}")

    print("\n[REPORT] Generating harmonization report...")
    save_harmonization_report(data, data_harmonized, MODEL_PATH,
                               n_ref=n_ref, n_total=n_total)

    print("\n[PLOTS] Generating gamma/delta parameter plots...")
    plot_combat_parameters(models, site_mapping, MODEL_PATH)

    print("\n" + "=" * 80)
    print("HARMONIZATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nFinal dataset shape: {data_harmonized.shape}")
    print(f"Harmonized metrics: {list(models.keys())}")

    if 'age' in data_harmonized.columns:
        print(f"\nAge verification:")
        print(f"  Original:   [{data['age'].min():.1f}, {data['age'].max():.1f}]")
        print(f"  Harmonized: [{data_harmonized['age'].min():.1f}, {data_harmonized['age'].max():.1f}]")
        if data['age'].equals(data_harmonized['age']):
            print("  [OK] Age PRESERVED correctly")
        else:
            print("  [!] WARNING: Age was modified!")


if __name__ == "__main__":
    main()
