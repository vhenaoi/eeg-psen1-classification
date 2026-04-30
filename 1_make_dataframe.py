"""
Data Integration Script
Merges all EEG metrics (power, SL, coherence, entropy, crossfreq) with demographics
No harmonization needed - single site data
"""

import pandas as pd
import os

def ensure_subject_column(df):
    SUBJECT_CANDIDATES = [
        'subject',
        'participant_id',
        'participant',
        'sub_id',
        'subject_id',
        'id',
        'id_subject'
    ]

    cols_lower = {c.lower(): c for c in df.columns}

    if 'subject' in cols_lower:
        df['subject'] = df[cols_lower['subject']]
        return df

    for candidate in SUBJECT_CANDIDATES:
        if candidate in cols_lower:
            df['subject'] = df[cols_lower[candidate]]
            return df

    raise ValueError(
        "No subject identifier column found. "
        f"Expected one of: {SUBJECT_CANDIDATES}"
    )


def deduplicate_subjects(df, source_name=''):
    """
    Keep only one row per subject.
    Warns if duplicates are found before removing them.
    """
    n_before = len(df)
    duplicated_mask = df.duplicated(subset='subject', keep=False)
    n_duplicated = duplicated_mask.sum()

    if n_duplicated > 0:
        dup_subjects = df.loc[duplicated_mask, 'subject'].unique()
        print(f"  ⚠ Duplicates found in {source_name}: {n_duplicated} rows "
              f"({len(dup_subjects)} subjects) → keeping first occurrence")
        if len(dup_subjects) <= 10:
            for s in dup_subjects:
                subset = df[df['subject'] == s]
                print(f"      {s}: {len(subset)} rows")
        df = df.drop_duplicates(subset='subject', keep='first')
        print(f"    Rows: {n_before} → {len(df)}")

    return df


def load_feather(path, label):
    """Load a feather file, ensure subject column, and deduplicate."""
    df = pd.read_feather(path)
    df = ensure_subject_column(df)
    df = deduplicate_subjects(df, source_name=label)
    print(f"  ✓ {label}: {df.shape}")
    return df


def merge_eeg_metrics(base_path, data_type='CE', space='roi'):
    """
    Merge all EEG metrics into single dataframe.
    
    Parameters:
    -----------
    base_path : str
        Base directory path
    data_type : str
        'CE' or 'IC'
    space : str
        'roi' or 'ic'
    
    Returns:
    --------
    DataFrame : Merged dataframe with all metrics
    """
    print(f"\n{'='*60}")
    print(f"MERGING EEG METRICS: {data_type} - {space}")
    print(f"{'='*60}\n")
    
    base_name = os.path.basename(base_path).lower()

    files = {
        'power':      f'power_{base_name}_{data_type}_{space}.feather',
        'sl':         f'sl_{base_name}_{data_type}_{space}.feather',
        'coherence':  f'cohfreq_{base_name}_{data_type}_{space}.feather',
        'entropy':    f'entropy_{base_name}_{data_type}_{space}.feather',
        'cross_freq': f'crossfreq_{base_name}_{data_type}_{space}.feather',
    }

    feather_dir = os.path.join(base_path, 'feather_files')

    # Load and deduplicate each metric
    print("Loading EEG metrics...")
    data_power     = load_feather(os.path.join(feather_dir, files['power']),      'Power')
    data_sl        = load_feather(os.path.join(feather_dir, files['sl']),         'SL')
    data_coherence = load_feather(os.path.join(feather_dir, files['coherence']),  'Coherence')
    data_entropy   = load_feather(os.path.join(feather_dir, files['entropy']),    'Entropy')
    data_cross     = load_feather(os.path.join(feather_dir, files['cross_freq']), 'Cross-frequency')

    # Merge all dataframes
    print("\nMerging dataframes...")
    data_complete = data_power.copy()
    data_complete = pd.merge(data_complete, data_sl,        on='subject', how='left', suffixes=('', '_sl'))
    data_complete = pd.merge(data_complete, data_coherence, on='subject', how='left', suffixes=('', '_coh'))
    data_complete = pd.merge(data_complete, data_entropy,   on='subject', how='left', suffixes=('', '_ent'))
    data_complete = pd.merge(data_complete, data_cross,     on='subject', how='left', suffixes=('', '_cross'))

    # Remove duplicate columns (same name, keep first)
    data_complete = data_complete.loc[:, ~data_complete.columns.duplicated(keep='first')]

    # Final deduplication safety check
    data_complete = deduplicate_subjects(data_complete, source_name='merged dataset')
    print(f"  ✓ Merged shape: {data_complete.shape}")

    # Standardize group names
    if 'group' in data_complete.columns:
        group_mapping = {
            'CTR':    'Control',
            'HC_AD':  'Control',
            'HC_MCI': 'Control',
            'HC_SCr': 'Control',
            'HC_ACr': 'Control',
            'DCL':    'ADMCI',
            'MCI':    'ADMCI',
            'GG':     'PSEN1',
            'ACr':    'PSEN1',
            'SCr':    'ADMCI',
            'AD':     'ADMCI',
            'G1':     'PSEN1',
            'GU':     'Relative',
        }
        data_complete['group'].replace(group_mapping, inplace=True)
        print(f"\n  Group distribution:")
        print(data_complete['group'].value_counts())

    return data_complete


def add_demographics(data, demographic_path):
    """
    Add sex and education from the consolidated demographics Excel.
    Age and SITE already come from the EEG feathers and are preserved as-is.

    Expected Excel columns: 'subject', 'sex' (M/F), 'education' (years, float).
    The file may contain many extra columns — only these three are used.
    Subjects with no match in the Excel will have sex=NaN / education=NaN.
    """
    print("\nAdding demographic information...")

    demographic = pd.read_excel(demographic_path)

    # One row per subject; keep only what we need
    dem_sel = (
        demographic[['subject', 'sex', 'education']]
        .drop_duplicates(subset='subject', keep='first')
    )

    data_with_demo = data.merge(dem_sel, on='subject', how='left')

    # Deduplicate again in case demographics introduced extra rows
    data_with_demo = deduplicate_subjects(data_with_demo, source_name='after demographics merge')

    n = len(data_with_demo)
    sex_ok = data_with_demo['sex'].notna().sum()
    edu_ok = data_with_demo['education'].notna().sum()
    print(f"  sex      : {sex_ok:>5}/{n} subjects ({100*sex_ok//n}%)")
    print(f"  education: {edu_ok:>5}/{n} subjects ({100*edu_ok//n}%)")
    print(f"  ✓ Final shape: {data_with_demo.shape}")

    return data_with_demo


def save_merged_data(data, output_path):
    """Save merged dataframe"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.reset_index(drop=True).to_feather(output_path)
    print(f"\n✓ Saved merged data to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    BASE_PATH = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES'
    DEMOGRAPHIC_FILE = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\datos_filtrados_concatenados.xlsx'
    
    list_data_types = ['CE']
    for i in list_data_types:
        DATA_TYPE = i
        SPACE = 'ROI'
        
        data = merge_eeg_metrics(BASE_PATH, DATA_TYPE, SPACE)
        
        if DEMOGRAPHIC_FILE is not None:
            data = add_demographics(data, DEMOGRAPHIC_FILE)
        
        output_file = os.path.join(BASE_PATH, 'Resultados', f'Data_complete_{DATA_TYPE.lower()}_{SPACE.lower()}.feather')
        save_merged_data(data, output_file)
        
        print("\n" + "="*60)
        print("DATA MERGING COMPLETED!")
        print("="*60)
        print(f"\nFinal dataset:")
        print(f"  Subjects : {len(data)}")
        print(f"  Features : {len(data.columns)}")
        print(f"  Groups   : {data['group'].value_counts().to_dict()}")
