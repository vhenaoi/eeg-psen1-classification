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

def merge_eeg_metrics(base_path, data_type='CE', space='roi'):
    """
    Merge all EEG metrics into single dataframe
    
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
    
    # File paths configuration
    base_name = os.path.basename(base_path).lower()

    files = {
        'power': f'power_{base_name}_{data_type}_{space}.feather',
        'sl': f'sl_{base_name}_{data_type}_{space}.feather',
        'coherence': f'cohfreq_{base_name}_{data_type}_{space}.feather',
        'entropy': f'entropy_{base_name}_{data_type}_{space}.feather',
        'cross_freq': f'crossfreq_{base_name}_{data_type}_{space}.feather'
    }

    
    # Load all metrics
    print("Loading EEG metrics...")
    data_power = pd.read_feather(os.path.join(base_path, 'feather_files', files["power"]))
    data_power = ensure_subject_column(data_power)
    print(f"  ✓ Power: {data_power.shape}")
    
    data_sl = pd.read_feather(os.path.join(base_path, 'feather_files', files["sl"]))
    data_sl = ensure_subject_column(data_sl)
    print(f"  ✓ SL: {data_sl.shape}")
    
    data_coherence = pd.read_feather(os.path.join(base_path, 'feather_files', files["coherence"]))
    data_coherence = ensure_subject_column(data_coherence)
    print(f"  ✓ Coherence: {data_coherence.shape}")
    
    data_entropy = pd.read_feather(os.path.join(base_path, 'feather_files', files["entropy"]))
    data_entropy = ensure_subject_column(data_entropy)
    print(f"  ✓ Entropy: {data_entropy.shape}")
    
    data_cross = pd.read_feather(os.path.join(base_path, 'feather_files', files["cross_freq"]))
    data_cross = ensure_subject_column(data_cross)
    print(f"  ✓ Cross-frequency: {data_cross.shape}")
    
    # Merge all dataframes
    print("\nMerging dataframes...")
    data_complete = data_power.copy()
    data_complete = pd.merge(data_complete, data_sl, on='subject', how='left', suffixes=('', '_sl'))
    data_complete = pd.merge(data_complete, data_coherence, on='subject', how='left', suffixes=('', '_coh'))
    data_complete = pd.merge(data_complete, data_entropy, on='subject', how='left', suffixes=('', '_ent'))
    data_complete = pd.merge(data_complete, data_cross, on='subject', how='left', suffixes=('', '_cross'))
    
    # Clean duplicate columns
    data_complete = data_complete.loc[:, ~data_complete.columns.duplicated(keep='first')]
    print(f"  ✓ Merged shape: {data_complete.shape}")
    
    # Standardize group names
    if 'group' in data_complete.columns:
        group_mapping = {
            'CTR': 'Control',
            'HC_AD': 'Control',
            'HC_MCI': 'Control',
            'HC_SCr': 'Control',
            'HC_ACr': 'Control',
            'DCL': 'ADMCI',
            'MCI': 'ADMCI',
            'GG': 'PSEN1',
            'ACr': 'PSEN1',
            'SCr': 'ADMCI',
            'AD': 'ADMCI',
            'G1': 'PSEN1',
            'GU': 'Relative'
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
    
    # Configuration
    BASE_PATH = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES'
    DEMOGRAPHIC_FILE = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\datos_filtrados_concatenados.xlsx'
    
    #DATA_TYPE = 'CE'  # 'CE', 'DTAN', 'ST1', 'DTS1', 'DTS7', DTCT','DTV','RTRU','RTVI' 
    list_data_types = ['CE']#, 'DTAN', 'ST1', 'DTS1', 'DTS7', 'DTCT','DTV','RTRU','RTVI']
    for i in list_data_types:
        DATA_TYPE = i   # Data type and space      
        SPACE = 'ROI'     # 'roi' or 'ic'
        
        # Merge EEG metrics
        data = merge_eeg_metrics(BASE_PATH, DATA_TYPE, SPACE)
        
        if DEMOGRAPHIC_FILE is not None:
            # Add demographics
            data = add_demographics(data, DEMOGRAPHIC_FILE)
        
        # Save
        output_file = os.path.join(BASE_PATH, 'Resultados', f'Data_complete_{DATA_TYPE.lower()}_{SPACE.lower()}.feather')
        save_merged_data(data, output_file)
        
        print("\n" + "="*60)
        print("DATA MERGING COMPLETED!")
        print("="*60)
        print(f"\nFinal dataset:")
        print(f"  Subjects: {len(data)}")
        print(f"  Features: {len(data.columns)}")
        print(f"  Groups: {data['group'].value_counts().to_dict()}")
