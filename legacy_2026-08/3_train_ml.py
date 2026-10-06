"""
Machine Learning Training Script - SUBJECT-LEVEL VERSION
CRITICAL: Train/test split at SUBJECT level to prevent data leakage
Each subject's data stays together in either train OR test, never both
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)

from utils import (
    mapa_de_correlacion, grid_search, randomFo,
    primeras_carateristicas, features_best3,
    curva_validacion3, plot_confusion_matrix,
    computerprecision
)


# Columns that should NOT be used as features in the model
COLS_TO_EXCLUDE_FROM_MODEL = ['subject', 'Task', 'group', 'ses', 'mmse', 'moca']


class MLConfig:
    """Configuration for ML pipeline"""
    
    def __init__(self, data_path, results_path, data_type, space='roi'):
        self.data_path = data_path
        self.results_path = results_path
        self.data_type = data_type
        self.space = space
    
    def build_paths(self, group1, ratio_str):
        """Build paths for a specific configuration"""
        plot_path = os.path.join(
            self.results_path, 'graphics', 'ML', 
            f'{group1}_{ratio_str}_{self.data_type}_{self.space}'
        )
        
        table_path = os.path.join(
            self.results_path, 'tables', 'ML', self.space, group1
        )
        
        os.makedirs(plot_path, exist_ok=True)
        os.makedirs(table_path, exist_ok=True)
        
        return plot_path, table_path


def create_demographic_summary_table(data, output_path, ratio_str):
    """
    Create demographic summary at SUBJECT LEVEL
    """
    print("\nCreating demographic summary (SUBJECT LEVEL)...")
    
    # Get one row per subject
    demo_cols = ['subject', 'group', 'age', 'sex', 'education', 'ses', 'mmse', 'moca']
    available_cols = [c for c in demo_cols if c in data.columns]
    
    subject_data = data[available_cols].drop_duplicates(subset=['subject'], keep='first')
    
    summary_data = []
    groups = sorted(subject_data['group'].unique())
    
    for group in groups:
        group_data = subject_data[subject_data['group'] == group]
        row = {'Group': group, 'N_Subjects': len(group_data)}
        
        # Age
        if 'age' in subject_data.columns:
            row['Age_Mean'] = f"{group_data['age'].mean():.1f}"
            row['Age_SD'] = f"{group_data['age'].std():.1f}"
            row['Age_Range'] = f"{group_data['age'].min():.0f}-{group_data['age'].max():.0f}"
        
        # Sex
        if 'sex' in subject_data.columns:
            sex_norm = group_data['sex'].map({
                'M': 1, 'Male': 1, 1: 1, '1': 1, 1.0: 1,
                'F': 0, 'Female': 0, 0: 0, '0': 0, 0.0: 0
            })
            male_count = (sex_norm == 1).sum()
            female_count = (sex_norm == 0).sum()
            
            row['Sex_Male'] = male_count
            row['Sex_Female'] = female_count
            row['Sex_Ratio'] = f"{male_count}:{female_count}"
            if male_count + female_count > 0:
                row['Sex_Male_Pct'] = f"{male_count/(male_count+female_count)*100:.1f}%"
        
        # Education
        if 'education' in subject_data.columns:
            row['Education_Mean'] = f"{group_data['education'].mean():.1f}"
            row['Education_SD'] = f"{group_data['education'].std():.1f}"
        
        # SES
        if 'ses' in subject_data.columns:
            row['SES_Mean'] = f"{group_data['ses'].mean():.1f}"
            row['SES_SD'] = f"{group_data['ses'].std():.1f}"
        
        # MMSE
        if 'mmse' in subject_data.columns:
            row['MMSE_Mean'] = f"{group_data['mmse'].mean():.1f}"
            row['MMSE_SD'] = f"{group_data['mmse'].std():.1f}"
        
        # MoCA
        if 'moca' in subject_data.columns:
            row['MoCA_Mean'] = f"{group_data['moca'].mean():.1f}"
            row['MoCA_SD'] = f"{group_data['moca'].std():.1f}"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    summary_file = os.path.join(output_path, f'demographic_summary_{ratio_str}.xlsx')
    summary_df.to_excel(summary_file, index=False)
    print(f"  ✓ Saved: {summary_file}")
    
    print("\nDemographic Summary (SUBJECTS):")
    print(summary_df.to_string(index=False))
    
    return summary_df


def subject_level_train_test_split(data, test_size=0.2, random_state=1):
    """
    CRITICAL: Split data at SUBJECT level, not row level
    
    This ensures:
    - Each subject's data stays in ONE split (train OR test)
    - No data leakage between splits
    - Proper evaluation of model generalization
    
    Parameters:
    -----------
    data : DataFrame
        Full dataset with 'subject' and 'group' columns
    test_size : float
        Proportion of subjects for test set
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : (train_data, test_data, train_subjects, test_subjects)
    """
    print(f"\n{'='*80}")
    print(f"SUBJECT-LEVEL TRAIN/TEST SPLIT")
    print(f"{'='*80}")
    
    # Get unique subjects with their groups
    subject_groups = data[['subject', 'group']].drop_duplicates()
    
    print(f"\nTotal unique subjects: {len(subject_groups)}")
    print(f"Group distribution (subjects):")
    for group, count in subject_groups['group'].value_counts().items():
        print(f"  {group}: {count} subjects")
    
    # Split subjects by group (stratified)
    train_subjects = []
    test_subjects = []
    
    np.random.seed(random_state)
    
    for group in subject_groups['group'].unique():
        group_subjects = subject_groups[subject_groups['group'] == group]['subject'].values
        
        # Shuffle subjects
        np.random.shuffle(group_subjects)
        
        # Calculate split point
        n_test = max(1, int(len(group_subjects) * test_size))
        n_train = len(group_subjects) - n_test
        
        # Split
        test_subjects.extend(group_subjects[:n_test])
        train_subjects.extend(group_subjects[n_test:])
        
        print(f"\n  {group}:")
        print(f"    Total: {len(group_subjects)} subjects")
        print(f"    Train: {n_train} subjects ({n_train/len(group_subjects)*100:.1f}%)")
        print(f"    Test: {n_test} subjects ({n_test/len(group_subjects)*100:.1f}%)")
    
    # Create train/test datasets
    train_data = data[data['subject'].isin(train_subjects)].copy()
    test_data = data[data['subject'].isin(test_subjects)].copy()
    
    # Verification
    print(f"\n{'='*80}")
    print(f"VERIFICATION")
    print(f"{'='*80}")
    
    print(f"\nTrain set:")
    print(f"  Unique subjects: {train_data['subject'].nunique()}")
    print(f"  Total rows: {len(train_data)}")
    train_groups = train_data[['subject', 'group']].drop_duplicates()['group'].value_counts()
    for group, count in train_groups.items():
        print(f"    {group}: {count} subjects")
    
    print(f"\nTest set:")
    print(f"  Unique subjects: {test_data['subject'].nunique()}")
    print(f"  Total rows: {len(test_data)}")
    test_groups = test_data[['subject', 'group']].drop_duplicates()['group'].value_counts()
    for group, count in test_groups.items():
        print(f"    {group}: {count} subjects")
    
    # CRITICAL CHECK: No subject overlap
    train_subject_set = set(train_data['subject'].unique())
    test_subject_set = set(test_data['subject'].unique())
    overlap = train_subject_set & test_subject_set
    
    if len(overlap) > 0:
        raise ValueError(f"ERROR: {len(overlap)} subjects appear in BOTH train and test!")
    else:
        print(f"\n✓ NO OVERLAP: Train and test sets are completely separate")
    
    print(f"{'='*80}\n")
    
    return train_data, test_data, train_subjects, test_subjects


def aggregate_subject_features(data, feature_cols, agg_method='mean'):
    """
    Aggregate multiple rows per subject into single feature vector
    
    Parameters:
    -----------
    data : DataFrame
        Data with potentially multiple rows per subject
    feature_cols : list
        Column names to use as features
    agg_method : str
        Aggregation method: 'mean', 'median', 'max', 'min'
    
    Returns:
    --------
    DataFrame : One row per subject with aggregated features
    """
    # Keep subject, group, and feature columns
    cols_to_keep = ['subject', 'group'] + feature_cols
    cols_available = [c for c in cols_to_keep if c in data.columns]
    
    data_subset = data[cols_available].copy()
    
    # CRITICAL: Get group for each subject BEFORE aggregation (group is constant per subject)
    subject_groups = data_subset[['subject', 'group']].drop_duplicates()
    
    # Now aggregate ONLY the numeric features (exclude 'group')
    numeric_cols = [c for c in feature_cols if c in data_subset.columns]
    
    # Group by subject and aggregate ONLY numeric features
    if agg_method == 'mean':
        aggregated = data_subset.groupby('subject')[numeric_cols].mean()
    elif agg_method == 'median':
        aggregated = data_subset.groupby('subject')[numeric_cols].median()
    elif agg_method == 'max':
        aggregated = data_subset.groupby('subject')[numeric_cols].max()
    elif agg_method == 'min':
        aggregated = data_subset.groupby('subject')[numeric_cols].min()
    else:
        raise ValueError(f"Unknown aggregation method: {agg_method}")
    
    # Reset index to get subject back as column
    aggregated = aggregated.reset_index()
    
    # Add group back (merge with subject_groups)
    aggregated = aggregated.merge(subject_groups, on='subject', how='left')
    
    return aggregated


def prepare_ml_data(train_data, test_data, group_label_mapping):
    """
    Prepare X, y for ML from subject-level data
    
    Parameters:
    -----------
    train_data : DataFrame
        Training data (already at subject level)
    test_data : DataFrame
        Test data (already at subject level)
    group_label_mapping : dict
        Mapping from group names to numeric labels
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, feature_names)
    """
    # Encode groups
    train_data['group_encoded'] = train_data['group'].map(group_label_mapping)
    test_data['group_encoded'] = test_data['group'].map(group_label_mapping)
    
    # Get feature columns (exclude group_encoded which we just added)
    feature_cols = [c for c in train_data.columns 
                   if c not in ['subject', 'group', 'group_encoded']]
    
    # Extract X and y
    X_train = train_data[feature_cols].values
    y_train = train_data['group_encoded'].values
    
    X_test = test_data[feature_cols].values
    y_test = test_data['group_encoded'].values
    
    feature_names = np.array(feature_cols)
    
    return X_train, X_test, y_train, y_test, feature_names


def train_model_on_psm_data(data_file, config, group1, ratio_str, class_names, group_mapping=None):
    """
    Train Random Forest with SUBJECT-LEVEL train/test split
    """
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL: {ratio_str}")
    print(f"{'='*80}\n")
    
    # Build paths
    plot_path, table_path = config.build_paths(group1, ratio_str.replace(':', 'to'))
    
    # Load data
    print("Loading data...")
    data_full = pd.read_feather(data_file)
    print(f"✓ Loaded: {data_full.shape}")
    print(f"  Total rows: {len(data_full)}")
    print(f"  Unique subjects: {data_full['subject'].nunique()}")
    
    # Apply group mapping if provided
    if group_mapping:
        data_full['group'] = data_full['group'].replace(group_mapping)
    
    # REPORT SAMPLE SIZES (SUBJECTS)
    print(f"\n{'='*80}")
    print(f"SAMPLE SIZE REPORT (SUBJECTS)")
    print(f"{'='*80}")
    subject_counts = data_full[['subject', 'group']].drop_duplicates()['group'].value_counts()
    for group, count in subject_counts.items():
        print(f"  {group}: {count} subjects")
    print(f"  TOTAL: {data_full['subject'].nunique()} subjects")
    print(f"{'='*80}\n")
    
    # CREATE DESCRIPTIVE STATISTICS
    create_demographic_summary_table(data_full, table_path, ratio_str.replace(":", "to"))
    
    # Prepare data for modeling
    print("\nPreparing data for modeling...")
    
    # Exclude non-model columns
    cols_to_exclude = [col for col in COLS_TO_EXCLUDE_FROM_MODEL if col in data_full.columns]
    print(f"  Excluding: {cols_to_exclude}")
    
    # Keep subject and group for splitting
    modeling_cols = [c for c in data_full.columns if c not in cols_to_exclude or c in ['subject', 'group']]
    data_model = data_full[modeling_cols].copy()
    
    # Remove columns with missing values
    cols_with_na = [col for col in data_model.columns if data_model[col].isna().sum() > 0]
    if cols_with_na:
        print(f"  Removing {len(cols_with_na)} columns with missing values")
        data_model = data_model.drop(cols_with_na, axis=1)
    
    # Encode sex if present
    if 'sex' in data_model.columns:
        sex_mapping = {
            'M': 1, 'Male': 1, 1: 1, '1': 1, 1.0: 1,
            'F': 0, 'Female': 0, 0: 0, '0': 0, 0.0: 0
        }
        data_model['sex'] = data_model['sex'].map(sex_mapping)
        print(f"  Sex encoded to 0/1")
    
    # Keep only numeric columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_cols = data_model.select_dtypes(include=numerics).columns.tolist()
    
    # Always keep subject and group
    if 'subject' not in numeric_cols:
        numeric_cols = ['subject'] + numeric_cols
    if 'group' not in numeric_cols:
        numeric_cols = ['group'] + numeric_cols
    
    data_model = data_model[numeric_cols]
    
    print(f"  Data shape after cleaning: {data_model.shape}")
    
    # Get feature columns (before aggregation)
    feature_cols = [c for c in data_model.columns if c not in ['subject', 'group']]
    
    # CRITICAL: Aggregate to subject level if multiple rows per subject
    rows_per_subject = data_model.groupby('subject').size()
    if rows_per_subject.max() > 1:
        print(f"\n  Multiple rows per subject detected:")
        print(f"    Mean rows/subject: {rows_per_subject.mean():.1f}")
        print(f"    Max rows/subject: {rows_per_subject.max()}")
        print(f"  Aggregating to subject level using MEAN...")
        
        data_aggregated = aggregate_subject_features(data_model, feature_cols, agg_method='mean')
        print(f"  ✓ Aggregated shape: {data_aggregated.shape}")
    else:
        print(f"  Already one row per subject - no aggregation needed")
        data_aggregated = data_model.copy()
    
    # SUBJECT-LEVEL TRAIN/TEST SPLIT
    train_data, test_data, train_subjects, test_subjects = subject_level_train_test_split(
        data_aggregated,
        test_size=0.2,
        random_state=1
    )
    
    # Encode groups
    group_label_mapping = {label: idx for idx, label in enumerate(sorted(train_data['group'].unique()))}
    print(f"\nGroup encoding: {group_label_mapping}")
    
    # Prepare X, y
    X_train, X_test, y_train, y_test, column_names = prepare_ml_data(
        train_data, test_data, group_label_mapping
    )
    
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape} ({len(set(train_subjects))} subjects)")
    print(f"  X_test: {X_test.shape} ({len(set(test_subjects))} subjects)")
    print(f"  Features: {len(column_names)}")
    
    # Check for overfitting risk
    ratio = X_train.shape[0] / X_train.shape[1]
    print(f"\n  Samples/Features ratio: {ratio:.1f}")
    if ratio >= 10:
        print(f"  ✓ LOW overfitting risk")
        overfitting_risk = "LOW"
    elif ratio >= 5:
        print(f"  ⚠ MODERATE overfitting risk")
        overfitting_risk = "MODERATE"
    else:
        print(f"  ✗ HIGH overfitting risk")
        overfitting_risk = "HIGH"
    
    # Correlation analysis
    print("\nPerforming correlation analysis...")
    temp_df = pd.DataFrame(X_train, columns=column_names)
    temp_df = mapa_de_correlacion(temp_df, plot_path, ratio_str.replace(':', 'to'))
    
    # Update data after correlation filtering
    X_train = temp_df.values
    X_test = test_data[temp_df.columns].values
    column_names = np.array(temp_df.columns)
    
    print(f"  After correlation filter: {X_train.shape[1]} features")
    
    # Grid search
    print("\nPerforming hyperparameter tuning...")
    random_grid = grid_search()
    random_grid['class_weight'] = ['balanced', 'balanced_subsample', None]
    
    rf_random = randomFo(random_grid, X_train, y_train)
    best_model = rf_random.best_estimator_
    best_params = rf_random.best_params_
    
    print(f"\n✓ Best parameters:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    
    # Feature importance
    print("\nAnalyzing feature importance...")
    feature_scores = best_model.feature_importances_
    sorted_idx = np.argsort(feature_scores)[::-1]
    
    # Check for age confounding
    if 'age' in column_names:
        age_idx = np.where(column_names == 'age')[0]
        if len(age_idx) > 0:
            age_rank = np.where(sorted_idx == age_idx[0])[0][0] + 1
            if age_rank <= 3:
                print(f"\n⚠ WARNING: 'age' is feature #{age_rank} - check balance!")
                age_confounding = True
            else:
                age_confounding = False
        else:
            age_confounding = False
    else:
        age_confounding = False
    
    # Save feature importance
    sorted_names = []
    feat_df = primeras_carateristicas(
        X_train, sorted_names, column_names,
        feature_scores, pd.DataFrame(), sorted_idx,
        plot_path, ratio_str.replace(':', 'to')
    )
    
    # Learning curve
    print("\nGenerating validation curve...")
    palette = ["#8AA6A3", "#127369"]
    curva_validacion3(
        best_model, X_train, y_train,
        f'validation_{ratio_str.replace(":", "to")}.png',
        palette, ratio_str.replace(':', 'to')
    )
    plt.savefig(os.path.join(plot_path, f'validation_{ratio_str.replace(":", "to")}.png'), 
               bbox_inches='tight')
    plt.close()
    
    # Feature selection
    print("\nPerforming feature selection...")
    acc, std, final_model, best_idx = features_best3(
        sorted_names[:min(50, len(sorted_names))],
        best_model, pd.DataFrame(X_train, columns=column_names),
        X_train, y_train, plot_path
    )
    
    if final_model is None or best_idx is None or len(best_idx) == 0:
        print("  Using all features")
        final_model = best_model
        best_idx = list(range(X_train.shape[1]))
    
    print(f"  ✓ Selected {len(best_idx)} features")
    
    # Train final model
    final_model.fit(X_train[:, best_idx], y_train)
    
    # Predictions
    y_pred = final_model.predict(X_test[:, best_idx])
    y_pred_proba = final_model.predict_proba(X_test[:, best_idx])[:, 1]
    
    # Calculate metrics
    print("\nEvaluating on test set...")
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    min_samples = min(np.bincount(y_train))
    cv = StratifiedKFold(n_splits=min(10, min_samples), shuffle=True, random_state=1)
    scores = cross_val_score(final_model, X_train[:, best_idx], y_train, cv=cv, n_jobs=-1)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Test Precision: {precision:.3f}")
    print(f"Test Recall: {recall:.3f}")
    print(f"Test F1: {f1:.3f}")
    print(f"Test AUC: {auc_score:.3f}")
    print(f"CV Accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print(f"{'='*80}\n")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(cm)
    
    plot_confusion_matrix(
        plot_path, ratio_str.replace(':', 'to'), cm,
        classes=class_names,
        title=f'Confusion Matrix'
    )
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {ratio_str}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plot_path, f'roc_curve_{ratio_str.replace(":", "to")}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save model
    print("\nSaving model...")
    model_info = {
        'model': final_model,
        'feature_indices': best_idx,
        'feature_names': column_names[best_idx].tolist(),
        'group_mapping': group_label_mapping,
        'class_names': class_names,
        'best_params': best_params,
        'train_subjects': list(train_subjects),
        'test_subjects': list(test_subjects),
        'overfitting_risk': overfitting_risk,
        'age_confounding': age_confounding,
        'aggregation_method': 'mean'
    }
    
    model_file = os.path.join(plot_path, f'model_{ratio_str.replace(":", "to")}.pkl')
    joblib.dump(model_info, model_file)
    print(f"✓ Model saved: {model_file}")
    
    # Save metadata
    metadata_file = os.path.join(plot_path, f'model_metadata_{ratio_str.replace(":", "to")}.txt')
    with open(metadata_file, 'w') as f:
        f.write(f"MODEL METADATA - {ratio_str}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Train subjects: {len(train_subjects)}\n")
        f.write(f"Test subjects: {len(test_subjects)}\n")
        f.write(f"Features: {len(best_idx)}\n")
        f.write(f"Overfitting risk: {overfitting_risk}\n")
        f.write(f"Age confounding: {age_confounding}\n\n")
        f.write(f"Performance:\n")
        f.write(f"  Precision: {precision:.3f}\n")
        f.write(f"  Recall: {recall:.3f}\n")
        f.write(f"  F1: {f1:.3f}\n")
        f.write(f"  AUC: {auc_score:.3f}\n")
        f.write(f"  CV: {np.mean(scores):.3f} ± {np.std(scores):.3f}\n")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'cv_mean': np.mean(scores),
        'cv_std': np.std(scores),
        'overfitting_risk': overfitting_risk,
        'age_confounding': age_confounding
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Configuration
    PSM_DATA_DIR = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\Resultados\PSM_datasets'
    RESULTS_PATH = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\Resultados'
    DATA_TYPE = 'ce'
    SPACE = 'roi'
    
    # ML parameters
    GROUP1 = 'PSEN1'
    RATIOS = ['1:1', '2:1', '5:1']
    CLASS_NAMES = ['Control', 'PSEN1']
    GROUP_MAPPING = None
    
    # Create configuration
    config = MLConfig(PSM_DATA_DIR, RESULTS_PATH, DATA_TYPE, SPACE)
    
    print(f"\n{'#'*80}")
    print(f"# MACHINE LEARNING PIPELINE - SUBJECT-LEVEL VERSION")
    print(f"# Training on {len(RATIOS)} PSM-matched datasets")
    print(f"{'#'*80}\n")
    
    results = {}
    
    for ratio in RATIOS:
        ratio_filename = ratio.replace(':', 'to')
        data_file = os.path.join(
            PSM_DATA_DIR,
            f'Data_matched_{DATA_TYPE}_{SPACE}_{GROUP1}_{ratio_filename}.feather'
        )
        
        if not os.path.exists(data_file):
            print(f"\n⚠ File not found: {data_file}")
            continue
        
        result = train_model_on_psm_data(
            data_file, config, GROUP1, ratio,
            CLASS_NAMES, GROUP_MAPPING
        )
        
        results[ratio] = result
    
    # Summary
    print(f"\n{'#'*80}")
    print(f"# TRAINING COMPLETED")
    print(f"{'#'*80}\n")
    
    print("COMPARATIVE PERFORMANCE:")
    print("="*80)
    print(f"{'Ratio':<10} {'AUC':<8} {'F1':<8} {'CV Acc':<12} {'Risk':<15} {'Age Conf'}")
    print("="*80)
    
    for ratio, res in results.items():
        print(f"{ratio:<10} "
              f"{res['auc']:.3f}   "
              f"{res['f1']:.3f}   "
              f"{res['cv_mean']:.3f}±{res['cv_std']:.3f}   "
              f"{res['overfitting_risk']:<15} "
              f"{'YES' if res['age_confounding'] else 'NO'}")
    
    print("="*80)
