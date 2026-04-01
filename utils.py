"""
EEG Data Processing Utilities - Compatible Version
Simplified but maintains compatibility with existing scripts

Functions needed by the scripts:
- Script 2 (neuroharmonize): mapsDrop, negativeTest, select, renameModel, renameDatabases,
  covars, covarsGen, extract_components_interes, rename_cols, organizarDataFrame, 
  graf, graf_DB, save_complete
- Script 4 (training): mapa_de_correlacion, grid_search, randomFo, primeras_carateristicas,
  curva_de_aprendizaje, curva_validacion3, curva_validacion4, features_best3,
  plot_confusion_matrix, computerprecision
"""

import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import (
    cross_val_score, learning_curve, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score


# ============================================================================
# HARMONIZATION FUNCTIONS (for script 2)
# ============================================================================

def mapsDrop(data, filtGroup=None, visit1=None, visit2=None):
    """Filter data by group and visit"""
    if filtGroup is not None:
        data = data[data['group'] == filtGroup]
    if visit1 is not None:
        data = data[data['visit'] == visit1]
    if visit2 is not None:
        data = data[data['visit'] == visit2]
    return data.reset_index(drop=True)


def negativeTest(data_array):
    """Count negative values in array"""
    return np.sum(data_array < 0)


def covars(data):
    """
    Extract covariates for harmonization.

    Includes SITE (batch) and age (biological covariate).
    Sex is intentionally excluded: 47% of controls (Seoul, n=210) and 45% of
    PSEN1 carriers (Medellin_ld, n=41) have no sex data. Imputing a neutral
    value (0.5) for entire sites is not physiologically defensible, so age
    alone is used as the biological covariate to protect within-group variation.

    Returns
    -------
    tuple : (processed_data, covariates_dict)
    """
    covars_dict = {
        'SITE': data['SITE'].to_numpy(),
        'age':  data['age'].to_numpy(),
    }

    cols_to_remove = [
        'group', 'subject', 'age', 'sex', 'education',
        'group_sl',    'age_sl',    'SITE_sl',
        'group_coh',   'age_coh',   'SITE_coh',
        'group_ent',   'age_ent',   'SITE_ent',
        'group_cross', 'age_cross', 'SITE_cross',
    ]

    data_processed = data.copy()
    for col in cols_to_remove:
        if col in data_processed.columns:
            data_processed.drop(col, axis=1, inplace=True)

    return data_processed, covars_dict


def covarsGen(data):
    """Extract covariates including genetic information"""
    # Similar to covars but may include additional genetic covariates
    return covars(data)


def extract_components_interes(data, components):
    """
    Filter dataframe to keep only specified components
    
    Parameters:
    -----------
    data : DataFrame
        Input dataframe
    components : list
        List of component names to keep (e.g., ['C1', 'C2', ...])
    
    Returns:
    --------
    DataFrame : Filtered dataframe
    """
    cols_to_keep = []
    for col in data.columns:
        # Keep column if it contains any of the component names
        if any(comp in col for comp in components):
            cols_to_keep.append(col)
        # Also keep non-EEG columns
        elif col in ['participant_id', 'visit', 'group', 'condition', 'database', 
                     'age', 'sex', 'education']:
            cols_to_keep.append(col)
    
    return data[cols_to_keep]


def rename_cols(data, original_data, group1, group2):
    """Rename columns after harmonization"""
    # Add back metadata columns
    metadata_cols = ['participant_id', 'visit', 'group', 'condition', 'database']
    
    for col in metadata_cols:
        if col in original_data.columns and col not in data.columns:
            data[col] = original_data[col].values
    
    return data


def organizarDataFrame(harmonized_data, database_col, metric, original_data, space, ica):
    """
    Organize harmonized dataframe
    
    Parameters:
    -----------
    harmonized_data : DataFrame
        Harmonized data
    database_col : array
        Database column values
    metric : str
        Metric type
    original_data : DataFrame
        Original dataframe
    space : str
        'roi' or 'ic'
    ica : str
        ICA configuration
    
    Returns:
    --------
    DataFrame : Organized dataframe
    """
    result = harmonized_data.copy()
    result['database'] = database_col
    
    # Add back other metadata
    metadata_cols = ['participant_id', 'visit', 'group', 'condition']
    for col in metadata_cols:
        if col in original_data.columns and col not in result.columns:
            result[col] = original_data[col].values
    
    # Add Gamma for power metric if needed
    if metric == 'power':
        result = add_Gamma(result, space, ica)
    
    return result


def add_Gamma(data, space, ica):
    """Add Gamma band columns if missing"""
    # This is a simplified version - implement full logic if needed
    return data


def save_complete(filename, data, original_data, output_path, group2, group1):
    """
    Save harmonized data
    
    Parameters:
    -----------
    filename : str
        Output filename (without extension)
    data : DataFrame
        Data to save
    original_data : DataFrame
        Original dataframe
    output_path : str
        Directory to save to
    group2 : str
        Group 2 name
    group1 : str
        Group 1 name
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f'{filename}.feather')
    data.reset_index(drop=True).to_feather(output_file)
    print(f"✓ Saved: {output_file}")


def renameModel(data):
    """Split data by database for model naming"""
    noGene = data[data['database'] == 0.0]
    Gene = data[data['database'] == 1.0]
    return noGene, Gene


def renameDatabases(data):
    """Split data by database"""
    Biomarcadores = data[data['database'] == 0.0]
    Duque = data[data['database'] == 2.0]
    SRM = data[data['database'] == 3.0]
    CHBMP = data[data['database'] == 1.0]
    return Biomarcadores, Duque, SRM, CHBMP


def select(data, metric, OneBand=None, WithoutBand=None, Gamma=None, space='roi', spatial_matrix='54x10'):
    """
    Select specific metrics and frequency bands
    
    Parameters:
    -----------
    data : DataFrame
        Input dataframe
    metric : str
        Metric to select
    OneBand : str, optional
        Keep only this band
    WithoutBand : str, optional
        Remove this band
    Gamma : str, optional
        Special handling for gamma
    space : str
        'roi' or 'ic'
    spatial_matrix : str
        ICA configuration
    
    Returns:
    --------
    tuple : (title, filtered_data)
    """
    # Simplified version - just return data as is with a title
    if metric == 'All':
        title = 'All_metrics_all_bands'
    else:
        title = f'Only_{metric}_metric'
    
    return title, data


def graf(path, columns, *args, **kwargs):
    """Plot harmonization comparison graphs"""
    # Placeholder - implement if needed
    pass


def graf_DB(path, columns, *args, **kwargs):
    """Plot database comparison graphs"""
    # Placeholder - implement if needed
    pass


# ============================================================================
# MACHINE LEARNING FUNCTIONS (for script 4)
# ============================================================================

def mapa_de_correlacion(data, path_plot, var):
    """
    Create correlation heatmap and remove highly correlated features
    
    Parameters:
    -----------
    data : DataFrame
        Input dataframe
    path_plot : str
        Path to save plots
    var : str
        Variable name for labeling
    
    Returns:
    --------
    DataFrame : Data with highly correlated features removed
    """
    data_numeric = data.select_dtypes(include=np.number)
    
    # Calculate correlation matrix
    correlation_matrix = data_numeric.corr()
    
    # Plot before
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                annot_kws={"size": 5}, cbar=True)
    plt.title(f"Correlation Matrix for {var} Ratio", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(path_plot, 'correlation_before.png'))
    plt.close()
    
    # Find highly correlated features
    threshold = 0.8
    features_to_drop = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                if colname not in features_to_drop:
                    features_to_drop.append(colname)
    
    # Remove features
    data_reduced = data_numeric.drop(columns=features_to_drop)
    
    # Plot after
    new_correlation_matrix = data_reduced.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(new_correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title(f"Reduced Correlation Matrix for {var} Ratio", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(path_plot, 'correlation_after.png'))
    plt.close()
    
    # Save
    feather_file_path = os.path.join(path_plot, 'Data_integration_corr.feather')
    data_reduced.reset_index(drop=True).to_feather(feather_file_path)
    
    print(f"Removed {len(features_to_drop)} highly correlated features")
    
    return data_reduced


def grid_search():
    """Return Random Forest hyperparameter grid"""
    random_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 15],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    return random_grid


def randomFo(random_grid, X_train, y_train):
    """
    Perform Random Forest training with RandomizedSearchCV
    
    Parameters:
    -----------
    random_grid : dict
        Hyperparameter grid
    X_train : array
        Training features
    y_train : array
        Training labels
    
    Returns:
    --------
    RandomizedSearchCV : Fitted model
    """
    forestclf_grid = RandomForestClassifier()
    rf_random = RandomizedSearchCV(
        estimator=forestclf_grid,
        param_distributions=random_grid,
        n_iter=100,
        cv=10,
        verbose=2,
        random_state=10,
        n_jobs=-1
    )
    
    rf_random.fit(X_train, y_train)
    return rf_random


def primeras_carateristicas(X_train, sorted_names, column_names, feature_scores, 
                            feat_df, sorted_idx, path_plot, var):
    """
    Get and save feature importance rankings
    
    Parameters:
    -----------
    X_train : array
        Training data
    sorted_names : list
        List to append sorted feature names
    column_names : Index
        Column names
    feature_scores : array
        Feature importance scores
    feat_df : DataFrame
        Dataframe to store results
    sorted_idx : array
        Sorted indices
    path_plot : str
        Path to save results
    var : str
        Variable name
    
    Returns:
    --------
    DataFrame : Feature importance dataframe
    """
    # Get top features
    n_features = min(50, len(feature_scores))
    
    for idx in sorted_idx[:n_features]:
        feature_name = column_names[idx]
        sorted_names.append(feature_name)
    
    # Create dataframe
    feat_df = pd.DataFrame({
        'Feature': [column_names[i] for i in sorted_idx[:n_features]],
        'Importance': feature_scores[sorted_idx[:n_features]]
    })
    
    # Save
    output_file = os.path.join(path_plot, f'best_features_{var}.txt')
    with open(output_file, 'w') as f:
        for feature in sorted_names:
            f.write(f"{feature}\n")
    
    feat_df.to_excel(os.path.join(path_plot, f'features_{var}.xlsx'), index=False)
    
    return feat_df


def curva_de_aprendizaje(sorted_names, data, model, X_train, y_train, 
                         modelos, acc_per_feature, std_per_feature, 
                         path_plot, var):
    """
    Generate learning curve by adding features incrementally
    
    Parameters:
    -----------
    sorted_names : list
        Sorted feature names
    data : DataFrame
        Full dataset
    model : estimator
        ML model
    X_train : array
        Training features
    y_train : array
        Training labels
    modelos : dict
        Dictionary to store models
    acc_per_feature : list
        List to store accuracies
    std_per_feature : list
        List to store standard deviations
    path_plot : str
        Path to save plot
    var : str
        Variable name
    """
    # This is a simplified version
    # The full implementation would incrementally add features and evaluate
    pass


def curva_validacion3(model, X_train, y_train, title, palette, var):
    """
    Plot validation curve (learning curve)
    
    Parameters:
    -----------
    model : estimator
        ML model
    X_train : array
        Training features
    y_train : array
        Training labels
    title : str
        Plot title
    palette : list
        Color palette
    var : str
        Variable name
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 0.8, 8),
        cv=10,
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    
    # Training accuracy
    plt.plot(train_sizes, train_mean, color=palette[0], 
            marker='o', markersize=5, label='Training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, 
                    train_mean - train_std, alpha=0.15, color=palette[0])
    
    # Validation accuracy
    plt.plot(train_sizes, test_mean, color=palette[1], 
            linestyle='--', marker='s', markersize=5, label='Validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, 
                    test_mean - test_std, alpha=0.15, color=palette[1])
    
    # Labels
    ratio_label = {'2to1': '2:1', '5to1': '5:1', '10to1': '10:1'}.get(var, var)
    
    plt.grid()
    plt.title(f'Validation Curve - {ratio_label}', fontsize=14)
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.5, 1.0])


def curva_validacion4(ax, model, X_train, y_train, title, palette, var, subplot_num):
    """
    Plot validation curve on existing axes (for subplots)
    
    Parameters:
    -----------
    ax : matplotlib.axes
        Axes to plot on
    model : estimator
        ML model
    X_train : array
        Training features
    y_train : array
        Training labels
    title : str
        Plot title
    palette : list
        Color palette
    var : str
        Variable name
    subplot_num : int
        Subplot number
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 0.8, 8),
        cv=10,
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot on provided axes
    ax.plot(train_sizes, train_mean, color=palette[0], 
           marker='o', markersize=5, label='Training')
    ax.fill_between(train_sizes, train_mean + train_std, 
                   train_mean - train_std, alpha=0.15, color=palette[0])
    
    ax.plot(train_sizes, test_mean, color=palette[1], 
           linestyle='--', marker='s', markersize=5, label='Validation')
    ax.fill_between(train_sizes, test_mean + test_std, 
                   test_mean - test_std, alpha=0.15, color=palette[1])
    
    ratio_label = {'2to1': '2:1', '5to1': '5:1', '10to1': '10:1'}.get(var, var)
    
    ax.grid()
    ax.set_title(f'{ratio_label}', fontsize=14)
    ax.set_xlabel('Training samples')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.set_ylim([0.5, 1.0])


def features_best3(best_features, model, data, X_train, y_train, path_plot):
    """
    Find best number of features by incremental evaluation
    
    Parameters:
    -----------
    best_features : list
        List of feature names in order of importance
    model : estimator
        ML model
    data : DataFrame
        Full dataset
    X_train : array
        Training features
    y_train : array
        Training labels
    path_plot : str
        Path to save plot
    
    Returns:
    --------
    tuple : (accuracies, stds, best_model, best_feature_indices)
    """
    acc = []
    std = []
    
    for index in range(1, min(len(best_features) + 1, 51)):  # Max 50 features
        try:
            # Get feature indices
            input_features = best_features[:index]
            input_idx = [data.columns.get_loc(c) for c in input_features if c in data.columns]
            
            if len(input_idx) == 0:
                continue
            
            # Train model
            fitted_model = model.fit(X_train[:, input_idx], y_train)
            
            # Cross-validate
            scores = cross_val_score(
                estimator=fitted_model,
                X=X_train[:, input_idx],
                y=y_train,
                cv=10,
                n_jobs=-1
            )
            
            acc.append(np.mean(scores))
            std.append(np.std(scores))
            
        except IndexError as e:
            print(f"Error at iteration {index}: {e}")
            continue
    
    if not acc:
        return [], [], None, None
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(acc) + 1), acc, color='red', label='Accuracy')
    plt.fill_between(
        range(1, len(acc) + 1),
        np.array(acc) + np.array(std),
        np.array(acc) - np.array(std),
        alpha=0.15,
        color='red'
    )
    plt.title('Learning Curve - Feature Selection')
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(path_plot, 'features_plot_best.png'))
    plt.close()
    
    # Get best configuration
    best_idx_pos = np.argmax(acc)
    best_n_features = best_idx_pos + 1
    best_features_selected = best_features[:best_n_features]
    best_feature_indices = [data.columns.get_loc(c) for c in best_features_selected if c in data.columns]
    
    # Train final model
    final_model = model.fit(X_train[:, best_feature_indices], y_train)
    
    return acc, std, final_model, best_feature_indices


def plot_confusion_matrix(path_plot, var, cm, classes, normalize=False, 
                          title='Confusion matrix', cmap=plt.cm.BuGn):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    path_plot : str
        Path to save plot
    var : str
        Variable name
    cm : array
        Confusion matrix
    classes : list
        Class names
    normalize : bool
        Whether to normalize
    title : str
        Plot title
    cmap : colormap
        Color map
    """
    ratio_label = {'2to1': '2:1', '5to1': '5:1', '10to1': '10:1'}.get(var, var)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Confusion Matrix normalize {ratio_label}")
    else:
        print(f'Confusion Matrix {ratio_label}')
    
    print(cm)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    fontsize=20,
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.savefig(os.path.join(path_plot, f'{title}_{var}.png'))
    plt.close()


def computerprecision(test_label, classes_x, output_file):
    """
    Compute and save precision, recall, F1-score
    
    Parameters:
    -----------
    test_label : array
        True labels
    classes_x : array
        Predicted labels
    output_file : str
        Path to save results
    """
    precision_test = precision_score(test_label, classes_x)
    recall_test = recall_score(test_label, classes_x)
    f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)
    
    # Print
    print(f'Precision: {precision_test}')
    print(f'Recall: {recall_test}')
    print(f'F1-score: {f1_test}')
    
    # Save
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Precision', precision_test])
        writer.writerow(['Recall', recall_test])
        writer.writerow(['F1-score', f1_test])
    
    print(f"✓ Saved metrics to: {output_file}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def remove_outliers(data, columns, max_loss_percent=5):
    """
    Remove outliers ensuring no more than 5% data loss per database
    
    Parameters:
    -----------
    data : DataFrame
        Input dataframe
    columns : list
        Columns to check for outliers
    max_loss_percent : float
        Maximum percentage of data loss allowed
    
    Returns:
    --------
    DataFrame : Data with outliers removed
    """
    data = data.copy()
    data['index_temp'] = data.index
    
    databases = data['database'].unique()
    
    for db in databases:
        db_data = data[data['database'] == db]
        outlier_indices = []
        
        # Find outliers for each column
        for col in columns:
            Q1 = np.percentile(db_data[col], 25)
            Q3 = np.percentile(db_data[col], 75)
            IQR = Q3 - Q1
            
            upper_outliers = db_data[db_data[col] >= (Q3 + 1.5 * IQR)].index.tolist()
            lower_outliers = db_data[db_data[col] <= (Q1 - 1.5 * IQR)].index.tolist()
            
            outlier_indices.extend(upper_outliers + lower_outliers)
        
        # Count repetitions
        repetitions = Counter(outlier_indices)
        
        # Find threshold
        threshold = 2
        while True:
            threshold += 1
            indices_to_remove = [idx for idx, count in repetitions.items() if count > threshold]
            
            loss_percent = len(indices_to_remove) * 100 / len(db_data)
            
            if loss_percent <= max_loss_percent:
                data = data.drop(indices_to_remove)
                print(f"Database {db}: Removed {len(indices_to_remove)} rows ({loss_percent:.2f}%)")
                break
    
    data = data.drop(columns='index_temp')
    return data.reset_index(drop=True)


if __name__ == "__main__":
    print("EEG Data Processing Utilities - Compatible Version")
    print("=" * 60)
    print("\nThis version maintains compatibility with existing scripts.")
    print("Import this module to use the utilities in your pipeline.")
