"""
Model Prediction Script
Load trained models and make predictions on new data
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt


def load_trained_model(model_path):
    """
    Load a trained model from disk
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model (.pkl file)
    
    Returns:
    --------
    dict : Model information including model, features, mappings
    """
    print(f"\nLoading model from: {model_path}")
    
    model_info = joblib.load(model_path)
    
    print(f"✓ Model loaded successfully")
    print(f"  Features: {len(model_info['feature_names'])}")
    print(f"  Classes: {model_info['class_names']}")
    print(f"  Group mapping: {model_info['group_mapping']}")
    
    return model_info


def prepare_new_data(new_data, model_info):
    """
    Prepare new data for prediction using the same preprocessing as training
    
    Parameters:
    -----------
    new_data : DataFrame
        New data to predict on
    model_info : dict
        Model information from load_trained_model
    
    Returns:
    --------
    array : Prepared features ready for prediction
    """
    print("\nPreparing new data...")
    
    # Get required features
    required_features = model_info['feature_names']
    
    # Check if all features are present
    missing_features = set(required_features) - set(new_data.columns)
    if missing_features:
        print(f"⚠ Warning: Missing features: {missing_features}")
        print(f"  Filling with zeros")
        for feat in missing_features:
            new_data[feat] = 0
    
    # Select only required features in correct order
    X_new = new_data[required_features].values
    
    print(f"✓ Data prepared: {X_new.shape}")
    
    return X_new


def predict_with_model(model_info, X_new, return_proba=True):
    """
    Make predictions with loaded model
    
    Parameters:
    -----------
    model_info : dict
        Model information from load_trained_model
    X_new : array
        Prepared features
    return_proba : bool
        Whether to return probabilities
    
    Returns:
    --------
    tuple : (predictions, probabilities) or just predictions
    """
    print("\nMaking predictions...")
    
    model = model_info['model']
    
    # Predict
    predictions = model.predict(X_new)
    
    if return_proba:
        probabilities = model.predict_proba(X_new)
        print(f"✓ Predictions completed: {len(predictions)} samples")
        return predictions, probabilities
    else:
        print(f"✓ Predictions completed: {len(predictions)} samples")
        return predictions


def evaluate_predictions(y_true, y_pred, y_pred_proba, class_names, output_path=None):
    """
    Evaluate predictions and optionally save results
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    y_pred_proba : array
        Prediction probabilities
    class_names : list
        Class names
    output_path : str, optional
        Path to save results
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  AUC: {auc:.3f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Save results if path provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1', 'AUC'],
            'Value': [precision, recall, f1, auc]
        })
        metrics_df.to_csv(os.path.join(output_path, 'evaluation_metrics.csv'), index=False)
        
        # Save confusion matrix plot
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Results saved to: {output_path}")


def predict_on_new_data_pipeline(model_path, new_data_path, output_path=None, 
                                 true_labels_col=None):
    """
    Complete pipeline to predict on new data
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
    new_data_path : str
        Path to new data file
    output_path : str, optional
        Path to save results
    true_labels_col : str, optional
        Name of column with true labels (for evaluation)
    
    Returns:
    --------
    DataFrame : Original data with predictions added
    """
    # Load model
    model_info = load_trained_model(model_path)
    
    # Load new data
    print(f"\nLoading new data from: {new_data_path}")
    new_data = pd.read_feather(new_data_path)
    print(f"✓ Loaded {len(new_data)} samples")
    
    # Prepare data
    X_new = prepare_new_data(new_data, model_info)
    
    # Make predictions
    predictions, probabilities = predict_with_model(model_info, X_new, return_proba=True)
    
    # Add predictions to dataframe
    result_df = new_data.copy()
    result_df['predicted_label'] = predictions
    result_df['predicted_class'] = [model_info['class_names'][int(p)] for p in predictions]
    result_df['probability_class_0'] = probabilities[:, 0]
    result_df['probability_class_1'] = probabilities[:, 1]
    
    # Reverse group mapping for readable output
    reverse_mapping = {v: k for k, v in model_info['group_mapping'].items()}
    result_df['predicted_group'] = result_df['predicted_label'].map(reverse_mapping)
    
    print(f"\nPrediction distribution:")
    print(result_df['predicted_group'].value_counts())
    
    # Evaluate if true labels are provided
    if true_labels_col and true_labels_col in new_data.columns:
        print(f"\nEvaluating predictions against '{true_labels_col}'...")
        
        # Map true labels to encoded values
        y_true_encoded = new_data[true_labels_col].map(model_info['group_mapping'])
        
        evaluate_predictions(
            y_true_encoded.values,
            predictions,
            probabilities,
            model_info['class_names'],
            output_path
        )
    
    # Save predictions
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        predictions_file = os.path.join(output_path, 'predictions.feather')
        result_df.to_feather(predictions_file)
        print(f"\n✓ Predictions saved to: {predictions_file}")
        
        # Also save as CSV for easy viewing
        csv_file = os.path.join(output_path, 'predictions.csv')
        result_df.to_csv(csv_file, index=False)
        print(f"✓ Predictions saved to: {csv_file}")
    
    return result_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Load model and examine it
    print("="*60)
    print("EXAMPLE 1: Load and examine model")
    print("="*60)
    
    MODEL_PATH = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\Resultados\graphics\ML\PSEN1_2to1_roi\model_2to1.pkl'
    
    if os.path.exists(MODEL_PATH):
        model_info = load_trained_model(MODEL_PATH)
        
        print("\nModel details:")
        print(f"  Number of estimators: {model_info['best_params'].get('n_estimators', 'N/A')}")
        print(f"  Max depth: {model_info['best_params'].get('max_depth', 'N/A')}")
        print(f"  Class weight: {model_info['best_params'].get('class_weight', 'N/A')}")
        print(f"\nTop 10 features:")
        for i, feat in enumerate(model_info['feature_names'][:10], 1):
            print(f"    {i}. {feat}")
    
    # Example 2: Predict on new data
    print("\n" + "="*60)
    print("EXAMPLE 2: Predict on new data")
    print("="*60)
    
    # Path to new data (could be validation set, new subjects, etc.)
    NEW_DATA_PATH = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\Resultados\PSM_datasets\Data_matched_ce_roi_PSEN1_5to1.feather'
    
    # Path to save predictions
    OUTPUT_PATH = r'E:\Academico\Universidad\Posgrado\Tesis\Datos\PORTABLES\Resultados\Predictions\2to1_model_on_5to1_data'
    
    if os.path.exists(MODEL_PATH) and os.path.exists(NEW_DATA_PATH):
        results = predict_on_new_data_pipeline(
            model_path=MODEL_PATH,
            new_data_path=NEW_DATA_PATH,
            output_path=OUTPUT_PATH,
            true_labels_col='group'  # Column with true labels for evaluation
        )
        
        print("\n" + "="*60)
        print("PREDICTION PIPELINE COMPLETED")
        print("="*60)
    else:
        print("\n⚠ Model or data file not found. Update paths in script.")
        print(f"  Model: {MODEL_PATH}")
        print(f"  Data: {NEW_DATA_PATH}")
