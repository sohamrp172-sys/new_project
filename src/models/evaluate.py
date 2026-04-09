"""
Model evaluation module for computing metrics and generating reports.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import json
import os


class ModelEvaluator:
    """
    Evaluates model performance with various metrics.
    """
    
    def __init__(self):
        self.metrics_report = {}
    
    def compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                       y_pred_proba: np.ndarray = None, model_name: str = "Model") -> dict:
        """
        Compute comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
            model_name: Name of model
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add ROC-AUC if probabilities available
        if y_pred_proba is not None:
            try:
                # Handle multi-class
                if y_pred_proba.ndim > 1:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except Exception as e:
                print(f"Warning: Could not compute ROC-AUC: {e}")
                metrics['roc_auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        self.metrics_report[model_name] = metrics
        
        return metrics
    
    def evaluate_models(self, models: dict, X_test: pd.DataFrame, 
                       y_test: pd.Series) -> dict:
        """
        Evaluate all models on test set.
        
        Args:
            models: Dictionary of models
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation results
        """
        results = {}
        
        for model_name, model_info in models.items():
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name}...")
            print(f"{'='*50}")
            
            model = model_info['model']
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Compute metrics
            metrics = self.compute_metrics(y_test, y_pred, y_pred_proba, model_name)
            
            # Print results
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1 Score:  {metrics['f1']:.4f}")
            if metrics.get('roc_auc'):
                print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"Confusion Matrix:")
            print(f"  TN: {metrics['tn']}, FP: {metrics['fp']}")
            print(f"  FN: {metrics['fn']}, TP: {metrics['tp']}")
            
            results[model_name] = metrics
        
        return results
    
    def save_metrics(self, metrics: dict, save_path: str):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics
            save_path: Path to save metrics
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nMetrics saved to {save_path}")


def evaluate_pipeline(models: dict, X_test: pd.DataFrame, y_test: pd.Series,
                     experiments_path: str) -> dict:
    """
    Complete evaluation pipeline.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        experiments_path: Path to save results
        
    Returns:
        Dictionary of evaluation results
    """
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_models(models, X_test, y_test)
    
    # Save metrics
    metrics_path = os.path.join(experiments_path, 'metrics.json')
    evaluator.save_metrics(results, metrics_path)
    
    # Find best model
    if 'roc_auc' in {k: v for v in results.values() for k in v}:
        best_model = max(
            results.items(),
            key=lambda x: x[1].get('roc_auc', 0)
        )
        print(f"\nBest model: {best_model[0]}")
        print(f"ROC-AUC: {best_model[1].get('roc_auc', 'N/A'):.4f}")
    
    return results


if __name__ == "__main__":
    print("Evaluation module for model assessment")
