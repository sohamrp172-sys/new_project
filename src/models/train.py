"""
Model training module.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import json
from datetime import datetime

from src.models.evaluate import evaluate_pipeline
from src.models.registry import register_best_model


class ModelTrainer:
    """
    Trains and manages binary classification models.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with model and training info
        """
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        
        print(f"Training accuracy: {train_score:.4f}")
        
        return {
            'name': 'logistic_regression',
            'model': model,
            'hyperparameters': {
                'max_iter': 1000,
                'random_state': self.random_state
            },
            'train_score': train_score
        }
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with model and training info
        """
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        
        print(f"Training accuracy: {train_score:.4f}")
        
        return {
            'name': 'random_forest',
            'model': model,
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': self.random_state
            },
            'train_score': train_score
        }
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train all models.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        lr_info = self.train_logistic_regression(X_train, y_train)
        rf_info = self.train_random_forest(X_train, y_train)
        
        self.models['logistic_regression'] = lr_info
        self.models['random_forest'] = rf_info
        
        print("\n" + "="*50)
        print("All models trained successfully!")
        print("="*50)
    
    def get_best_model(self, metric_scores: dict):
        """
        Select best model based on metric scores.
        
        Args:
            metric_scores: Dictionary of model scores by metric
        """
        if 'roc_auc' in metric_scores:
            scores = metric_scores['roc_auc']
            best_model_name = max(scores, key=scores.get)
            self.best_model = self.models[best_model_name]['model']
            self.best_model_name = best_model_name
            
            print(f"\nBest model selected: {best_model_name}")
            print(f"ROC-AUC: {scores[best_model_name]:.4f}")


def train_pipeline(data_path: str, models_path: str, 
                   experiments_path: str, target_col: str = 'Churn',
                   test_size: float = 0.2) -> tuple:
    """
    Complete training pipeline.
    
    Args:
        data_path: Path to preprocessed data
        models_path: Path to save models
        experiments_path: Path to save experiment info
        target_col: Target column name
        test_size: Test set size
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, trainer)
    """
    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # ensure target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # drop any rows with missing target values to avoid errors during stratification
    if df[target_col].isna().any():
        na_count = df[target_col].isna().sum()
        print(f"Warning: {na_count} rows have NaN in '{target_col}' - dropping them")
        df = df.dropna(subset=[target_col])
        print(f"Shape after dropping NaNs: {df.shape}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train)
    
    # Save training info
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'data_shape': df.shape,
        'train_test_split': test_size,
        'train_size': X_train.shape,
        'test_size': X_test.shape,
        'models_trained': list(trainer.models.keys())  
    }
    
    os.makedirs(experiments_path, exist_ok=True)
    training_file = os.path.join(experiments_path, 'training_info.json')
    with open(training_file, 'w') as f:
        json.dump(training_info, f, indent=4)
    
    print(f"\nTraining info saved to {training_file}")
    
    return X_train, X_test, y_train, y_test, trainer


if __name__ == "__main__":
    data_path = "data/processed/engineered_features.csv"
    models_path = "models"
    experiments_path = "experiments"
    
    X_train, X_test, y_train, y_test, trainer = train_pipeline(
        data_path, models_path, experiments_path
    )
    
    # evaluate trained models on test set
    eval_results = evaluate_pipeline(trainer.models, X_test, y_test, experiments_path)

    # register best model based on evaluation
    try:
        best_model_path = register_best_model(trainer, eval_results, models_dir=models_path,
                                              registry_path=os.path.join(models_path, 'registry.json'))
        print(f"Best model has been registered at {best_model_path}")
    except Exception as e:
        print(f"Could not register best model: {e}")

    print("\nTraining pipeline completed successfully!")
