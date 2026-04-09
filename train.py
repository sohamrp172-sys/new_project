"""
Main training pipeline orchestrator.
"""
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.merge_datasets import (
    load_all_datasets, merge_datasets, log_data_version
)
from src.data.preprocess import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.registry import register_best_model
from sklearn.model_selection import train_test_split


def run_full_pipeline():
    """Run complete ML pipeline from data loading to model registration."""
    
    print("\n" + "="*70)
    print("TELCO CHURN PREDICTION - FULL TRAINING PIPELINE")
    print("="*70)
    
    # Configuration
    raw_data_path = "data/raw"
    processed_data_path = "data/processed"
    models_path = "models"
    experiments_path = "experiments"
    
    # Ensure directories exist
    os.makedirs(raw_data_path, exist_ok=True)
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(experiments_path, exist_ok=True)
    
    try:
        # ============ STEP 1: DATA MERGING ============
        print("\n" + "="*70)
        print("STEP 1: MERGING DATASETS")
        print("="*70)
        
        datasets = load_all_datasets(raw_data_path)
        merged_df = merge_datasets(datasets)
        
        # Sample data for faster processing (use 50k rows for initial test)
        # For production, use full dataset
        if len(merged_df) > 50000:
            print(f"\nSampling {50000} rows for faster processing...")
            merged_df = merged_df.sample(n=50000, random_state=42)
            print(f"Sampled dataset shape: {merged_df.shape}")
        
        # Log data version
        log_data_version(merged_df, experiments_path)
        print("[OK] Data merged and versioned successfully")
        
        # ============ STEP 2: PREPROCESSING ============
        print("\n" + "="*70)
        print("STEP 2: DATA PREPROCESSING")
        print("="*70)
        
        # Separate features and target
        target_col = 'Churn'
        if target_col not in merged_df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        X = merged_df.drop(columns=[target_col])
        y = merged_df[target_col].copy()
        
        # Drop rows with NaN in target variable
        valid_idx = ~y.isna()
        X = X[valid_idx].reset_index(drop=True)
        y = y[valid_idx].reset_index(drop=True)
        
        # Also ensure target is numeric and has no NaN
        y = pd.to_numeric(y, errors='coerce')
        valid_idx = ~y.isna()
        X = X[valid_idx].reset_index(drop=True)
        y = y[valid_idx].reset_index(drop=True)
        
        print(f"Features shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        # Preprocess data
        print("Processing data (handling missing values, encoding, scaling)...")
        preprocessor = DataPreprocessor()
        X_processed = preprocessor.fit_transform(X, target_col=None)
        
        # Verify no NaN values remain
        if X_processed.isnull().any().any():
            print("Warning: NaN values still exist in features, filling with 0...")
            X_processed = X_processed.fillna(0)
        
        print("[OK] Data preprocessed successfully")
        
        # ============ STEP 3: FEATURE ENGINEERING ============
        print("\n" + "="*70)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*70)
        
        # Engineer features
        print("Generating engineered features...")
        engineer = FeatureEngineer()
        X_engineered = engineer.engineer_features(X_processed)
        
        # Select top features
        X_engineered['target_temp'] = y.values
        top_features = engineer.select_top_features(X_engineered, target_col='target_temp', top_n=20)
        X_engineered = X_engineered.drop(columns=['target_temp'])
        
        # Keep top features + engineered ones (limit to avoid curse of dimensionality)
        cols_to_keep = [col for col in X_engineered.columns if col in top_features or 'squared' in col or 'x_' in col]
        if len(cols_to_keep) < len(top_features):
            cols_to_keep = top_features[:min(20, len(top_features))]
        
        X_final = X_engineered[cols_to_keep] if cols_to_keep else X_engineered.iloc[:, :20]
        
        print(f"Final features shape: {X_final.shape}")
        print("[OK] Features engineered successfully")
        
        # ============ STEP 4: MODEL TRAINING ============
        print("\n" + "="*70)
        print("STEP 4: MODEL TRAINING")
        print("="*70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train models
        trainer = ModelTrainer()
        trainer.train_all_models(X_train, y_train)
        
        print("[OK] Models trained successfully")
        
        # ============ STEP 5: MODEL EVALUATION ============
        print("\n" + "="*70)
        print("STEP 5: MODEL EVALUATION")
        print("="*70)
        
        evaluator = ModelEvaluator()
        eval_results = evaluator.evaluate_models(trainer.models, X_test, y_test)
        
        # Save metrics
        import json
        metrics_path = os.path.join(experiments_path, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(eval_results, f, indent=4)
        
        print(f"[OK] Metrics saved to {metrics_path}")
        
        # ============ STEP 6: MODEL REGISTRATION ============
        print("\n" + "="*70)
        print("STEP 6: MODEL REGISTRATION")
        print("="*70)
        
        best_model_path = register_best_model(
            trainer, eval_results, models_path
        )
        
        print("\n" + "="*70)
        print("[OK] PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nKey Artifacts:")
        print(f"  - Merged data: {merged_df.shape}")
        print(f"  - Processed data: {X_processed.shape}")
        print(f"  - Engineered features: {X_final.shape}")
        print(f"  - Best model: {best_model_path}")
        print(f"  - Metrics: {metrics_path}")
        print(f"  - Registry: {os.path.join(models_path, 'registry.json')}")
        
        return True
    
    except Exception as e:
        print(f"\n[ERROR] PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_full_pipeline()
    sys.exit(0 if success else 1)
