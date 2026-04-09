"""
Improved training pipeline with all enhancements.
"""
import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.load_data import load_and_prepare_data
from src.data.preprocess import DataPreprocessor
from src.features.advanced_features import AdvancedFeatureEngineer
from src.models.advanced_train import AdvancedModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.registry import register_best_model
from src.utils.config import config
from src.utils.logger import logger


def run_improved_pipeline():
    """Run complete improved ML pipeline."""

    logger.info("\n" + "=" * 70)
    logger.info("TELCO CHURN PREDICTION - IMPROVED TRAINING PIPELINE")
    logger.info("=" * 70)

    processed_data_path = "data/processed"
    models_path         = "models"
    experiments_path    = "experiments"

    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(experiments_path, exist_ok=True)

    try:
        # ── STEP 1: DATA LOADING ──────────────────────────────────────
        df = load_and_prepare_data()
        df.to_csv(os.path.join(processed_data_path, 'merged_dataset.csv'), index=False)
        logger.info("Saved merged data")

        # ── STEP 2: PREPROCESSING ─────────────────────────────────────
        logger.info("\n" + "=" * 70)
        logger.info("DATA PREPROCESSING")
        logger.info("=" * 70)

        target_col = 'churn'
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        X = df.drop(columns=[target_col])
        y = df[target_col].copy()

        # Drop rows with NaN target
        valid = ~y.isna()
        X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
        y = pd.to_numeric(y, errors='coerce')
        valid = ~y.isna()
        X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")

        preprocessor = DataPreprocessor()
        X_processed  = preprocessor.fit_transform(X, target_col=None)

        if X_processed.isnull().any().any():
            logger.warning("NaN values remain after preprocessing — filling with 0")
            X_processed = X_processed.fillna(0)

        # Save preprocessor so inference can reproduce the same transforms
        preprocessor.save(os.path.join(models_path, 'preprocessor.pkl'))

        preprocessed_path = os.path.join(processed_data_path, 'preprocessed_data.csv')
        X_processed_save  = X_processed.copy()
        X_processed_save['churn'] = y.values
        X_processed_save.to_csv(preprocessed_path, index=False)
        logger.info(f"Saved preprocessed data to {preprocessed_path}")

        # ── STEP 3: TRAIN / TEST SPLIT (before feature engineering) ───
        # Split FIRST so feature engineering thresholds are computed on
        # training data only — prevents data leakage.
        test_size    = config.get('data.test_size', 0.2)
        random_state = config.get('data.random_state', 42)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_processed, y,
            test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Train set (raw): {X_train_raw.shape}")
        logger.info(f"Test  set (raw): {X_test_raw.shape}")

        # ── STEP 4: FEATURE ENGINEERING ───────────────────────────────
        engineer = AdvancedFeatureEngineer()

        # Fit on training data only (is_training=True stores thresholds)
        X_train = engineer.engineer_and_select(X_train_raw, y_train, is_training=True)

        # Transform test data using stored thresholds (is_training=False)
        X_test_eng = engineer.create_domain_features(X_test_raw, is_training=False)
        X_test_eng = engineer.create_interaction_features(X_test_eng, top_n=5)
        X_test_eng = X_test_eng.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Keep only the selected features
        X_test = X_test_eng[engineer.selected_features]

        # Save feature engineer state
        engineer.save_state(os.path.join(experiments_path, 'feature_engineer_state.json'))

        # Save engineered features (train only, for SHAP background)
        engineered_path = os.path.join(processed_data_path, 'engineered_features.csv')
        X_train_save = X_train.copy()
        X_train_save['churn'] = y_train.values
        X_train_save.to_csv(engineered_path, index=False)
        logger.info(f"Saved engineered features to {engineered_path}")

        # Save feature names for API
        feature_names_path = os.path.join(experiments_path, 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump({
                'features':   X_train.columns.tolist(),
                'n_features': len(X_train.columns)
            }, f, indent=4)
        logger.info(f"Saved feature names to {feature_names_path}")

        # ── STEP 5: CLASS IMBALANCE ────────────────────────────────────
        trainer = AdvancedModelTrainer(random_state=random_state)
        X_train_bal, y_train_bal = trainer.handle_class_imbalance(X_train, y_train)

        # ── STEP 6: MODEL TRAINING ─────────────────────────────────────
        trainer.train_all_models(X_train_bal, y_train_bal)

        # ── STEP 7: EVALUATION ────────────────────────────────────────
        logger.info("\n" + "=" * 70)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 70)

        evaluator    = ModelEvaluator()
        eval_results = evaluator.evaluate_models(trainer.models, X_test, y_test)

        metrics_path = os.path.join(experiments_path, 'metrics.json')
        evaluator.save_metrics(eval_results, metrics_path)

        # ── STEP 8: MODEL REGISTRATION ────────────────────────────────
        logger.info("\n" + "=" * 70)
        logger.info("MODEL REGISTRATION")
        logger.info("=" * 70)

        # FIX: use F1 for selection — better metric for imbalanced churn data
        best_model_path = register_best_model(
            trainer, eval_results, models_path,
            metric='f1'
        )

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info(f"  Best model : {best_model_path}")
        logger.info(f"  Metrics    : {metrics_path}")
        logger.info(f"  Registry   : {os.path.join(models_path, 'registry.json')}")

        return True

    except Exception as e:
        logger.error(f"\nPIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_improved_pipeline()
    sys.exit(0 if success else 1)
