"""
Advanced model training with XGBoost, LightGBM, SMOTE, and hyperparameter tuning.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import json
from datetime import datetime
from typing import Dict, Any, Tuple

from src.utils.config import config
from src.utils.logger import logger


class AdvancedModelTrainer:
    """
    Advanced model trainer with multiple algorithms and tuning.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -1
    
    def handle_class_imbalance(self, X_train: pd.DataFrame, 
                               y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using SMOTE.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Resampled X_train, y_train
        """
        if not config.get('preprocessing.handle_imbalance', True):
            return X_train, y_train
        
        method = config.get('preprocessing.imbalance_method', 'smote')
        
        if method in ['smote', 'both']:
            logger.info("Applying SMOTE to balance classes...")
            logger.info(f"Before SMOTE: {y_train.value_counts().to_dict()}")
            
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            logger.info(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
            
            return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)
        
        return X_train, y_train
    
    def train_logistic_regression(self, X_train: pd.DataFrame, 
                                  y_train: pd.Series) -> Dict[str, Any]:
        """Train Logistic Regression."""
        logger.info("\n" + "="*50)
        logger.info("Training Logistic Regression...")
        logger.info("="*50)
        
        model = LogisticRegression(
            max_iter=config.get('models.logistic_regression.max_iter', 1000),
            class_weight=config.get('models.logistic_regression.class_weight', 'balanced'),
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        return {
            'name': 'logistic_regression',
            'model': model,
            'hyperparameters': {
                'max_iter': config.get('models.logistic_regression.max_iter', 1000),
                'class_weight': config.get('models.logistic_regression.class_weight', 'balanced')
            },
            'train_score': train_score
        }
    
    def train_random_forest(self, X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           tune: bool = False) -> Dict[str, Any]:
        """Train Random Forest with optional hyperparameter tuning."""
        logger.info("\n" + "="*50)
        logger.info("Training Random Forest...")
        logger.info("="*50)
        
        if tune and config.get('models.hyperparameter_tuning.enabled', False):
            logger.info("Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': config.get('models.random_forest.n_estimators', [100, 200]),
                'max_depth': config.get('models.random_forest.max_depth', [10, 15, 20]),
                'min_samples_split': config.get('models.random_forest.min_samples_split', [5, 10]),
                'min_samples_leaf': config.get('models.random_forest.min_samples_leaf', [2, 5])
            }
            
            base_model = RandomForestClassifier(
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=config.get('models.hyperparameter_tuning.n_iter', 20),
                cv=config.get('models.hyperparameter_tuning.cv_folds', 3),
                scoring=config.get('models.optimization_metric', 'roc_auc'),
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            model = search.best_estimator_
            
            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score: {search.best_score_:.4f}")
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        return {
            'name': 'random_forest',
            'model': model,
            'hyperparameters': {
                'n_estimators': 100, 'max_depth': 15,
                'min_samples_split': 10, 'min_samples_leaf': 5,
                'class_weight': 'balanced'
            },
            'train_score': train_score
        }
    
    def train_xgboost(self, X_train: pd.DataFrame, 
                     y_train: pd.Series,
                     tune: bool = False) -> Dict[str, Any]:
        """Train XGBoost with optional hyperparameter tuning."""
        logger.info("\n" + "="*50)
        logger.info("Training XGBoost...")
        logger.info("="*50)
        
        if tune and config.get('models.hyperparameter_tuning.enabled', False):
            logger.info("Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': config.get('models.xgboost.n_estimators', [100, 200]),
                'max_depth': config.get('models.xgboost.max_depth', [5, 7, 10]),
                'learning_rate': config.get('models.xgboost.learning_rate', [0.01, 0.1, 0.3])
            }
            
            base_model = xgb.XGBClassifier(
                scale_pos_weight=config.get('models.xgboost.scale_pos_weight', 8),
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=config.get('models.hyperparameter_tuning.n_iter', 20),
                cv=config.get('models.hyperparameter_tuning.cv_folds', 3),
                scoring=config.get('models.optimization_metric', 'roc_auc'),
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            model = search.best_estimator_
            
            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score: {search.best_score_:.4f}")
        else:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                scale_pos_weight=8,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        return {
            'name': 'xgboost',
            'model': model,
            'hyperparameters': {
                'n_estimators': 200, 'max_depth': 7,
                'learning_rate': 0.1, 'scale_pos_weight': 8
            },
            'train_score': train_score
        }
    
    def train_lightgbm(self, X_train: pd.DataFrame, 
                      y_train: pd.Series,
                      tune: bool = False) -> Dict[str, Any]:
        """Train LightGBM with optional hyperparameter tuning."""
        logger.info("\n" + "="*50)
        logger.info("Training LightGBM...")
        logger.info("="*50)
        
        if tune and config.get('models.hyperparameter_tuning.enabled', False):
            logger.info("Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': config.get('models.lightgbm.n_estimators', [100, 200]),
                'max_depth': config.get('models.lightgbm.max_depth', [5, 10, 15]),
                'learning_rate': config.get('models.lightgbm.learning_rate', [0.01, 0.1])
            }
            
            base_model = lgb.LGBMClassifier(
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=config.get('models.hyperparameter_tuning.n_iter', 20),
                cv=config.get('models.hyperparameter_tuning.cv_folds', 3),
                scoring=config.get('models.optimization_metric', 'roc_auc'),
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            model = search.best_estimator_
            
            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score: {search.best_score_:.4f}")
        else:
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        return {
            'name': 'lightgbm',
            'model': model,
            'hyperparameters': {
                'n_estimators': 200, 'max_depth': 10,
                'learning_rate': 0.1, 'class_weight': 'balanced'
            },
            'train_score': train_score
        }
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train all configured models."""
        logger.info("="*70)
        logger.info("MODEL TRAINING")
        logger.info("="*70)
        
        models_to_train = config.get('models.train_models', 
                                     ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm'])
        tune = config.get('models.hyperparameter_tuning.enabled', False)
        
        for model_name in models_to_train:
            try:
                if model_name == 'logistic_regression':
                    model_info = self.train_logistic_regression(X_train, y_train)
                elif model_name == 'random_forest':
                    model_info = self.train_random_forest(X_train, y_train, tune=tune)
                elif model_name == 'xgboost':
                    model_info = self.train_xgboost(X_train, y_train, tune=tune)
                elif model_name == 'lightgbm':
                    model_info = self.train_lightgbm(X_train, y_train, tune=tune)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                self.models[model_name] = model_info
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        logger.info("\n" + "="*50)
        logger.info(f"Successfully trained {len(self.models)} models")
        logger.info("="*50)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, 
                               weights=[0.9, 0.1], random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y_series = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    trainer = AdvancedModelTrainer()
    trainer.train_all_models(X_train, y_train)
    
    print(f"\nTrained models: {list(trainer.models.keys())}")
