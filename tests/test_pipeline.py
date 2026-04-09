"""
Basic test suite for the ML pipeline.
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_imports():
    """Test that all main modules can be imported."""
    from src.data.merge_datasets import load_all_datasets, merge_datasets
    from src.data.preprocess import DataPreprocessor
    from src.features.feature_engineering import FeatureEngineer
    from src.models.train import ModelTrainer
    from src.models.evaluate import ModelEvaluator
    from src.models.registry import ModelRegistry
    from src.monitoring.monitor import DataDriftMonitor
    
    assert True


def test_data_preprocessor():
    """Test data preprocessing functionality."""
    from src.data.preprocess import DataPreprocessor
    
    # Create dummy data
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, np.nan, 5.0],
        'feature2': [10, 20, 30, 40, 50],
        'category': ['A', 'B', 'A', 'B', 'A'],
        'Churn': [0, 1, 0, 1, 0]
    })
    
    preprocessor = DataPreprocessor()
    result = preprocessor.fit_transform(df)
    
    # Check no NaN values remain
    assert result.isnull().sum().sum() == 0
    assert result.shape[0] == df.shape[0]


def test_feature_engineer():
    """Test feature engineering functionality."""
    from src.features.feature_engineering import FeatureEngineer
    
    # Create dummy data
    df = pd.DataFrame({
        'tenure': [1, 12, 24, 36, 48],
        'monthlycharges': [20, 50, 75, 100, 120],
        'Churn': [1, 1, 0, 0, 0]
    })
    
    engineer = FeatureEngineer()
    result = engineer.engineer_features(df)
    
    # Check that new features were created
    assert result.shape[1] > df.shape[1]
    assert 'tenure_squared' in result.columns


def test_model_registry():
    """Test model registry functionality."""
    from src.models.registry import ModelRegistry
    from sklearn.linear_model import LogisticRegression
    
    registry = ModelRegistry('tests/test_registry.json')
    
    # Create dummy model
    model = LogisticRegression()
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model.fit(X_train, y_train)
    
    # Register model
    metrics = {'accuracy': 0.85, 'f1': 0.83}
    hyperparams = {'random_state': 42}
    
    registry.register_model(
        model, 'test_model', 'v1',
        metrics, hyperparams,
        models_dir='tests'
    )
    
    # Check registry
    assert 'test_model_v1' in registry.list_models()


def test_drift_monitor():
    """Test data drift monitoring."""
    from src.monitoring.monitor import DataDriftMonitor
    
    # Create baseline data
    baseline = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.normal(50, 10, 1000)
    })
    
    # Create new data with some drift
    new_data = pd.DataFrame({
        'feature1': np.random.normal(105, 15, 100),  # Slight drift
        'feature2': np.random.normal(50, 10, 100)
    })
    
    monitor = DataDriftMonitor(drift_threshold=0.10)
    monitor.compute_baseline_stats(baseline)
    report = monitor.detect_drift(new_data)
    
    assert 'drift_detected' in report
    assert 'details' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
