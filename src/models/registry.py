"""
Model registry for versioning and managing models.
"""
import json
import os
import joblib
from datetime import datetime
from typing import Optional, Dict, Any


class ModelRegistry:
    """
    Manages model versioning and storage.
    """
    
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()
    
    def _load_registry(self) -> dict:
        """Load registry from file or create new one."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {'models': {}, 'best_model': None}
    
    def _save_registry(self):
        """Save registry to file."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)
    
    def register_model(self, model: Any, model_name: str, version: str,
                      metrics: dict, hyperparameters: dict,
                      models_dir: str = "models") -> str:
        """
        Register and save a model.
        
        Args:
            model: Trained model object
            model_name: Name of model (e.g., 'logistic_regression')
            version: Version string (e.g., 'v1')
            metrics: Dictionary of performance metrics
            hyperparameters: Dictionary of model hyperparameters
            models_dir: Directory to save models
            
        Returns:
            Path to saved model
        """
        # Create model file name
        model_filename = f"{model_name}_{version}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        
        # Save model
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(model, model_path)
        
        print(f"Saved model to {model_path}")
        
        # Register in registry
        model_id = f"{model_name}_{version}"
        
        self.registry['models'][model_id] = {
            'name': model_name,
            'version': version,
            'path': model_path,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'hyperparameters': hyperparameters
        }
        
        self._save_registry()
        
        print(f"Registered model: {model_id}")
        
        return model_path
    
    def set_best_model(self, model_id: str):
        """
        Set the best model in registry.
        
        Args:
            model_id: ID of best model
        """
        if model_id in self.registry['models']:
            self.registry['best_model'] = model_id
            self._save_registry()
            print(f"Set best model to: {model_id}")
        else:
            print(f"Model {model_id} not found in registry")
    
    def get_best_model(self) -> Optional[str]:
        """Get path to best model."""
        if self.registry['best_model']:
            model_id = self.registry['best_model']
            return self.registry['models'][model_id]['path']
        return None
    
    def load_model(self, model_id: str) -> Any:
        """
        Load a model from registry.
        
        Args:
            model_id: ID of model to load
            
        Returns:
            Loaded model
        """
        if model_id not in self.registry['models']:
            raise ValueError(f"Model {model_id} not found in registry")
        
        model_path = self.registry['models'][model_id]['path']
        model = joblib.load(model_path)
        
        print(f"Loaded model from {model_path}")
        
        return model
    
    def load_best_model(self) -> Any:
        """Load the best model."""
        best_id = self.registry['best_model']
        if not best_id:
            raise ValueError("No best model set in registry")
        
        return self.load_model(best_id)
    
    def get_registry_info(self) -> dict:
        """Get registry information."""
        return self.registry
    
    def list_models(self) -> list:
        """List all registered models."""
        return list(self.registry['models'].keys())


def register_best_model(trainer, evaluator_results: dict,
                       models_dir: str = "models",
                       registry_path: str = "models/registry.json",
                       metric: str = "f1") -> str:
    """
    Register best model based on evaluation results.

    Args:
        trainer: ModelTrainer instance with trained models
        evaluator_results: Results from ModelEvaluator
        models_dir: Directory to save models
        registry_path: Path to registry file
        metric: Metric to use for model selection ('f1', 'roc_auc', 'recall')

    Returns:
        Path to best model
    """
    best_model_name = None
    best_score      = -1
    best_metrics    = None

    for model_name, metrics in evaluator_results.items():
        score = metrics.get(metric, 0) or 0
        if score > best_score:
            best_score      = score
            best_model_name = model_name
            best_metrics    = metrics

    if not best_model_name:
        raise ValueError("No valid models to register")
    
    # Get model and hyperparameters
    model = trainer.models[best_model_name]['model']
    hyperparams = trainer.models[best_model_name]['hyperparameters']
    
    # Register
    registry = ModelRegistry(registry_path)
    model_path = registry.register_model(
        model=model,
        model_name=best_model_name,
        version='v1',
        metrics=best_metrics,
        hyperparameters=hyperparams,
        models_dir=models_dir
    )
    
    # Set as best
    registry.set_best_model(f"{best_model_name}_v1")
    
    print(f"\n{'='*50}")
    print(f"Best Model Registered: {best_model_name}")
    print(f"{metric.upper()}: {best_score:.4f}")
    print(f"Path: {model_path}")
    print(f"{'='*50}")
    
    return model_path


if __name__ == "__main__":
    print("Model registry for versioning")
