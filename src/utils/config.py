"""
Configuration loader utility.
"""
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration manager."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot notation key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self.get(key)
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration."""
        return self._config


# Global config instance
config = Config()
