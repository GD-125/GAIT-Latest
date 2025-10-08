# File: src/utils/config.py
# Configuration management

import yaml
import os
from pathlib import Path
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for FE-AI system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration"""
        if config_path is None:
            # Try to find config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config_data = {}
        self.load()
    
    def load(self):
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                return
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            
            # Replace environment variables
            self._replace_env_vars(self.config_data)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except yaml.YAMLError as e:
            logger.error(f"Error loading config: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading config: {e}")
            raise
    
    def _replace_env_vars(self, data: Any):
        """Replace ${VAR_NAME} with environment variable values"""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    data[key] = os.environ.get(env_var)
                elif isinstance(value, (dict, list)):
                    self._replace_env_vars(value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str) and item.startswith('${') and item.endswith('}'):
                    env_var = item[2:-1]
                    data[i] = os.environ.get(env_var)
                elif isinstance(item, (dict, list)):
                    self._replace_env_vars(item)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default
                
                if value is None:
                    return default
            
            return value
            
        except Exception as e:
            logger.debug(f"Error getting config key '{key}': {e}")
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        data = self.config_data
        
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        
        data[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise
    
    def reload(self):
        """Reload configuration from file"""
        self.load()

# Global config instance
config = Config()