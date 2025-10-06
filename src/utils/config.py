# File: src/utils/config.py
# Configuration management for FE-AI System

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config_data = self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        return str(project_root / "config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'models': {
                'gait_detector': {
                    'architecture': 'CNN-BiLSTM',
                    'confidence_threshold': 0.8,
                    'sequence_length': 250,
                    'input_dim': 6,
                    'cnn_filters': 64,
                    'lstm_units': 128,
                    'dropout_rate': 0.3
                },
                'disease_classifier': {
                    'transformer': {
                        'd_model': 512,
                        'nhead': 8,
                        'num_layers': 6,
                        'dim_feedforward': 2048,
                        'dropout': 0.1,
                        'num_classes': 5
                    },
                    'xgboost': {
                        'n_estimators': 1000,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.1,
                        'reg_lambda': 1.0
                    }
                }
            },
            'federated_learning': {
                'enabled': True,
                'min_clients': 3,
                'rounds': 50,
                'privacy_budget': 1.0,
                'aggregation_method': 'FedAvg'
            },
            'system': {
                'gpu_enabled': True,
                'max_concurrent_analyses': 10,
                'log_level': 'INFO',
                'session_timeout': 7200,
                'max_file_size_mb': 100
            },
            'database': {
                'mongodb': {
                    'host': 'localhost',
                    'port': 27017,
                    'database': 'fe_ai_system',
                    'collection': 'analyses'
                }
            },
            'security': {
                'enable_2fa': False,
                'password_expiry_days': 90,
                'max_login_attempts': 5,
                'session_encryption': True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation)"""
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config_data, file, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")

# Global configuration instance
config = Config()