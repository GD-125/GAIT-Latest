# File: src/data/__init__.py
# Data handling package initialization

"""
Data handling modules for FE-AI system
Provides data loading, validation, and synthetic generation capabilities
"""

from .data_loader import DataLoader
from .data_validator import DataValidator
from .synthetic_generator import SyntheticDataGenerator

__all__ = ['DataLoader', 'DataValidator', 'SyntheticDataGenerator']

---