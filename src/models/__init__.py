# File: src/models/__init__.py
# Models package initialization

"""
Machine Learning models for FE-AI system
Provides gait detection and disease classification models
"""

from .gait_detector import GaitDetector, CNNBiLSTMGaitDetector
from .disease_classifier import DiseaseClassifier, TransformerDiseaseClassifier

__all__ = [
    'GaitDetector', 
    'CNNBiLSTMGaitDetector',
    'DiseaseClassifier', 
    'TransformerDiseaseClassifier'
]