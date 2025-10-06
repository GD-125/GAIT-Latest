# File: src/federated/__init__.py
# Federated learning package initialization

"""
Federated learning modules for FE-AI system
Provides privacy-preserving distributed training capabilities
"""

from .fl_client import FederatedClient
from .fl_server import FederatedServer, CustomFedAvg
from .privacy import DifferentialPrivacy, SecureAggregation, PrivacyAccountant

__all__ = [
    'FederatedClient',
    'FederatedServer', 
    'CustomFedAvg',
    'DifferentialPrivacy',
    'SecureAggregation', 
    'PrivacyAccountant'
]