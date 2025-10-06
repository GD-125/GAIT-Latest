# File: tests/test_federated.py
# Federated learning tests

import pytest
import torch
from src.federated.aggregation import FederatedAggregator

def test_fedavg_aggregation():
    models = [
        {'layer1': torch.randn(10, 5)},
        {'layer1': torch.randn(10, 5)}
    ]
    
    aggregator = FederatedAggregator(strategy='fedavg')
    result = aggregator.aggregate(models)
    
    assert 'layer1' in result
    assert result['layer1'].shape == (10, 5)