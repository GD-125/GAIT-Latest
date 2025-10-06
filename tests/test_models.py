# File: tests/test_models.py
# Model tests

import pytest
import torch
import numpy as np
from src.models.gait_detector import GaitDetector
from src.models.disease_classifier import DiseaseClassifier

def test_gait_detector_initialization():
    model = GaitDetector()
    assert model is not None

def test_gait_detector_forward_pass():
    model = GaitDetector()
    x = torch.randn(4, 9, 250)
    output = model(x)
    assert output.shape == (4, 2)

def test_disease_classifier_initialization():
    model = DiseaseClassifier()
    assert model is not None

def test_disease_classifier_forward_pass():
    model = DiseaseClassifier()
    x = torch.randn(4, 100, 256)
    output = model(x)
    assert output.shape[0] == 4

def test_model_save_load(temp_model_path):
    model = GaitDetector()
    torch.save(model.state_dict(), temp_model_path)
    
    model2 = GaitDetector()
    model2.load_state_dict(torch.load(temp_model_path))
    assert True