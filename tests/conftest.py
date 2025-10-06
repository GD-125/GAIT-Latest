"""
File: tests/conftest.py
Pytest configuration and fixtures for FE-AI test suite
Complete production-grade test configuration
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import sys
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gait_detector import GaitDetector
from src.models.disease_classifier import DiseaseClassifier
from src.preprocessing.signal_processor import SignalProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.database.mongo_handler import MongoHandler


# ============================================================================
# Session-level Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary test data directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def project_root_path():
    """Get project root path"""
    return Path(__file__).parent.parent


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_gait_data():
    """Generate sample gait sensor data (1000 samples)"""
    num_samples = 1000
    
    data = pd.DataFrame({
        'timestamp': np.arange(num_samples) * 0.01,  # 100Hz sampling
        'accel_x': np.random.randn(num_samples) * 0.5 + 0.1,
        'accel_y': np.random.randn(num_samples) * 0.5 - 0.05,
        'accel_z': np.random.randn(num_samples) * 0.5 + 9.8,
        'gyro_x': np.random.randn(num_samples) * 0.1,
        'gyro_y': np.random.randn(num_samples) * 0.1,
        'gyro_z': np.random.randn(num_samples) * 0.1,
        'emg_1': np.abs(np.random.randn(num_samples) * 50 + 100),
        'emg_2': np.abs(np.random.randn(num_samples) * 50 + 100),
        'emg_3': np.abs(np.random.randn(num_samples) * 50 + 100),
        'label': np.random.choice(['gait', 'non_gait'], num_samples)
    })
    
    return data


@pytest.fixture
def sample_gait_tensor():
    """Generate sample gait data as PyTorch tensor"""
    # [batch_size, sequence_length, features]
    batch_size = 8
    seq_len = 100
    features = 9
    
    return torch.randn(batch_size, seq_len, features)


@pytest.fixture
def sample_large_dataset():
    """Generate large dataset for performance testing"""
    num_samples = 10000
    
    data = pd.DataFrame({
        'accel_x': np.random.randn(num_samples),
        'accel_y': np.random.randn(num_samples),
        'accel_z': np.random.randn(num_samples) + 9.8,
        'gyro_x': np.random.randn(num_samples) * 0.1,
        'gyro_y': np.random.randn(num_samples) * 0.1,
        'gyro_z': np.random.randn(num_samples) * 0.1,
        'emg_1': np.abs(np.random.randn(num_samples) * 50),
        'emg_2': np.abs(np.random.randn(num_samples) * 50),
        'emg_3': np.abs(np.random.randn(num_samples) * 50),
    })
    
    return data


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def gait_detector_model():
    """Initialize gait detector model"""
    model = GaitDetector(input_dim=9, hidden_dim=64, num_layers=2)
    return model


@pytest.fixture
def disease_classifier_model():
    """Initialize disease classifier model"""
    model = DiseaseClassifier(num_classes=5, hidden_dim=128)
    return model


@pytest.fixture
def trained_gait_model(gait_detector_model, sample_gait_tensor):
    """Provide a pre-trained gait detector for testing"""
    model = gait_detector_model
    
    # Quick training for 2 epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    X = sample_gait_tensor
    y = torch.randint(0, 2, (len(X), 1)).float()
    
    model.train()
    for _ in range(2):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model


# ============================================================================
# Preprocessing Fixtures
# ============================================================================

@pytest.fixture
def signal_processor():
    """Initialize signal processor"""
    return SignalProcessor()


@pytest.fixture
def feature_extractor():
    """Initialize feature extractor"""
    return FeatureExtractor()


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def mock_database(monkeypatch):
    """Mock database handler for testing"""
    class MockMongoHandler:
        def __init__(self):
            self.data = {}
            self.users = {}
            self.predictions = {}
        
        def insert_upload(self, document):
            upload_id = f"test_upload_{len(self.data)}"
            self.data[upload_id] = document
            return upload_id
        
        def get_upload(self, upload_id):
            return self.data.get(upload_id)
        
        def get_upload_count(self):
            return len(self.data)
        
        def get_active_users_count(self):
            return len(self.users)
        
        def get_prediction_count(self):
            return len(self.predictions)
        
        def update_upload_status(self, upload_id, status):
            if upload_id in self.data:
                self.data[upload_id]['status'] = status
                return True
            return False
        
        def query_uploads_by_patient(self, patient_id):
            return [v for v in self.data.values() if v.get('patient_id') == patient_id]
        
        def add_user(self, user_data):
            user_id = f"user_{len(self.users)}"
            self.users[user_id] = user_data
            return user_id
        
        def get_recent_activity(self, limit=10):
            return [
                {
                    'timestamp': datetime.now().isoformat(),
                    'action': 'upload',
                    'user': 'test_user',
                    'details': 'Uploaded test file'
                }
            ]
    
    return MockMongoHandler()


# ============================================================================
# Metadata Fixtures
# ============================================================================

@pytest.fixture
def sample_patient_metadata():
    """Sample patient metadata for testing"""
    return {
        'patient_id': 'TEST_PATIENT_001',
        'age': 55,
        'gender': 'Male',
        'medical_history': 'No prior neurological conditions',
        'collection_date': datetime.now().isoformat(),
        'collection_site': 'Test Hospital',
        'device_type': 'IMU Sensor Array',
        'sampling_rate': 100,
        'consent_obtained': True
    }


@pytest.fixture
def sample_prediction_results():
    """Sample prediction results for testing"""
    return {
        'prediction': 0.85,
        'confidence': 85.0,
        'model_version': 'v1.0.0',
        'processing_time': 0.5,
        'patient_id': 'TEST_PATIENT_001',
        'shap_explanation': {
            'feature_importance': [
                {'feature': 'accel_x', 'importance': 0.15, 'rank': 1},
                {'feature': 'gyro_z', 'importance': 0.12, 'rank': 2},
                {'feature': 'emg_1', 'importance': 0.10, 'rank': 3},
                {'feature': 'accel_y', 'importance': 0.08, 'rank': 4},
                {'feature': 'gyro_x', 'importance': 0.07, 'rank': 5}
            ],
            'shap_values': [0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03],
            'feature_names': ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 
                            'gyro_z', 'emg_1', 'emg_2', 'emg_3'],
            'explanation_text': 'Gait detected with high confidence based on accelerometer patterns.'
        },
        'lime_explanation': {
            'top_features': [
                {'feature': 'accel_x', 'weight': 0.20, 'rank': 1},
                {'feature': 'gyro_z', 'weight': 0.15, 'rank': 2},
                {'feature': 'emg_1', 'weight': 0.12, 'rank': 3}
            ],
            'feature_weights': {
                'accel_x': 0.20,
                'accel_y': 0.08,
                'accel_z': 0.05,
                'gyro_x': 0.07,
                'gyro_y': 0.06,
                'gyro_z': 0.15,
                'emg_1': 0.12,
                'emg_2': 0.09,
                'emg_3': 0.08
            },
            'explanation_text': 'LIME analysis confirms accelerometer as primary factor.'
        },
        'metrics': {
            'accuracy': 0.968,
            'precision': 0.955,
            'recall': 0.972,
            'f1_score': 0.963,
            'roc_auc': 0.985
        }
    }


# ============================================================================
# File Fixtures
# ============================================================================

@pytest.fixture
def temp_model_path(test_data_dir):
    """Temporary path for saving models during tests"""
    model_dir = test_data_dir / "models"
    model_dir.mkdir(exist_ok=True)
    return model_dir


@pytest.fixture
def sample_csv_file(test_data_dir, sample_gait_data):
    """Create temporary CSV file for testing"""
    csv_path = test_data_dir / "test_data.csv"
    sample_gait_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_excel_file(test_data_dir, sample_gait_data):
    """Create temporary Excel file for testing"""
    excel_path = test_data_dir / "test_data.xlsx"
    sample_gait_data.to_excel(excel_path, index=False)
    return excel_path


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def test_config():
    """Test configuration dictionary"""
    return {
        'model': {
            'input_dim': 9,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.3
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 10,
            'early_stopping_patience': 5
        },
        'preprocessing': {
            'window_size': 100,
            'step_size': 50,
            'normalization': 'zscore',
            'filter_cutoff': 20
        },
        'federated': {
            'min_clients': 3,
            'aggregation_strategy': 'fedavg',
            'differential_privacy': True,
            'epsilon': 1.0,
            'delta': 1e-5
        }
    }


# ============================================================================
# Helper Classes
# ============================================================================

class TestHelpers:
    """Helper methods for testing"""
    
    @staticmethod
    def assert_valid_dataframe(df):
        """Assert dataframe is valid gait data"""
        required_columns = [
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'emg_1', 'emg_2', 'emg_3'
        ]
        
        assert isinstance(df, pd.DataFrame), "Must be a pandas DataFrame"
        assert all(col in df.columns for col in required_columns), "Missing required columns"
        assert len(df) > 0, "DataFrame cannot be empty"
        assert not df[required_columns].isna().any().any(), "Contains NaN values"
    
    @staticmethod
    def assert_valid_tensor(tensor, expected_shape=None):
        """Assert tensor is valid"""
        assert isinstance(tensor, torch.Tensor), "Must be a PyTorch tensor"
        if expected_shape:
            assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
        assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
        assert not torch.isinf(tensor).any(), "Tensor contains Inf values"
    
    @staticmethod
    def assert_valid_prediction(prediction):
        """Assert prediction is valid"""
        assert 0 <= prediction <= 1, f"Prediction must be in [0, 1], got {prediction}"
        assert not np.isnan(prediction), "Prediction is NaN"
        assert not np.isinf(prediction), "Prediction is Inf"
    
    @staticmethod
    def assert_valid_model(model):
        """Assert model is valid"""
        assert isinstance(model, torch.nn.Module), "Must be a PyTorch model"
        assert sum(p.numel() for p in model.parameters()) > 0, "Model has no parameters"
    
    @staticmethod
    def compare_tensors(tensor1, tensor2, rtol=1e-5, atol=1e-8):
        """Compare two tensors with tolerance"""
        return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)


@pytest.fixture
def helpers():
    """Provide helper functions to tests"""
    return TestHelpers()


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "federated: marks tests for federated learning"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )
    config.addinivalue_line(
        "markers", "database: marks tests requiring database"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers based on test location
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "test_federated" in item.nodeid:
            item.add_marker(pytest.mark.federated)
        if "test_database" in item.nodeid:
            item.add_marker(pytest.mark.database)


# ============================================================================
# Cleanup Hooks
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Cleanup code here
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ============================================================================
# Mock Streamlit for Dashboard Tests
# ============================================================================

@pytest.fixture
def mock_streamlit(monkeypatch):
    """Mock Streamlit for testing dashboard components"""
    mock_st = MagicMock()
    
    # Mock session state
    mock_st.session_state = {}
    
    # Mock common streamlit functions
    mock_st.set_page_config = MagicMock()
    mock_st.title = MagicMock()
    mock_st.header = MagicMock()
    mock_st.subheader = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.write = MagicMock()
    mock_st.success = MagicMock()
    mock_st.error = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.info = MagicMock()
    mock_st.button = MagicMock(return_value=False)
    mock_st.selectbox = MagicMock(return_value="Option 1")
    mock_st.multiselect = MagicMock(return_value=[])
    mock_st.file_uploader = MagicMock(return_value=None)
    
    monkeypatch.setattr("streamlit", mock_st)
    return mock_st


# ============================================================================
# Performance Profiling
# ============================================================================

@pytest.fixture
def profile_test(request):
    """Profile test execution time"""
    import time
    start_time = time.time()
    
    yield
    
    end_time = time.time()
    duration = end_time - start_time
    
    if duration > 1.0:  # Log slow tests
        print(f"\n⚠️  Slow test: {request.node.name} took {duration:.2f}s")
