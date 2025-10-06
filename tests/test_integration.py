"""
File: tests/test_integration.py
Integration tests for end-to-end workflows
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile

from src.data.data_loader import DataLoader
from src.preprocessing.signal_processor import SignalProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.models.gait_detector import GaitDetector
from src.models.disease_classifier import DiseaseClassifier
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.federated.fl_server import FederatedServer
from src.federated.fl_client import FederatedClient


@pytest.mark.integration
class TestDataPipeline:
    """Integration tests for data processing pipeline"""
    
    def test_complete_data_pipeline(self, sample_gait_data, temp_model_path):
        """Test complete data processing pipeline"""
        # 1. Load data
        assert sample_gait_data is not None
        assert len(sample_gait_data) > 0
        
        # 2. Preprocess
        processor = SignalProcessor()
        processed_data = processor.process(sample_gait_data)
        
        assert processed_data.shape == sample_gait_data.shape
        
        # 3. Extract features
        extractor = FeatureExtractor()
        features = extractor.extract_time_domain_features(processed_data)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # 4. Verify data is ready for model
        sensor_cols = ['accel_x', 'accel_y', 'accel_z', 
                      'gyro_x', 'gyro_y', 'gyro_z',
                      'emg_1', 'emg_2', 'emg_3']
        
        model_input = processed_data[sensor_cols].values
        
        assert model_input.shape[1] == 9
        assert not np.isnan(model_input).any()


@pytest.mark.integration
class TestModelPipeline:
    """Integration tests for model training and inference pipeline"""
    
    def test_training_and_inference_pipeline(self, sample_gait_tensor, temp_model_path):
        """Test complete training and inference workflow"""
        # 1. Initialize model
        model = GaitDetector(input_dim=9, hidden_dim=64)
        
        # 2. Create training data
        X_train = sample_gait_tensor
        y_train = torch.randint(0, 2, (len(X_train), 1)).float()
        
        # 3. Train for few epochs
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(2):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # 4. Save model
        save_path = temp_model_path / 'test_model.pth'
        torch.save(model.state_dict(), save_path)
        
        assert save_path.exists()
        
        # 5. Load model and infer
        loaded_model = GaitDetector(input_dim=9, hidden_dim=64)
        loaded_model.load_state_dict(torch.load(save_path))
        loaded_model.eval()
        
        with torch.no_grad():
            predictions = loaded_model(X_train[:4])
        
        assert predictions.shape[0] == 4
        assert torch.all((predictions >= 0) & (predictions <= 1))


@pytest.mark.integration
class TestExplainabilityPipeline:
    """Integration tests for explainability pipeline"""
    
    def test_shap_lime_explainability(self, sample_gait_tensor):
        """Test SHAP and LIME explainability integration"""
        # 1. Create and train simple model
        model = GaitDetector(input_dim=9, hidden_dim=32)
        model.eval()
        
        # 2. Prepare data
        background_data = sample_gait_tensor[:20].numpy()
        test_instance = sample_gait_tensor[0].numpy()
        
        # 3. SHAP explanation
        shap_explainer = SHAPExplainer(
            model,
            background_data,
            device='cpu'
        )
        
        shap_result = shap_explainer.explain_instance(test_instance)
        
        assert 'prediction' in shap_result
        assert 'shap_values' in shap_result
        assert 'feature_importance' in shap_result
        
        # 4. LIME explanation
        lime_explainer = LIMEExplainer(
            model,
            device='cpu'
        )
        
        lime_result = lime_explainer.explain_instance(test_instance)
        
        assert 'prediction' in lime_result
        assert 'top_features' in lime_result
        
        # 5. Compare predictions (should be similar)
        assert abs(shap_result['prediction'] - lime_result['prediction']) < 0.1


@pytest.mark.integration
@pytest.mark.slow
class TestFederatedLearningPipeline:
    """Integration tests for federated learning pipeline"""
    
    def test_complete_federated_learning_round(self):
        """Test complete federated learning workflow"""
        # 1. Initialize server
        server = FederatedServer(min_clients=3)
        server.initialize_model(model_type="gait_detection", input_dim=9)
        
        # 2. Create clients with local data
        num_clients = 3
        clients = []
        
        for i in range(num_clients):
            client = FederatedClient(f"client_{i}")
            model = GaitDetector(input_dim=9, hidden_dim=64)
            client.set_model(model)
            
            # Register with server
            server.register_client(client.client_id)
            clients.append(client)
        
        # 3. Distribute global model
        global_params = server.get_global_model_parameters()
        
        for client in clients:
            client.set_parameters(global_params)
        
        # 4. Local training on each client
        client_updates = []
        
        for i, client in enumerate(clients):
            # Create client-specific data
            X_local = torch.randn(50, 100, 9)
            y_local = torch.randint(0, 2, (50, 1)).float()
            
            # Train locally
            client.train(X_local, y_local, epochs=2, batch_size=16)
            
            # Get updated parameters
            client_updates.append(client.get_parameters())
        
        # 5. Aggregate on server
        initial_round = server.current_round
        server.aggregate_client_updates(client_updates)
        
        assert server.current_round == initial_round + 1
        
        # 6. Verify global model updated
        new_global_params = server.get_global_model_parameters()
        
        # Parameters should have changed
        assert not np.array_equal(global_params[0], new_global_params[0])


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end integration tests"""
    
    def test_complete_system_workflow(self, sample_gait_data, temp_model_path):
        """Test complete system from upload to prediction"""
        # 1. Data Upload & Validation
        from src.data.data_validator import DataValidator
        
        validator = DataValidator()
        validation = validator.validate_dataframe(sample_gait_data)
        
        assert validation['valid'] == True
        
        # 2. Preprocessing
        processor = SignalProcessor()
        processed_data = processor.process(sample_gait_data)
        
        # 3. Segmentation
        from src.preprocessing.segmentation import WindowSegmenter
        
        segmenter = WindowSegmenter(window_size=100, step_size=50)
        sensor_cols = ['accel_x', 'accel_y', 'accel_z',
                      'gyro_x', 'gyro_y', 'gyro_z',
                      'emg_1', 'emg_2', 'emg_3']
        
        sensor_data = processed_data[sensor_cols].values
        windows = segmenter.segment(sensor_data)
        
        assert len(windows) > 0
        
        # 4. Model Inference
        model = GaitDetector(input_dim=9, hidden_dim=64)
        model.eval()
        
        # Convert first window to tensor
        window_tensor = torch.FloatTensor(windows[0]).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(window_tensor)
        
        assert prediction.shape == (1, 1)
        assert 0 <= prediction.item() <= 1
        
        # 5. Explainability
        lime_explainer = LIMEExplainer(model, device='cpu')
        explanation = lime_explainer.explain_instance(windows[0])
        
        assert 'prediction' in explanation
        assert 'top_features' in explanation
        
        # 6. Store results
        results = {
            'prediction': float(prediction.item()),
            'confidence': float(prediction.item() * 100),
            'explanation': explanation,
            'num_windows': len(windows),
            'validation': validation
        }
        
        assert results['prediction'] >= 0
        assert results['confidence'] >= 0


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests with database"""
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_upload_to_database_workflow(self, mock_client, sample_gait_data):
        """Test uploading and retrieving data from database"""
        from src.database.mongo_handler import MongoHandler
        
        # Mock setup
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.insert_one.return_value.inserted_id = "test_upload_id"
        mock_collection.find_one.return_value = {
            '_id': 'test_upload_id',
            'patient_id': 'TEST001',
            'data': sample_gait_data.to_dict('records')
        }
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        # 1. Upload data
        handler = MongoHandler()
        
        upload_doc = {
            'patient_id': 'TEST001',
            'filename': 'test.csv',
            'data': sample_gait_data.to_dict('records'),
            'upload_timestamp': datetime.now()
        }
        
        upload_id = handler.insert_upload(upload_doc)
        
        assert upload_id is not None
        
        # 2. Retrieve data
        retrieved = handler.get_upload(upload_id)
        
        assert retrieved is not None
        assert retrieved['patient_id'] == 'TEST001'


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Integration tests for system performance"""
    
    def test_large_batch_processing(self):
        """Test processing large batches of data"""
        # Create large dataset
        large_data = pd.DataFrame({
            'accel_x': np.random.randn(10000),
            'accel_y': np.random.randn(10000),
            'accel_z': np.random.randn(10000),
            'gyro_x': np.random.randn(10000),
            'gyro_y': np.random.randn(10000),
            'gyro_z': np.random.randn(10000),
            'emg_1': np.random.randn(10000),
            'emg_2': np.random.randn(10000),
            'emg_3': np.random.randn(10000),
        })
        
        # Process
        processor = SignalProcessor()
        processed = processor.process(large_data)
        
        assert len(processed) == len(large_data)
        assert not processed.isna().any().any()
    
    def test_concurrent_predictions(self):
        """Test handling multiple concurrent predictions"""
        model = GaitDetector(input_dim=9, hidden_dim=64)
        model.eval()
        
        # Create multiple batches
        num_batches = 10
        batch_size = 32
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_batches):
                batch = torch.randn(batch_size, 100, 9)
                pred = model(batch)
                predictions.append(pred)
        
        assert len(predictions) == num_batches
        assert all(p.shape[0] == batch_size for p in predictions)


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security features"""
    
    def test_data_encryption_workflow(self, sample_gait_data):
        """Test data encryption and decryption workflow"""
        from cryptography.fernet import Fernet
        
        # Generate key
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        # Encrypt data
        data_json = sample_gait_data.to_json()
        encrypted = cipher.encrypt(data_json.encode())
        
        # Decrypt data
        decrypted = cipher.decrypt(encrypted).decode()
        recovered_data = pd.read_json(decrypted)
        
        # Verify data integrity
        pd.testing.assert_frame_equal(sample_gait_data, recovered_data)
    
    def test_differential_privacy_workflow(self):
        """Test differential privacy in federated learning"""
        from src.federated.privacy import DifferentialPrivacy
        
        dp = DifferentialPrivacy(
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            delta=1e-5
        )
        
        # Simulate gradients
        gradients = [np.random.randn(10, 10) for _ in range(5)]
        
        # Apply privacy
        private_gradients = [dp.apply_privacy(g) for g in gradients]
        
        # Verify noise was added
        for orig, priv in zip(gradients, private_gradients):
            assert not np.array_equal(orig, priv)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
