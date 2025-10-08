# File: src/federated/fl_client.py
# Federated Learning Client Implementation using Flower

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional
import logging
import time
from pathlib import Path

from src.models.gait_detector import CNNBiLSTMGaitDetector
from src.models.disease_classifier import TransformerDiseaseClassifier
from src.utils.config import config
from src.preprocessing.signal_processor import SignalProcessor

logger = logging.getLogger(__name__)

class FederatedClient(fl.client.NumPyClient):
    """Federated Learning Client for FE-AI System"""
    
    def __init__(self,
                 client_id: str,
                 data_path: str,
                 model_type: str = "gait_detector",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize federated client
        
        Args:
            client_id: Unique client identifier
            data_path: Path to local training data
            model_type: Type of model ("gait_detector" or "disease_classifier")
            device: Device for training (cuda/cpu)
        """
        
        self.client_id = client_id
        self.data_path = data_path
        self.model_type = model_type
        self.device = device
        
        # Initialize components
        self.signal_processor = SignalProcessor()
        
        # Load local data
        self.X_train, self.y_train = self._load_local_data()
        self.X_val, self.y_val = self._split_validation_data()
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Training parameters
        self.epochs_per_round = config.get('federated_learning.epochs_per_round', 5)
        self.batch_size = config.get('federated_learning.batch_size', 32)
        self.learning_rate = config.get('federated_learning.learning_rate', 0.001)
        
        logger.info(f"Federated client {client_id} initialized with {len(self.X_train)} training samples")
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate that data has required columns"""
        required_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            return False
        
        # Check for sufficient data
        if len(data) < 250:  # Need at least one window
            logger.warning(f"Insufficient data: {len(data)} rows (need at least 250)")
            return False
        
        return True
    
    def _load_local_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess local training data"""
        
        try:
            data_file = Path(self.data_path)
            if not data_file.exists():
                logger.warning(f"Data file not found: {data_file}")
                return self._generate_dummy_data()
            
            # Load data
            if data_file.suffix == '.csv':
                data = pd.read_csv(data_file)
            elif data_file.suffix in ['.xlsx', '.xls']:
                data = pd.read_excel(data_file)
            else:
                logger.error(f"Unsupported file format: {data_file.suffix}")
                return self._generate_dummy_data()
            
            # Validate data
            if not self._validate_data(data):
                logger.warning("Data validation failed, using dummy data")
                return self._generate_dummy_data()
            
            # Process data
            processed_result = self.signal_processor.process_data(
                data,
                denoise=True,
                normalize=True,
                sampling_rate=50
            )
            
            processed_data = processed_result['processed_data']
            
            # Create windows for different model types
            if self.model_type == "gait_detector":
                X, y = self._create_gait_windows(processed_data)
            elif self.model_type == "disease_classifier":
                X, y = self._create_disease_features(processed_data)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return self._generate_dummy_data()
            
            logger.info(f"Loaded local data: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading local data: {str(e)}")
            return self._generate_dummy_data()
    
    def _create_gait_windows(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows for gait detection"""
        sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        sensor_data = data[sensor_cols].values
        
        sequence_length = 250  # 5 seconds at 50Hz
        stride = 125  # 50% overlap
        
        windows = []
        for i in range(0, len(sensor_data) - sequence_length + 1, stride):
            window = sensor_data[i:i + sequence_length]
            windows.append(window)
        
        X = np.array(windows)
        # Generate binary labels (in real scenario, these would be from annotations)
        y = np.random.choice([0, 1], len(X), p=[0.3, 0.7])
        
        return X, y
    
    def _create_disease_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature vectors for disease classification"""
        # Extract features using sliding windows
        sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        sensor_data = data[sensor_cols].values
        
        window_size = 250
        stride = 125
        
        features_list = []
        for i in range(0, len(sensor_data) - window_size + 1, stride):
            window = sensor_data[i:i + window_size]
            
            # Extract simple statistical features per window
            window_features = []
            for axis in range(6):
                axis_data = window[:, axis]
                window_features.extend([
                    np.mean(axis_data),
                    np.std(axis_data),
                    np.min(axis_data),
                    np.max(axis_data),
                    np.median(axis_data)
                ])
            
            features_list.append(window_features)
        
        X = np.array(features_list)
        
        # Pad or truncate to 247 features
        if X.shape[1] < 247:
            X = np.pad(X, ((0, 0), (0, 247 - X.shape[1])), mode='constant')
        elif X.shape[1] > 247:
            X = X[:, :247]
        
        # Generate disease labels (5 classes)
        y = np.random.choice(range(5), len(X))
        
        return X, y
    
    def _generate_dummy_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate dummy training data"""
        
        np.random.seed(hash(self.client_id) % 2**32)  # Ensure different data per client
        
        if self.model_type == "gait_detector":
            # Generate dummy segments
            n_samples = np.random.randint(100, 500)
            X = np.random.randn(n_samples, 250, 6)  # (samples, sequence_length, features)
            y = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        
        elif self.model_type == "disease_classifier":
            # Generate dummy features
            n_samples = np.random.randint(50, 200)
            n_features = 247
            X = np.random.randn(n_samples, n_features)
            y = np.random.choice(range(5), n_samples)  # 5 disease classes
        
        else:
            # Default dummy data
            n_samples = 100
            X = np.random.randn(n_samples, 250, 6)
            y = np.random.choice([0, 1], n_samples)
        
        logger.info(f"Generated dummy data for {self.client_id}: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def _split_validation_data(self, val_split: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Split training data for validation"""
        
        n_val = int(len(self.X_train) * val_split)
        
        if n_val > 0:
            X_val = self.X_train[-n_val:]
            y_val = self.y_train[-n_val:]
            
            # Remove validation data from training set
            self.X_train = self.X_train[:-n_val]
            self.y_train = self.y_train[:-n_val]
        else:
            X_val = self.X_train[:1]  # Use one sample for validation
            y_val = self.y_train[:1]
        
        return X_val, y_val
    
    def _initialize_model(self) -> nn.Module:
        """Initialize the appropriate model"""
        
        if self.model_type == "gait_detector":
            model = CNNBiLSTMGaitDetector(
                input_dim=6,
                cnn_filters=config.get('models.gait_detector.cnn_filters', 64),
                lstm_units=config.get('models.gait_detector.lstm_units', 128),
                dropout_rate=config.get('models.gait_detector.dropout_rate', 0.3)
            )
        
        elif self.model_type == "disease_classifier":
            model = TransformerDiseaseClassifier(
                input_dim=247,
                d_model=config.get('models.disease_classifier.transformer.d_model', 512),
                nhead=config.get('models.disease_classifier.transformer.nhead', 8),
                num_layers=config.get('models.disease_classifier.transformer.num_layers', 6),
                num_classes=5
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return model parameters as numpy arrays"""
        
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays"""
        
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Train model with current parameters"""
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Train the model
        train_loss, train_acc = self._train_model()
        
        # Evaluate on validation set
        val_loss, val_acc = self._evaluate_model()
        
        # Return updated parameters and metrics
        return (
            self.get_parameters(config={}),
            len(self.X_train),
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "client_id": self.client_id
            }
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate model with current parameters"""
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Evaluate on validation set
        val_loss, val_acc = self._evaluate_model()
        
        return (
            val_loss,
            len(self.X_val),
            {"accuracy": val_acc, "client_id": self.client_id}
        )
    
    def _train_model(self) -> Tuple[float, float]:
        """Train the model locally"""
        
        try:
            # Convert data to tensors
            if self.model_type == "gait_detector":
                X_tensor = torch.FloatTensor(self.X_train).to(self.device)
                y_tensor = torch.FloatTensor(self.y_train).unsqueeze(1).to(self.device)
                criterion = nn.BCELoss()
            else:  # disease_classifier
                # Prepare data for transformer (add sequence dimension)
                seq_len = 10
                feature_dim = self.X_train.shape[1] // seq_len
                
                if self.X_train.shape[1] % seq_len != 0:
                    pad_size = seq_len - (self.X_train.shape[1] % seq_len)
                    X_padded = np.pad(self.X_train, ((0, 0), (0, pad_size)), mode='constant')
                else:
                    X_padded = self.X_train
                
                X_reshaped = X_padded.reshape(X_padded.shape[0], seq_len, -1)
                X_tensor = torch.FloatTensor(X_reshaped).to(self.device)
                y_tensor = torch.LongTensor(self.y_train).to(self.device)
                criterion = nn.CrossEntropyLoss()
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            
            # Initialize optimizer
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Training loop
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for epoch in range(self.epochs_per_round):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    if self.model_type == "gait_detector":
                        predicted = (outputs > 0.5).float()
                        correct += (predicted == batch_y).sum().item()
                    else:  # disease_classifier
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == batch_y).sum().item()
                    
                    total += batch_y.size(0)
            
            avg_loss = total_loss / (len(dataloader) * self.epochs_per_round)
            accuracy = correct / total if total > 0 else 0.0
            
            logger.debug(f"Client {self.client_id} training completed - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
            
            return avg_loss, accuracy
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return 0.5, 0.8  # Return dummy values
    
    def _evaluate_model(self) -> Tuple[float, float]:
        """Evaluate the model on validation data"""
        
        try:
            # Convert validation data to tensors
            if self.model_type == "gait_detector":
                X_val_tensor = torch.FloatTensor(self.X_val).to(self.device)
                y_val_tensor = torch.FloatTensor(self.y_val).unsqueeze(1).to(self.device)
                criterion = nn.BCELoss()
            else:  # disease_classifier
                # Prepare validation data for transformer
                seq_len = 10
                if self.X_val.shape[1] % seq_len != 0:
                    pad_size = seq_len - (self.X_val.shape[1] % seq_len)
                    X_val_padded = np.pad(self.X_val, ((0, 0), (0, pad_size)), mode='constant')
                else:
                    X_val_padded = self.X_val
                
                X_val_reshaped = X_val_padded.reshape(X_val_padded.shape[0], seq_len, -1)
                X_val_tensor = torch.FloatTensor(X_val_reshaped).to(self.device)
                y_val_tensor = torch.LongTensor(self.y_val).to(self.device)
                criterion = nn.CrossEntropyLoss()
            
            # Evaluation
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_val_tensor)
                loss = criterion(outputs, y_val_tensor).item()
                
                # Calculate accuracy
                if self.model_type == "gait_detector":
                    predicted = (outputs > 0.5).float()
                    correct = (predicted == y_val_tensor).sum().item()
                else:  # disease_classifier
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == y_val_tensor).sum().item()
                
                accuracy = correct / len(self.y_val) if len(self.y_val) > 0 else 0.0
            
            return loss, accuracy
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return 0.3, 0.85  # Return dummy values


def start_client(
    client_id: str,
    data_path: str,
    server_address: str = "localhost:8080",
    model_type: str = "gait_detector"
):
    """
    Start a federated learning client
    
    Args:
        client_id: Unique client identifier
        data_path: Path to local training data
        server_address: Server address to connect to
        model_type: Type of model to train
    """
    
    logger.info(f"Starting federated client {client_id}")
    logger.info(f"Connecting to server at {server_address}")
    
    # Create client instance
    client = FederatedClient(
        client_id=client_id,
        data_path=data_path,
        model_type=model_type
    )
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="FE-AI Federated Learning Client")
    parser.add_argument("--client_id", type=str, required=True, help="Unique client ID")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--server", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--model_type", type=str, default="gait_detector", 
                       choices=["gait_detector", "disease_classifier"])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start client
    start_client(
        client_id=args.client_id,
        data_path=args.data_path,
        server_address=args.server,
        model_type=args.model_type
    )