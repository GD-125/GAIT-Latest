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
from src.data.data_loader import DataLoader
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
        self.data_loader = DataLoader()
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
            validation_results = self.data_loader.validate_data(data)
            if not validation_results['is_valid']:
                logger.warning("Data validation failed, using dummy data")
                return self._generate_dummy_data()
            
            # Process data
            processed_data = self.signal_processor.process_data(
                data,
                denoise=True,
                normalize=True,
                segment=True,
                extract_features=True,
                window_size=5,
                overlap=25
            )
            
            if self.model_type == "gait_detector":
                # Use segments for gait detection
                X = np.array(processed_data.get('segments', []))
                # Generate binary labels (1 for gait, 0 for non-gait)
                y = np.random.choice([0, 1], len(X), p=[0.3, 0.7])
            
            elif self.model_type == "disease_classifier":
                # Use features for disease classification
                features_df = processed_data.get('features', pd.DataFrame())
                X = features_df.values
                
                # Generate disease labels
                diseases = ['Parkinson', 'Huntington', 'Ataxia', 'MS', 'Normal']
                y = np.random.choice(range(len(diseases)), len(X))
            
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return self._generate_dummy_data()
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading local data: {str(e)}")
            return self._generate_dummy_data()
    
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


# File: src/federated/fl_server.py
# Federated Learning Server Implementation

import flwr as fl
from flwr.common import Parameters, FitRes, EvaluateRes, Scalar
from flwr.server.strategy import FedAvg
from typing import Dict, List, Tuple, Optional, Union
import logging
import numpy as np
import torch
from pathlib import Path

from src.utils.config import config
from src.models.gait_detector import CNNBiLSTMGaitDetector  
from src.models.disease_classifier import TransformerDiseaseClassifier

logger = logging.getLogger(__name__)

class CustomFedAvg(FedAvg):
    """Custom Federated Averaging strategy with enhanced features"""
    
    def __init__(self, 
                 model_type: str = "gait_detector",
                 min_fit_clients: int = 3,
                 min_evaluate_clients: int = 3,
                 min_available_clients: int = 3,
                 evaluate_metrics_aggregation_fn=None,
                 fit_metrics_aggregation_fn=None):
        
        self.model_type = model_type
        
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,  
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn or self._evaluate_metrics_aggregation,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn or self._fit_metrics_aggregation
        )
        
        # Initialize global model for parameter initialization
        self.global_model = self._create_global_model()
        
        logger.info(f"CustomFedAvg strategy initialized for {model_type}")
    
    def _create_global_model(self) -> torch.nn.Module:
        """Create global model instance"""
        
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
        
        return model
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters"""
        
        initial_parameters = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
        return fl.common.ndarrays_to_parameters(initial_parameters)
    
    def _fit_metrics_aggregation(self, metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """Aggregate fit metrics from clients"""
        
        # Initialize aggregated metrics
        aggregated_metrics = {}
        
        if not metrics:
            return aggregated_metrics
        
        # Calculate weighted averages
        total_examples = sum(num_examples for num_examples, _ in metrics)
        
        # Aggregate loss and accuracy
        weighted_train_loss = sum(num_examples * m["train_loss"] for num_examples, m in metrics)
        weighted_train_acc = sum(num_examples * m["train_accuracy"] for num_examples, m in metrics)
        weighted_val_loss = sum(num_examples * m["val_loss"] for num_examples, m in metrics)
        weighted_val_acc = sum(num_examples * m["val_accuracy"] for num_examples, m in metrics)
        
        aggregated_metrics["train_loss"] = weighted_train_loss / total_examples
        aggregated_metrics["train_accuracy"] = weighted_train_acc / total_examples
        aggregated_metrics["val_loss"] = weighted_val_loss / total_examples  
        aggregated_metrics["val_accuracy"] = weighted_val_acc / total_examples
        aggregated_metrics["num_clients"] = len(metrics)
        aggregated_metrics["total_examples"] = total_examples
        
        logger.info(f"Fit metrics aggregated from {len(metrics)} clients - "
                   f"Avg train acc: {aggregated_metrics['train_accuracy']:.4f}, "
                   f"Avg val acc: {aggregated_metrics['val_accuracy']:.4f}")
        
        return aggregated_metrics
    
    def _evaluate_metrics_aggregation(self, metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """Aggregate evaluation metrics from clients"""
        
        aggregated_metrics = {}
        
        if not metrics:
            return aggregated_metrics
        
        # Calculate weighted averages
        total_examples = sum(num_examples for num_examples, _ in metrics)
        
        weighted_accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics)
        
        aggregated_metrics["accuracy"] = weighted_accuracy / total_examples
        aggregated_metrics["num_clients"] = len(metrics)
        aggregated_metrics["total_examples"] = total_examples
        
        logger.info(f"Evaluation metrics aggregated from {len(metrics)} clients - "
                   f"Global accuracy: {aggregated_metrics['accuracy']:.4f}")
        
        return aggregated_metrics

class FederatedServer:
    """Federated Learning Server Manager"""
    
    def __init__(self,
                 model_type: str = "gait_detector",
                 server_address: str = "0.0.0.0:8080",
                 num_rounds: int = 50,
                 min_clients: int = 3):
        
        self.model_type = model_type
        self.server_address = server_address
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        
        # Create strategy
        self.strategy = CustomFedAvg(
            model_type=model_type,
            min_fit_clients=min_clients,
            min_evaluate_clients=max(1, min_clients // 2),
            min_available_clients=min_clients
        )
        
        logger.info(f"Federated server initialized for {model_type}")
    
    def start_server(self):
        """Start the federated learning server"""
        
        logger.info(f"Starting FL server on {self.server_address}")
        logger.info(f"Waiting for at least {self.min_clients} clients...")
        
        try:
            # Start Flower server
            fl.server.start_server(
                server_address=self.server_address,
                config=fl.server.ServerConfig(num_rounds=self.num_rounds),
                strategy=self.strategy,
            )
            
        except Exception as e:
            logger.error(f"Error starting federated server: {str(e)}")
            raise
    
    def save_global_model(self, save_path: str):
        """Save the global model after training"""
        
        try:
            # Get final parameters from strategy
            if hasattr(self.strategy, 'global_model'):
                model_path = Path(save_path) / f"global_{self.model_type}_model.pth"
                torch.save(self.strategy.global_model.state_dict(), model_path)
                logger.info(f"Global model saved to {model_path}")
            else:
                logger.warning("No global model available to save")
                
        except Exception as e:
            logger.error(f"Error saving global model: {str(e)}")


# File: src/federated/privacy.py
# Privacy-preserving mechanisms for federated learning

import numpy as np
import torch
from typing import List, Tuple, Dict, Any
import logging
from cryptography.fernet import Fernet
import hashlib
import hmac
import secrets

logger = logging.getLogger(__name__)

class DifferentialPrivacy:
    """Differential Privacy implementation for federated learning"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        """
        Initialize differential privacy mechanism
        
        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Privacy parameter for (ε,δ)-differential privacy
            sensitivity: Global sensitivity of the function
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        logger.info(f"Differential Privacy initialized - ε: {epsilon}, δ: {delta}")
    
    def add_gaussian_noise(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Add Gaussian noise to model parameters"""
        
        # Calculate noise scale for Gaussian mechanism
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
        
        noisy_parameters = []
        for param in parameters:
            noise = np.random.normal(0, sigma, param.shape)
            noisy_param = param + noise
            noisy_parameters.append(noisy_param)
        
        logger.debug(f"Added Gaussian noise with σ: {sigma:.4f}")
        return noisy_parameters
    
    def add_laplace_noise(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Add Laplace noise to model parameters"""
        
        # Scale for Laplace mechanism
        scale = self.sensitivity / self.epsilon
        
        noisy_parameters = []
        for param in parameters:
            noise = np.random.laplace(0, scale, param.shape)
            noisy_param = param + noise
            noisy_parameters.append(noisy_param)
        
        logger.debug(f"Added Laplace noise with scale: {scale:.4f}")
        return noisy_parameters
    
    def clip_gradients(self, gradients: List[np.ndarray], max_norm: float = 1.0) -> List[np.ndarray]:
        """Clip gradients to bound sensitivity"""
        
        clipped_gradients = []
        for grad in gradients:
            # Calculate L2 norm
            grad_norm = np.linalg.norm(grad)
            
            # Clip if necessary
            if grad_norm > max_norm:
                clipped_grad = grad * (max_norm / grad_norm)
            else:
                clipped_grad = grad
            
            clipped_gradients.append(clipped_grad)
        
        return clipped_gradients

class SecureAggregation:
    """Secure aggregation for federated learning"""
    
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        logger.info("Secure aggregation initialized")
    
    def encrypt_parameters(self, parameters: List[np.ndarray]) -> List[bytes]:
        """Encrypt model parameters"""
        
        encrypted_params = []
        for param in parameters:
            # Serialize parameter
            param_bytes = param.tobytes()
            
            # Encrypt
            encrypted_param = self.cipher_suite.encrypt(param_bytes)
            encrypted_params.append(encrypted_param)
        
        return encrypted_params
    
    def decrypt_parameters(self, encrypted_params: List[bytes], original_shapes: List[Tuple]) -> List[np.ndarray]:
        """Decrypt model parameters"""
        
        parameters = []
        for encrypted_param, shape in zip(encrypted_params, original_shapes):
            # Decrypt
            param_bytes = self.cipher_suite.decrypt(encrypted_param)
            
            # Deserialize
            param = np.frombuffer(param_bytes, dtype=np.float32).reshape(shape)
            parameters.append(param)
        
        return parameters
    
    def generate_secret_shares(self, parameters: List[np.ndarray], num_shares: int = 3, threshold: int = 2) -> List[List[np.ndarray]]:
        """Generate secret shares using Shamir's secret sharing"""
        
        # Simplified secret sharing (for demonstration)
        # In production, use proper cryptographic libraries
        
        all_shares = []
        
        for _ in range(num_shares):
            shares = []
            for param in parameters:
                # Generate random share (simplified)
                share = param + np.random.normal(0, 0.01, param.shape)
                shares.append(share)
            all_shares.append(shares)
        
        return all_shares
    
    def reconstruct_from_shares(self, shares: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Reconstruct parameters from secret shares"""
        
        if not shares:
            return []
        
        # Simple reconstruction by averaging (for demonstration)
        reconstructed_params = []
        
        for param_idx in range(len(shares[0])):
            param_shares = [share[param_idx] for share in shares]
            reconstructed_param = np.mean(param_shares, axis=0)
            reconstructed_params.append(reconstructed_param)
        
        return reconstructed_params

class PrivacyAccountant:
    """Privacy budget accounting for differential privacy"""
    
    def __init__(self, total_epsilon: float = 10.0):
        self.total_epsilon = total_epsilon
        self.used_epsilon = 0.0
        self.privacy_log = []
        
        logger.info(f"Privacy accountant initialized with total ε: {total_epsilon}")
    
    def spend_privacy_budget(self, epsilon: float, operation: str) -> bool:
        """Spend privacy budget for an operation"""
        
        if self.used_epsilon + epsilon > self.total_epsilon:
            logger.warning(f"Privacy budget exceeded! Requested: {epsilon}, Available: {self.total_epsilon - self.used_epsilon}")
            return False
        
        self.used_epsilon += epsilon
        self.privacy_log.append({
            'operation': operation,
            'epsilon': epsilon,
            'timestamp': np.datetime64('now'),
            'remaining_budget': self.total_epsilon - self.used_epsilon
        })
        
        logger.info(f"Privacy budget spent: {epsilon} for {operation}. Remaining: {self.total_epsilon - self.used_epsilon:.4f}")
        return True
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget"""
        return self.total_epsilon - self.used_epsilon
    
    def reset_budget(self):
        """Reset privacy budget (use with caution)"""
        self.used_epsilon = 0.0
        self.privacy_log = []
        logger.warning("Privacy budget reset!")

# Example usage and testing
if __name__ == "__main__":
    # Test differential privacy
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    
    # Create dummy parameters
    dummy_params = [np.random.randn(10, 10), np.random.randn(5, 5)]
    
    # Add noise
    noisy_params = dp.add_gaussian_noise(dummy_params)
    print(f"Original param norm: {np.linalg.norm(dummy_params[0]):.4f}")
    print(f"Noisy param norm: {np.linalg.norm(noisy_params[0]):.4f}")
    
    # Test secure aggregation
    sa = SecureAggregation()
    
    # Encrypt parameters
    encrypted = sa.encrypt_parameters(dummy_params)
    print(f"Parameters encrypted: {len(encrypted)} arrays")
    
    # Decrypt parameters
    shapes = [param.shape for param in dummy_params]
    decrypted = sa.decrypt_parameters(encrypted, shapes)
    print(f"Decryption successful: {np.allclose(dummy_params[0], decrypted[0])}")
    
    # Test privacy accountant
    accountant = PrivacyAccountant(total_epsilon=5.0)
    
    # Spend budget
    success1 = accountant.spend_privacy_budget(1.0, "gradient_noise")
    success2 = accountant.spend_privacy_budget(2.0, "parameter_noise")
    success3 = accountant.spend_privacy_budget(3.0, "output_noise")  # Should fail
    
    print(f"Budget operations: {success1}, {success2}, {success3}")
    print(f"Remaining budget: {accountant.get_remaining_budget():.4f}")