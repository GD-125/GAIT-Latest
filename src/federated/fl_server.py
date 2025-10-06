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
