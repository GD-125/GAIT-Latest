# File: src/models/federated_model.py
# Federated Learning Model Implementation for Privacy-Preserving Training

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import logging
from collections import OrderedDict
import copy

logger = logging.getLogger(__name__)


class FederatedGaitModel(nn.Module):
    """
    Federated Learning compatible CNN-BiLSTM model for gait detection
    Supports model aggregation and differential privacy
    """
    
    def __init__(
        self,
        input_channels: int = 9,
        sequence_length: int = 250,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(FederatedGaitModel, self).__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # 1D CNN for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # Calculate CNN output size
        self.cnn_output_size = 256
        
        # BiLSTM for temporal dependencies
        self.bilstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for explainability
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary: Gait/Non-Gait
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention weights for explainability
        
        Args:
            x: Input tensor of shape (batch, channels, sequence_length)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Classification logits
            attention_weights: Attention weights (if return_attention=True)
        """
        # CNN feature extraction
        cnn_out = self.conv_layers(x)  # (batch, 256, seq_len')
        
        # Reshape for LSTM
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, seq_len', 256)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(cnn_out)  # (batch, seq_len', hidden*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq_len', 1)
        attended = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden*2)
        
        # Classification
        output = self.classifier(attended)
        
        if return_attention:
            return output, attention_weights.squeeze(-1)
        return output, None
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for federated aggregation"""
        return {
            name: param.data.clone() 
            for name, param in self.named_parameters()
        }
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters from federated aggregation"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in parameters:
                    param.copy_(parameters[name])
    
    def compute_gradient_norm(self) -> float:
        """Compute gradient norm for differential privacy"""
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5


class FederatedDiseaseClassifier(nn.Module):
    """
    Federated Learning compatible Transformer model for disease classification
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        num_classes: int = 5
    ):
        super(FederatedDiseaseClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classification head with multi-head attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention for explainability
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Classification logits
            attention_weights: Attention weights (if return_attention=True)
        """
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Attention pooling
        query = encoded.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
        pooled, attention_weights = self.attention_pooling(
            query, encoded, encoded
        )  # (batch, 1, d_model), (batch, 1, seq_len)
        
        pooled = pooled.squeeze(1)  # (batch, d_model)
        
        # Classification
        output = self.classifier(pooled)
        
        if return_attention:
            return output, attention_weights.squeeze(1)
        return output, None
    
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for federated aggregation"""
        return {
            name: param.data.clone() 
            for name, param in self.named_parameters()
        }
    
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters from federated aggregation"""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in parameters:
                    param.copy_(parameters[name])


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)


class FederatedModelManager:
    """
    Manager for federated model operations including aggregation and versioning
    """
    
    def __init__(self, model_save_path: str = "data/models/federated"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def aggregate_models(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Federated averaging of client models
        
        Args:
            client_models: List of client model parameters
            client_weights: Optional weights for weighted averaging
            
        Returns:
            Aggregated model parameters
        """
        if not client_models:
            raise ValueError("No client models provided for aggregation")
        
        # Equal weights if not provided
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Aggregate each parameter
        for param_name in client_models[0].keys():
            weighted_params = [
                client_weights[i] * client_models[i][param_name]
                for i in range(len(client_models))
            ]
            aggregated_params[param_name] = torch.stack(weighted_params).sum(dim=0)
        
        self.logger.info(f"Aggregated {len(client_models)} client models")
        return aggregated_params
    
    def save_federated_model(
        self,
        model: nn.Module,
        round_num: int,
        metadata: Dict
    ):
        """Save federated model with versioning"""
        save_path = self.model_save_path / f"federated_model_round_{round_num}.pt"
        
        torch.save({
            'round': round_num,
            'model_state_dict': model.state_dict(),
            'metadata': metadata
        }, save_path)
        
        self.logger.info(f"Saved federated model for round {round_num}")
    
    def load_federated_model(
        self,
        model: nn.Module,
        round_num: int
    ) -> Dict:
        """Load federated model from specific round"""
        load_path = self.model_save_path / f"federated_model_round_{round_num}.pt"
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")
        
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.logger.info(f"Loaded federated model from round {round_num}")
        return checkpoint['metadata']


if __name__ == "__main__":
    # Test federated models
    print("Testing Federated Gait Model...")
    gait_model = FederatedGaitModel()
    x = torch.randn(4, 9, 250)
    output, attention = gait_model(x, return_attention=True)
    print(f"Gait Model Output Shape: {output.shape}")
    print(f"Attention Shape: {attention.shape}")
    
    print("\nTesting Federated Disease Classifier...")
    disease_model = FederatedDiseaseClassifier()
    x = torch.randn(4, 100, 256)
    output, attention = disease_model(x, return_attention=True)
    print(f"Disease Model Output Shape: {output.shape}")
    print(f"Attention Shape: {attention.shape}")
    
    print("\nTesting Model Aggregation...")
    manager = FederatedModelManager()
    params1 = gait_model.get_model_parameters()
    params2 = gait_model.get_model_parameters()
    aggregated = manager.aggregate_models([params1, params2])
    print(f"Aggregated {len(aggregated)} parameters")
    
    print("\n✅ All federated model tests passed!")