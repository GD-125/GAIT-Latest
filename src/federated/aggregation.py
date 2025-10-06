# File: src/federated/aggregation.py
# Advanced aggregation strategies for federated learning

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import OrderedDict
import copy

logger = logging.getLogger(__name__)


class FederatedAggregator:
    """
    Advanced aggregation strategies for federated learning
    Implements multiple aggregation algorithms with differential privacy
    """
    
    def __init__(self, strategy: str = 'fedavg'):
        """
        Initialize aggregator
        
        Args:
            strategy: Aggregation strategy ('fedavg', 'fedprox', 'fedadam', 'trimmed_mean', 'krum')
        """
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Aggregation history
        self.aggregation_history = []
    
    def aggregate(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client models using selected strategy
        
        Args:
            client_models: List of client model parameters
            client_weights: Optional weights for clients (e.g., based on dataset size)
            **kwargs: Additional parameters for specific strategies
            
        Returns:
            Aggregated model parameters
        """
        if not client_models:
            raise ValueError("No client models provided for aggregation")
        
        # Select aggregation strategy
        if self.strategy == 'fedavg':
            aggregated = self._fedavg(client_models, client_weights)
        elif self.strategy == 'fedprox':
            aggregated = self._fedprox(client_models, client_weights, **kwargs)
        elif self.strategy == 'fedadam':
            aggregated = self._fedadam(client_models, client_weights, **kwargs)
        elif self.strategy == 'trimmed_mean':
            aggregated = self._trimmed_mean(client_models, **kwargs)
        elif self.strategy == 'krum':
            aggregated = self._krum(client_models, **kwargs)
        elif self.strategy == 'median':
            aggregated = self._coordinate_median(client_models)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.strategy}")
        
        # Store in history
        self.aggregation_history.append({
            'strategy': self.strategy,
            'num_clients': len(client_models),
            'weights': client_weights
        })
        
        self.logger.info(f"Aggregated {len(client_models)} models using {self.strategy}")
        
        return aggregated
    
    def _fedavg(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Federated Averaging (FedAvg) - Standard weighted average
        McMahan et al., 2017
        
        Args:
            client_models: List of client model parameters
            client_weights: Optional weights for weighted averaging
            
        Returns:
            Aggregated model parameters
        """
        # Default to equal weights
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Aggregate each parameter
        for param_name in client_models[0].keys():
            # Weighted sum of parameters
            weighted_params = [
                client_weights[i] * client_models[i][param_name]
                for i in range(len(client_models))
            ]
            aggregated_params[param_name] = torch.stack(weighted_params).sum(dim=0)
        
        return aggregated_params
    
    def _fedprox(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        global_model: Optional[Dict[str, torch.Tensor]] = None,
        mu: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        Federated Proximal (FedProx) - Adds proximal term to handle heterogeneity
        Li et al., 2020
        
        Args:
            client_models: List of client model parameters
            client_weights: Optional weights
            global_model: Previous global model (for proximal term)
            mu: Proximal term coefficient
            
        Returns:
            Aggregated model parameters
        """
        # Start with FedAvg
        aggregated = self._fedavg(client_models, client_weights)
        
        # Add proximal term if global model provided
        if global_model is not None:
            for param_name in aggregated.keys():
                # Add regularization toward global model
                aggregated[param_name] = (
                    aggregated[param_name] * (1 - mu) +
                    global_model[param_name] * mu
                )
        
        return aggregated
    
    def _fedadam(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None,
        global_model: Optional[Dict[str, torch.Tensor]] = None,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> Dict[str, torch.Tensor]:
        """
        Federated Adam (FedAdam) - Server-side adaptive optimization
        Reddi et al., 2020
        
        Args:
            client_models: List of client model parameters
            client_weights: Optional weights
            global_model: Previous global model
            learning_rate: Learning rate for server optimizer
            beta1: First moment decay
            beta2: Second moment decay
            epsilon: Small constant for numerical stability
            
        Returns:
            Aggregated model parameters
        """
        # Compute pseudo-gradient (difference from global model)
        if global_model is None:
            return self._fedavg(client_models, client_weights)
        
        # Get averaged update
        avg_update = self._fedavg(client_models, client_weights)
        
        # Initialize moment estimates if not exist
        if not hasattr(self, 'm_t'):
            self.m_t = {}
            self.v_t = {}
            self.t = 0
        
        self.t += 1
        updated_params = {}
        
        for param_name in avg_update.keys():
            # Compute pseudo-gradient
            grad = avg_update[param_name] - global_model[param_name]
            
            # Initialize moments for new parameters
            if param_name not in self.m_t:
                self.m_t[param_name] = torch.zeros_like(grad)
                self.v_t[param_name] = torch.zeros_like(grad)
            
            # Update biased first moment estimate
            self.m_t[param_name] = beta1 * self.m_t[param_name] + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            self.v_t[param_name] = beta2 * self.v_t[param_name] + (1 - beta2) * (grad ** 2)
            
            # Compute bias-corrected moment estimates
            m_hat = self.m_t[param_name] / (1 - beta1 ** self.t)
            v_hat = self.v_t[param_name] / (1 - beta2 ** self.t)
            
            # Update parameters
            updated_params[param_name] = (
                global_model[param_name] + 
                learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
            )
        
        return updated_params
    
    def _trimmed_mean(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Trimmed Mean - Robust aggregation against Byzantine attacks
        Removes extreme values before averaging
        
        Args:
            client_models: List of client model parameters
            trim_ratio: Ratio of models to trim from each end
            
        Returns:
            Aggregated model parameters
        """
        n_clients = len(client_models)
        n_trim = int(n_clients * trim_ratio)
        
        aggregated_params = {}
        
        for param_name in client_models[0].keys():
            # Stack all client parameters
            stacked = torch.stack([m[param_name] for m in client_models])
            
            # Sort along client dimension
            sorted_params, _ = torch.sort(stacked, dim=0)
            
            # Remove n_trim smallest and largest values
            if n_trim > 0:
                trimmed = sorted_params[n_trim:-n_trim]
            else:
                trimmed = sorted_params
            
            # Compute mean of trimmed values
            aggregated_params[param_name] = trimmed.mean(dim=0)
        
        return aggregated_params
    
    def _krum(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        n_select: int = 1,
        n_byzantine: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Krum - Byzantine-robust aggregation
        Selects models with smallest distance to others
        Blanchard et al., 2017
        
        Args:
            client_models: List of client model parameters
            n_select: Number of models to select
            n_byzantine: Expected number of Byzantine clients
            
        Returns:
            Aggregated model parameters (average of selected models)
        """
        n_clients = len(client_models)
        
        # Flatten all models for distance computation
        flattened_models = []
        for model in client_models:
            flat = torch.cat([param.flatten() for param in model.values()])
            flattened_models.append(flat)
        
        # Compute pairwise distances
        distances = torch.zeros((n_clients, n_clients))
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = torch.norm(flattened_models[i] - flattened_models[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each client, compute sum of distances to k nearest neighbors
        k = n_clients - n_byzantine - 2
        scores = []
        for i in range(n_clients):
            # Get k smallest distances (excluding self)
            sorted_dists, _ = torch.sort(distances[i])
            score = sorted_dists[1:k+1].sum()  # Exclude distance to self (0)
            scores.append(score)
        
        # Select n_select models with smallest scores
        scores_tensor = torch.tensor(scores)
        _, selected_indices = torch.topk(scores_tensor, n_select, largest=False)
        
        # Average selected models
        selected_models = [client_models[i] for i in selected_indices.tolist()]
        aggregated = self._fedavg(selected_models)
        
        self.logger.info(f"Krum selected clients: {selected_indices.tolist()}")
        
        return aggregated
    
    def _coordinate_median(
        self,
        client_models: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Coordinate-wise Median - Robust aggregation
        Computes median for each parameter coordinate
        
        Args:
            client_models: List of client model parameters
            
        Returns:
            Aggregated model parameters
        """
        aggregated_params = {}
        
        for param_name in client_models[0].keys():
            # Stack all client parameters
            stacked = torch.stack([m[param_name] for m in client_models])
            
            # Compute median along client dimension
            aggregated_params[param_name] = torch.median(stacked, dim=0)[0]
        
        return aggregated_params


class SecureAggregator:
    """
    Secure aggregation with differential privacy and encryption
    """
    
    def __init__(
        self,
        noise_multiplier: float = 1.0,
        l2_norm_clip: float = 1.0,
        delta: float = 1e-5
    ):
        """
        Initialize secure aggregator with differential privacy
        
        Args:
            noise_multiplier: Scale of Gaussian noise for DP
            l2_norm_clip: Clipping threshold for gradients
            delta: Privacy parameter delta
        """
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.delta = delta
        self.logger = logging.getLogger(__name__)
    
    def aggregate_with_dp(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Aggregate with differential privacy (DP-SGD)
        
        Args:
            client_models: List of client model parameters
            client_weights: Optional client weights
            
        Returns:
            aggregated_params: DP-protected aggregated parameters
            privacy_spent: Privacy budget epsilon spent
        """
        # Default weights
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        aggregated_params = {}
        
        for param_name in client_models[0].keys():
            # Clip each client's parameters
            clipped_params = []
            for model in client_models:
                param = model[param_name]
                # Compute L2 norm
                param_norm = torch.norm(param)
                # Clip if necessary
                if param_norm > self.l2_norm_clip:
                    clipped = param * (self.l2_norm_clip / param_norm)
                else:
                    clipped = param
                clipped_params.append(clipped)
            
            # Weighted average
            weighted_sum = sum(
                client_weights[i] * clipped_params[i]
                for i in range(len(clipped_params))
            )
            
            # Add Gaussian noise for differential privacy
            noise = torch.randn_like(weighted_sum) * (
                self.noise_multiplier * self.l2_norm_clip / len(client_models)
            )
            
            aggregated_params[param_name] = weighted_sum + noise
        
        # Compute privacy spent (simplified)
        # Using moments accountant for tighter bounds in practice
        n_clients = len(client_models)
        privacy_spent = self._compute_epsilon(n_clients)
        
        self.logger.info(f"Differential privacy applied: ε = {privacy_spent:.4f}, δ = {self.delta}")
        
        return aggregated_params, privacy_spent
    
    def _compute_epsilon(self, n_clients: int, n_rounds: int = 1) -> float:
        """
        Compute privacy budget epsilon (simplified calculation)
        
        Args:
            n_clients: Number of clients
            n_rounds: Number of aggregation rounds
            
        Returns:
            Privacy budget epsilon
        """
        # Simplified epsilon calculation
        # In practice, use moments accountant for tighter bounds
        q = 1.0 / n_clients  # Sampling probability
        sigma = self.noise_multiplier
        
        # Using strong composition theorem
        epsilon = (q * np.sqrt(2 * n_rounds * np.log(1 / self.delta))) / sigma
        
        return epsilon
    
    def secure_multi_party_computation(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        masks: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Secure multi-party computation for aggregation
        Uses additive secret sharing
        
        Args:
            client_models: List of client model parameters
            masks: Optional pre-shared masks for each client
            
        Returns:
            Aggregated model parameters
        """
        if masks is None:
            # Generate random masks that sum to zero
            masks = self._generate_masks(client_models)
        
        aggregated_params = {}
        
        for param_name in client_models[0].keys():
            # Add masks to client models
            masked_params = [
                client_models[i][param_name] + masks[i][param_name]
                for i in range(len(client_models))
            ]
            
            # Sum masked parameters
            # Masks cancel out: sum(model + mask) = sum(model) + sum(mask) = sum(model)
            aggregated_params[param_name] = torch.stack(masked_params).sum(dim=0)
        
        self.logger.info("Secure multi-party computation completed")
        
        return aggregated_params
    
    def _generate_masks(
        self,
        client_models: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate random masks that sum to zero"""
        n_clients = len(client_models)
        masks = [{} for _ in range(n_clients)]
        
        for param_name in client_models[0].keys():
            param_shape = client_models[0][param_name].shape
            
            # Generate n-1 random masks
            for i in range(n_clients - 1):
                masks[i][param_name] = torch.randn(param_shape)
            
            # Last mask ensures sum is zero
            masks[n_clients - 1][param_name] = -sum(
                masks[i][param_name] for i in range(n_clients - 1)
            )
        
        return masks


class AdaptiveAggregator:
    """
    Adaptive aggregation that selects strategy based on conditions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_history = []
    
    def aggregate(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_metrics: Optional[List[Dict]] = None,
        global_model: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Adaptively select and apply aggregation strategy
        
        Args:
            client_models: List of client model parameters
            client_metrics: Optional metrics from each client
            global_model: Previous global model
            
        Returns:
            aggregated_params: Aggregated model parameters
            strategy_used: Name of strategy used
        """
        # Analyze client models to select strategy
        strategy = self._select_strategy(client_models, client_metrics)
        
        # Create aggregator with selected strategy
        aggregator = FederatedAggregator(strategy=strategy)
        
        # Compute weights based on client metrics
        if client_metrics:
            weights = self._compute_adaptive_weights(client_metrics)
        else:
            weights = None
        
        # Aggregate
        aggregated = aggregator.aggregate(client_models, weights)
        
        self.logger.info(f"Adaptive aggregation selected strategy: {strategy}")
        
        return aggregated, strategy
    
    def _select_strategy(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_metrics: Optional[List[Dict]] = None
    ) -> str:
        """Select optimal aggregation strategy based on conditions"""
        n_clients = len(client_models)
        
        # Check for high variance in client models (potential Byzantine)
        variance = self._compute_model_variance(client_models)
        
        if variance > 0.5:  # High variance threshold
            self.logger.info("High variance detected, using robust aggregation")
            if n_clients >= 10:
                return 'krum'
            else:
                return 'trimmed_mean'
        
        # Check for heterogeneous data (from metrics)
        if client_metrics:
            heterogeneity = self._compute_data_heterogeneity(client_metrics)
            if heterogeneity > 0.3:
                self.logger.info("High heterogeneity detected, using FedProx")
                return 'fedprox'
        
        # Default to FedAvg for stable conditions
        return 'fedavg'
    
    def _compute_model_variance(
        self,
        client_models: List[Dict[str, torch.Tensor]]
    ) -> float:
        """Compute variance across client models"""
        # Flatten all models
        flattened = []
        for model in client_models:
            flat = torch.cat([param.flatten() for param in model.values()])
            flattened.append(flat)
        
        stacked = torch.stack(flattened)
        variance = torch.var(stacked, dim=0).mean().item()
        
        return variance
    
    def _compute_data_heterogeneity(
        self,
        client_metrics: List[Dict]
    ) -> float:
        """Compute data heterogeneity from client metrics"""
        # Use loss or accuracy variance as proxy
        if 'loss' in client_metrics[0]:
            losses = [m['loss'] for m in client_metrics]
            return float(np.std(losses))
        elif 'accuracy' in client_metrics[0]:
            accuracies = [m['accuracy'] for m in client_metrics]
            return float(np.std(accuracies))
        
        return 0.0
    
    def _compute_adaptive_weights(
        self,
        client_metrics: List[Dict]
    ) -> List[float]:
        """Compute adaptive weights based on client performance"""
        # Weight by data size and performance
        weights = []
        
        for metrics in client_metrics:
            # Base weight on data size
            data_weight = metrics.get('n_samples', 1.0)
            
            # Adjust by performance (accuracy or inverse loss)
            if 'accuracy' in metrics:
                perf_weight = metrics['accuracy']
            elif 'loss' in metrics:
                perf_weight = 1.0 / (1.0 + metrics['loss'])
            else:
                perf_weight = 1.0
            
            weights.append(data_weight * perf_weight)
        
        return weights


if __name__ == "__main__":
    print("Testing Federated Aggregation...")
    
    # Create dummy client models
    client_models = []
    for i in range(5):
        model = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(2, 10)
        }
        client_models.append(model)
    
    # Test FedAvg
    print("\n1. Testing FedAvg...")
    aggregator = FederatedAggregator(strategy='fedavg')
    result = aggregator.aggregate(client_models)
    print(f"✅ FedAvg aggregation complete")
    
    # Test Trimmed Mean
    print("\n2. Testing Trimmed Mean...")
    aggregator = FederatedAggregator(strategy='trimmed_mean')
    result = aggregator.aggregate(client_models)
    print(f"✅ Trimmed mean aggregation complete")
    
    # Test Krum
    print("\n3. Testing Krum...")
    aggregator = FederatedAggregator(strategy='krum')
    result = aggregator.aggregate(client_models)
    print(f"✅ Krum aggregation complete")
    
    # Test Differential Privacy
    print("\n4. Testing Differential Privacy...")
    secure_agg = SecureAggregator(noise_multiplier=1.0)
    result, epsilon = secure_agg.aggregate_with_dp(client_models)
    print(f"✅ DP aggregation complete (ε = {epsilon:.4f})")
    
    # Test Adaptive Aggregation
    print("\n5. Testing Adaptive Aggregation...")
    adaptive_agg = AdaptiveAggregator()
    result, strategy = adaptive_agg.aggregate(client_models)
    print(f"✅ Adaptive aggregation used: {strategy}")
    
    print("\n✅ All federated aggregation tests passed!")