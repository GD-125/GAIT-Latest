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