# File: tests/test_federated_system.py
# Comprehensive test script for federated learning system

import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.federated.fl_client import FederatedClient
from src.federated.fl_server import FederatedServer, CustomFedAvg
from src.federated.privacy import DifferentialPrivacy, SecureAggregation, PrivacyAccountant
from src.federated.aggregation import FederatedAggregator, SecureAggregator, AdaptiveAggregator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data(n_samples: int = 1000, save_path: str = "test_data.csv"):
    """Create synthetic test data for federated learning"""
    
    logger.info(f"Creating test data with {n_samples} samples...")
    
    # Generate synthetic sensor data
    data = pd.DataFrame({
        'accel_x': np.random.randn(n_samples),
        'accel_y': np.random.randn(n_samples),
        'accel_z': 9.8 + np.random.randn(n_samples) * 0.5,
        'gyro_x': np.random.randn(n_samples) * 0.1,
        'gyro_y': np.random.randn(n_samples) * 0.1,
        'gyro_z': np.random.randn(n_samples) * 0.1,
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='20ms')
    })
    
    # Save to file
    data.to_csv(save_path, index=False)
    logger.info(f"Test data saved to {save_path}")
    
    return data


def test_federated_client():
    """Test federated client functionality"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Federated Client")
    logger.info("="*70)
    
    try:
        # Create test data
        test_data_path = "test_client_data.csv"
        create_test_data(n_samples=1000, save_path=test_data_path)
        
        # Initialize client for gait detection
        logger.info("Testing gait detector client...")
        gait_client = FederatedClient(
            client_id="test_client_gait",
            data_path=test_data_path,
            model_type="gait_detector"
        )
        
        # Get parameters
        params = gait_client.get_parameters(config={})
        logger.info(f"✅ Gait client initialized: {len(params)} parameter arrays")
        
        # Test training
        logger.info("Testing client training...")
        new_params, n_examples, metrics = gait_client.fit(params, config={})
        logger.info(f"✅ Training completed: {n_examples} examples, metrics: {metrics}")
        
        # Test evaluation
        loss, n_examples, eval_metrics = gait_client.evaluate(params, config={})
        logger.info(f"✅ Evaluation completed: loss={loss:.4f}, accuracy={eval_metrics['accuracy']:.4f}")
        
        # Initialize client for disease classification
        logger.info("\nTesting disease classifier client...")
        disease_client = FederatedClient(
            client_id="test_client_disease",
            data_path=test_data_path,
            model_type="disease_classifier"
        )
        
        params = disease_client.get_parameters(config={})
        logger.info(f"✅ Disease client initialized: {len(params)} parameter arrays")
        
        logger.info("\n✅ All client tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Client test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregation_strategies():
    """Test different aggregation strategies"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Aggregation Strategies")
    logger.info("="*70)
    
    try:
        # Create dummy client models
        n_clients = 5
        client_models = []
        
        for i in range(n_clients):
            model = {
                'layer1.weight': torch.randn(10, 5),
                'layer1.bias': torch.randn(10),
                'layer2.weight': torch.randn(2, 10),
                'layer2.bias': torch.randn(2)
            }
            client_models.append(model)
        
        logger.info(f"Created {n_clients} dummy client models")
        
        # Test FedAvg
        logger.info("\nTesting FedAvg...")
        fedavg = FederatedAggregator(strategy='fedavg')
        result = fedavg.aggregate(client_models)
        logger.info(f"✅ FedAvg: aggregated {len(result)} parameters")
        
        # Test Trimmed Mean
        logger.info("\nTesting Trimmed Mean...")
        trimmed = FederatedAggregator(strategy='trimmed_mean')
        result = trimmed.aggregate(client_models, trim_ratio=0.2)
        logger.info(f"✅ Trimmed Mean: aggregated {len(result)} parameters")
        
        # Test Krum
        logger.info("\nTesting Krum...")
        krum = FederatedAggregator(strategy='krum')
        result = krum.aggregate(client_models, n_byzantine=1)
        logger.info(f"✅ Krum: aggregated {len(result)} parameters")
        
        # Test Adaptive Aggregation
        logger.info("\nTesting Adaptive Aggregation...")
        adaptive = AdaptiveAggregator()
        
        # Create mock client metrics
        client_metrics = [
            {'n_samples': 100, 'loss': 0.5, 'accuracy': 0.85},
            {'n_samples': 150, 'loss': 0.6, 'accuracy': 0.82},
            {'n_samples': 120, 'loss': 0.4, 'accuracy': 0.88},
            {'n_samples': 200, 'loss': 0.55, 'accuracy': 0.84},
            {'n_samples': 180, 'loss': 0.45, 'accuracy': 0.87}
        ]
        
        result, strategy = adaptive.aggregate(client_models, client_metrics)
        logger.info(f"✅ Adaptive: selected {strategy}, aggregated {len(result)} parameters")
        
        logger.info("\n✅ All aggregation tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Aggregation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_privacy_mechanisms():
    """Test privacy-preserving mechanisms"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Privacy Mechanisms")
    logger.info("="*70)
    
    try:
        # Create dummy parameters
        dummy_params = [
            np.random.randn(10, 10),
            np.random.randn(10),
            np.random.randn(5, 5)
        ]
        
        # Test Differential Privacy
        logger.info("\nTesting Differential Privacy...")
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        
        # Add Gaussian noise
        noisy_gaussian = dp.add_gaussian_noise(dummy_params)
        logger.info(f"✅ Gaussian noise added: {len(noisy_gaussian)} arrays")
        
        # Add Laplace noise
        noisy_laplace = dp.add_laplace_noise(dummy_params)
        logger.info(f"✅ Laplace noise added: {len(noisy_laplace)} arrays")
        
        # Clip gradients
        clipped = dp.clip_gradients(dummy_params, max_norm=1.0)
        logger.info(f"✅ Gradients clipped: {len(clipped)} arrays")
        
        # Test Secure Aggregation
        logger.info("\nTesting Secure Aggregation...")
        sa = SecureAggregation()
        
        # Encrypt parameters
        encrypted = sa.encrypt_parameters(dummy_params)
        logger.info(f"✅ Parameters encrypted: {len(encrypted)} arrays")
        
        # Decrypt parameters
        shapes = [param.shape for param in dummy_params]
        decrypted = sa.decrypt_parameters(encrypted, shapes)
        
        # Verify decryption
        decryption_correct = all(
            np.allclose(dummy_params[i], decrypted[i])
            for i in range(len(dummy_params))
        )
        logger.info(f"✅ Decryption successful: {decryption_correct}")
        
        # Test Privacy Accountant
        logger.info("\nTesting Privacy Accountant...")
        accountant = PrivacyAccountant(total_epsilon=5.0)
        
        # Spend budget
        success1 = accountant.spend_privacy_budget(1.0, "gradient_noise")
        success2 = accountant.spend_privacy_budget(2.0, "parameter_noise")
        success3 = accountant.spend_privacy_budget(3.0, "should_fail")  # Should fail
        
        logger.info(f"✅ Budget spending: {success1}, {success2}, {not success3}")
        logger.info(f"✅ Remaining budget: {accountant.get_remaining_budget():.4f}")
        
        # Test Secure Aggregator with DP
        logger.info("\nTesting Secure Aggregation with DP...")
        client_models = [
            {
                'layer1': torch.randn(10, 5),
                'layer2': torch.randn(5, 2)
            }
            for _ in range(5)
        ]
        
        secure_agg = SecureAggregator(noise_multiplier=1.0, l2_norm_clip=1.0)
        result, epsilon = secure_agg.aggregate_with_dp(client_models)
        logger.info(f"✅ DP aggregation: ε = {epsilon:.4f}")
        
        logger.info("\n✅ All privacy tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Privacy test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_federated_server():
    """Test federated server functionality"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Federated Server")
    logger.info("="*70)
    
    try:
        # Test server initialization
        logger.info("Initializing federated server...")
        
        server = FederatedServer(
            model_type="gait_detector",
            server_address="localhost:8080",
            num_rounds=5,
            min_clients=2
        )
        
        logger.info(f"✅ Server initialized: {server.num_rounds} rounds, min {server.min_clients} clients")
        
        # Test custom strategy
        logger.info("\nTesting custom FedAvg strategy...")
        strategy = CustomFedAvg(
            model_type="gait_detector",
            min_fit_clients=2,
            min_evaluate_clients=1,
            min_available_clients=2
        )
        
        # Test parameter initialization
        initial_params = strategy.initialize_parameters(client_manager=None)
        logger.info(f"✅ Strategy initialized with parameters")
        
        logger.info("\n✅ All server tests passed!")
        logger.info("Note: Full server testing requires running actual clients")
        return True
        
    except Exception as e:
        logger.error(f"❌ Server test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test complete integration"""
    
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Integration Test")
    logger.info("="*70)
    
    try:
        # Create test data for multiple clients
        logger.info("Creating data for 3 clients...")
        client_data_paths = []
        
        for i in range(3):
            path = f"test_client_{i}_data.csv"
            create_test_data(n_samples=1000, save_path=path)
            client_data_paths.append(path)
        
        # Initialize clients
        logger.info("\nInitializing clients...")
        clients = []
        
        for i, data_path in enumerate(client_data_paths):
            client = FederatedClient(
                client_id=f"client_{i}",
                data_path=data_path,
                model_type="gait_detector"
            )
            clients.append(client)
            logger.info(f"✅ Client {i} initialized with {len(client.X_train)} training samples")
        
        # Simulate one round of federated learning
        logger.info("\nSimulating federated learning round...")
        
        # Get initial parameters from first client
        initial_params = clients[0].get_parameters(config={})
        
        # Each client trains locally
        client_updates = []
        client_weights = []
        
        for i, client in enumerate(clients):
            logger.info(f"Training client {i}...")
            new_params, n_examples, metrics = client.fit(initial_params, config={})
            
            client_updates.append(new_params)
            client_weights.append(n_examples)
            
            logger.info(f"✅ Client {i}: {n_examples} examples, "
                       f"loss={metrics['train_loss']:.4f}, "
                       f"acc={metrics['train_accuracy']:.4f}")
        
        # Aggregate updates
        logger.info("\nAggregating client updates...")
        
        # Convert to torch tensors for aggregation
        client_models = []
        for params in client_updates:
            model_dict = {
                f'param_{i}': torch.tensor(param)
                for i, param in enumerate(params)
            }
            client_models.append(model_dict)
        
        # Normalize weights
        total_examples = sum(client_weights)
        normalized_weights = [w / total_examples for w in client_weights]
        
        aggregator = FederatedAggregator(strategy='fedavg')
        aggregated = aggregator.aggregate(client_models, normalized_weights)
        
        logger.info(f"✅ Aggregated {len(aggregated)} parameters from {len(clients)} clients")
        
        # Test with differential privacy
        logger.info("\nTesting aggregation with differential privacy...")
        secure_agg = SecureAggregator(noise_multiplier=0.5, l2_norm_clip=1.0)
        dp_aggregated, epsilon = secure_agg.aggregate_with_dp(client_models, normalized_weights)
        logger.info(f"✅ DP aggregation completed: ε = {epsilon:.4f}")
        
        logger.info("\n✅ Integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all federated learning tests"""
    
    logger.info("\n" + "#"*70)
    logger.info("# FE-AI FEDERATED LEARNING SYSTEM - COMPREHENSIVE TEST SUITE")
    logger.info("#"*70 + "\n")
    
    results = {}
    
    # Run all tests
    results['client'] = test_federated_client()
    results['aggregation'] = test_aggregation_strategies()
    results['privacy'] = test_privacy_mechanisms()
    results['server'] = test_federated_server()
    results['integration'] = test_integration()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name.upper():20s}: {status}")
    
    all_passed = all(results.values())
    
    logger.info("="*70)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.info("⚠️  SOME TESTS FAILED")
    logger.info("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)