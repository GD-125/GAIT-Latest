# File: src/explainability/lime_explainer.py
# LIME-based explainability for local interpretable predictions

import torch
import numpy as np
from lime import lime_tabular
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) explainer
    Provides local explanations for individual predictions
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize LIME explainer
        
        Args:
            model: PyTorch model to explain
            device: Device for computations
            feature_names: Names of input features
            class_names: Names of output classes
        """
        self.model = model.to(device)
        self.device = device
        self.feature_names = feature_names
        self.class_names = class_names or ['Non-Gait', 'Gait']
        self.explainer = None
        self.logger = logging.getLogger(__name__)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def create_explainer(
        self,
        training_data: np.ndarray,
        mode: str = 'classification'
    ):
        """
        Create LIME explainer with training data
        
        Args:
            training_data: Training dataset for LIME (flattened)
            mode: 'classification' or 'regression'
        """
        # Flatten training data if needed
        if len(training_data.shape) > 2:
            training_data_flat = training_data.reshape(training_data.shape[0], -1)
        else:
            training_data_flat = training_data
        
        # Generate feature names if not provided
        if self.feature_names is None:
            n_features = training_data_flat.shape[1]
            self.feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data_flat,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode=mode,
            discretize_continuous=True
        )
        
        self.logger.info(f"LIME explainer created with training data shape: {training_data_flat.shape}")
    
    def _predict_fn(self, instances: np.ndarray) -> np.ndarray:
        """
        Prediction function wrapper for LIME
        
        Args:
            instances: Flattened input instances
            
        Returns:
            Prediction probabilities
        """
        # Get original shape from first instance
        batch_size = instances.shape[0]
        
        # Reshape back to original model input shape
        # Assuming input is (batch, channels, sequence_length)
        if len(instances.shape) == 2:
            # Need to infer original shape
            # For gait detection: (batch, 9, 250) -> 9 * 250 = 2250 features
            n_features = instances.shape[1]
            if n_features == 2250:  # 9 channels * 250 timesteps
                instances_reshaped = instances.reshape(batch_size, 9, 250)
            else:
                # Generic reshaping
                instances_reshaped = instances.reshape(batch_size, 1, -1)
        else:
            instances_reshaped = instances
        
        # Convert to tensor
        instances_tensor = torch.FloatTensor(instances_reshaped).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            output, _ = self.model(instances_tensor)
            probs = torch.softmax(output, dim=1)
        
        return probs.cpu().numpy()
    
    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict:
        """
        Explain a single prediction instance
        
        Args:
            instance: Single input instance to explain
            num_features: Number of top features to show
            num_samples: Number of perturbed samples for LIME
            
        Returns:
            Dictionary with explanation details
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        # Flatten instance
        if len(instance.shape) > 1:
            instance_flat = instance.reshape(-1)
        else:
            instance_flat = instance
        
        # Get LIME explanation
        exp = self.explainer.explain_instance(
            instance_flat,
            self._predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Get prediction
        instance_reshaped = instance.reshape(1, *instance.shape) if len(instance.shape) == 2 else instance.reshape(1, 9, 250)
        instance_tensor = torch.FloatTensor(instance_reshaped).to(self.device)
        
        with torch.no_grad():
            output, _ = self.model(instance_tensor)
            probs = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item()
        
        # Extract explanation details
        explanation_list = exp.as_list(label=predicted_class)
        
        # Parse feature contributions
        contributions = []
        for feature_desc, weight in explanation_list:
            contributions.append({
                'feature': feature_desc,
                'weight': float(weight),
                'contribution': 'positive' if weight > 0 else 'negative'
            })
        
        explanation = {
            'prediction': self.class_names[predicted_class],
            'confidence': float(confidence),
            'predicted_class': int(predicted_class),
            'top_contributions': contributions,
            'local_pred': float(exp.local_pred[0] if hasattr(exp, 'local_pred') else 0),
            'intercept': float(exp.intercept[predicted_class] if hasattr(exp, 'intercept') else 0),
            'score': float(exp.score),
            'summary': self._generate_explanation_text(
                self.class_names[predicted_class],
                confidence,
                contributions[:5]
            )
        }
        
        return explanation
    
    def explain_gait_classification(
        self,
        instance: np.ndarray,
        sensor_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Explain gait classification with sensor-specific details
        
        Args:
            instance: Input instance (channels, sequence_length)
            sensor_names: Names of sensors
            
        Returns:
            Detailed explanation dictionary
        """
        explanation = self.explain_instance(instance, num_features=15)
        
        # Add sensor-specific analysis
        if sensor_names and len(instance.shape) == 2:
            n_channels, seq_len = instance.shape
            
            # Calculate contribution per sensor
            sensor_contributions = {name: 0.0 for name in sensor_names}
            
            for contrib in explanation['top_contributions']:
                # Extract feature index from description
                try:
                    feature_idx = int(contrib['feature'].split('_')[1].split()[0])
                    sensor_idx = feature_idx % n_channels
                    sensor_contributions[sensor_names[sensor_idx]] += abs(contrib['weight'])
                except:
                    pass
            
            explanation['sensor_contributions'] = sensor_contributions
            explanation['dominant_sensor'] = max(
                sensor_contributions.items(),
                key=lambda x: x[1]
            )[0]
        
        return explanation
    
    def _generate_explanation_text(
        self,
        prediction: str,
        confidence: float,
        top_contributions: List[Dict]
    ) -> str:
        """Generate human-readable explanation"""
        text = f"LIME Explanation: Predicted as '{prediction}' with {confidence*100:.1f}% confidence.\n\n"
        text += "Key factors influencing this prediction:\n"
        
        for i, contrib in enumerate(top_contributions, 1):
            direction = "increases" if contrib['contribution'] == 'positive' else "decreases"
            text += f"{i}. {contrib['feature']} {direction} prediction (weight: {contrib['weight']:.4f})\n"
        
        text += "\nLIME creates a local linear model around this instance to explain the prediction."
        
        return text
    
    def compare_with_shap(
        self,
        instance: np.ndarray,
        shap_explanation: Dict
    ) -> Dict:
        """
        Compare LIME explanation with SHAP explanation
        
        Args:
            instance: Input instance
            shap_explanation: SHAP explanation dictionary
            
        Returns:
            Comparison analysis
        """
        lime_explanation = self.explain_instance(instance)
        
        comparison = {
            'lime_prediction': lime_explanation['prediction'],
            'shap_prediction': shap_explanation['prediction'],
            'predictions_match': lime_explanation['prediction'] == shap_explanation['prediction'],
            'lime_confidence': lime_explanation['confidence'],
            'shap_confidence': shap_explanation['confidence'],
            'confidence_difference': abs(
                lime_explanation['confidence'] - shap_explanation['confidence']
            ),
            'lime_top_features': [c['feature'] for c in lime_explanation['top_contributions'][:5]],
            'shap_top_features': [str(c) for c in shap_explanation['top_contributions'][:5]],
            'agreement_score': self._calculate_agreement(
                lime_explanation['top_contributions'],
                shap_explanation['top_contributions']
            )
        }
        
        return comparison
    
    def _calculate_agreement(
        self,
        lime_contribs: List[Dict],
        shap_contribs: List[Dict]
    ) -> float:
        """Calculate agreement score between LIME and SHAP"""
        # Simple overlap metric
        lime_features = set([c['feature'][:20] for c in lime_contribs[:10]])
        shap_features = set([str(c)[:20] for c in shap_contribs[:10]])
        
        if not lime_features or not shap_features:
            return 0.0
        
        overlap = len(lime_features.intersection(shap_features))
        return overlap / max(len(lime_features), len(shap_features))
    
    def save_explanation(
        self,
        explanation: Dict,
        save_path: str
    ):
        """Save explanation to JSON"""
        import json
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(explanation, f, indent=4)
        
        self.logger.info(f"LIME explanation saved to {save_path}")


if __name__ == "__main__":
    print("Testing LIME Explainer...")
    
    # Create dummy model
    from src.models.federated_model import FederatedGaitModel
    model = FederatedGaitModel()
    
    # Create explainer
    explainer = LIMEExplainer(model)
    
    # Create training data
    training_data = np.random.randn(100, 9, 250)
    explainer.create_explainer(training_data)
    
    # Explain instance
    test_instance = np.random.randn(9, 250)
    explanation = explainer.explain_instance(test_instance)
    
    print(f"✅ Prediction: {explanation['prediction']}")
    print(f"✅ Confidence: {explanation['confidence']:.2f}")
    print(f"✅ Top contributions: {len(explanation['top_contributions'])}")
    
    print("\n✅ LIME Explainer tests passed!")