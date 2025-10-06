"""
File: src/explainability/shap_explainer.py
SHAP-based explainability for gait detection and disease classification
Provides feature attribution and interpretability
"""

import numpy as np
import pandas as pd
import torch
import shap
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) explainer
    for deep learning models
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        device: str = 'cpu'
    ):
        """
        Initialize SHAP explainer
        
        Args:
            model: PyTorch model to explain
            background_data: Background dataset for SHAP
            feature_names: Names of input features
            device: Device for computation
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Convert background data to tensor
        self.background_data = torch.FloatTensor(background_data).to(device)
        
        # Feature names
        if feature_names is None:
            self.feature_names = [
                'accel_x', 'accel_y', 'accel_z',
                'gyro_x', 'gyro_y', 'gyro_z',
                'emg_1', 'emg_2', 'emg_3'
            ]
        else:
            self.feature_names = feature_names
        
        # Initialize SHAP explainer
        self._init_explainer()
    
    def _init_explainer(self):
        """Initialize SHAP Deep Explainer"""
        
        def model_wrapper(x):
            """Wrapper for SHAP to work with PyTorch models"""
            with torch.no_grad():
                x_tensor = torch.FloatTensor(x).to(self.device)
                output = self.model(x_tensor)
                
                # Handle tuple outputs (model with attention)
                if isinstance(output, tuple):
                    output = output[0]
                
                # Convert to numpy
                return output.cpu().numpy()
        
        try:
            # Use DeepExplainer for neural networks
            self.explainer = shap.DeepExplainer(
                self.model,
                self.background_data
            )
            logger.info("Initialized SHAP DeepExplainer")
        except Exception as e:
            logger.warning(f"DeepExplainer failed, using KernelExplainer: {e}")
            # Fallback to KernelExplainer
            self.explainer = shap.KernelExplainer(
                model_wrapper,
                self.background_data.cpu().numpy()
            )
    
    def explain_instance(
        self,
        instance: np.ndarray,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        """
        Explain a single prediction instance
        
        Args:
            instance: Single data instance [seq_len, features]
            return_dict: Return as dictionary
            
        Returns:
            SHAP values and explanation details
        """
        # Convert to tensor
        instance_tensor = torch.FloatTensor(instance).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            prediction = self.model(instance_tensor)
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            pred_value = prediction.cpu().numpy()[0]
        
        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(instance_tensor)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Aggregate SHAP values across time steps
            if len(shap_values.shape) > 2:
                shap_values_agg = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                shap_values_agg = np.mean(np.abs(shap_values), axis=0)
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            shap_values_agg = np.zeros(len(self.feature_names))
        
        if return_dict:
            return {
                'prediction': float(pred_value),
                'shap_values': shap_values_agg.tolist(),
                'feature_names': self.feature_names,
                'feature_importance': self._rank_features(shap_values_agg),
                'explanation_text': self._generate_explanation(
                    pred_value,
                    shap_values_agg
                )
            }
        
        return shap_values_agg
    
    def explain_batch(
        self,
        batch: np.ndarray,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Explain multiple instances
        
        Args:
            batch: Batch of data [batch_size, seq_len, features]
            max_samples: Maximum samples to explain
            
        Returns:
            Aggregated SHAP values and statistics
        """
        batch = batch[:max_samples]
        batch_tensor = torch.FloatTensor(batch).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(batch_tensor)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            pred_values = predictions.cpu().numpy()
        
        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(batch_tensor)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Aggregate across time and batch
            if len(shap_values.shape) > 2:
                shap_values_agg = np.mean(np.abs(shap_values), axis=(0, 1))
            else:
                shap_values_agg = np.mean(np.abs(shap_values), axis=0)
                
        except Exception as e:
            logger.error(f"Error in batch SHAP calculation: {e}")
            shap_values_agg = np.zeros(len(self.feature_names))
        
        return {
            'predictions': pred_values.tolist(),
            'mean_shap_values': shap_values_agg.tolist(),
            'feature_names': self.feature_names,
            'global_importance': self._rank_features(shap_values_agg),
            'num_samples': len(batch)
        }
    
    def _rank_features(self, shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """Rank features by importance"""
        importance_pairs = [
            {
                'feature': name,
                'importance': float(val),
                'rank': idx + 1
            }
            for idx, (name, val) in enumerate(
                sorted(
                    zip(self.feature_names, shap_values),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
            )
        ]
        return importance_pairs
    
    def _generate_explanation(
        self,
        prediction: float,
        shap_values: np.ndarray
    ) -> str:
        """Generate human-readable explanation"""
        # Get top 3 features
        top_features = self._rank_features(shap_values)[:3]
        
        # Determine prediction class
        if prediction > 0.5:
            pred_class = "GAIT DETECTED"
            confidence = prediction * 100
        else:
            pred_class = "NO GAIT DETECTED"
            confidence = (1 - prediction) * 100
        
        # Generate explanation
        explanation = f"Prediction: {pred_class} (Confidence: {confidence:.1f}%)\n\n"
        explanation += "Top Contributing Features:\n"
        
        for i, feat in enumerate(top_features, 1):
            explanation += f"{i}. {feat['feature']}: {feat['importance']:.4f}\n"
        
        explanation += "\nInterpretation:\n"
        explanation += f"The model's decision was primarily influenced by "
        explanation += f"{top_features[0]['feature']}, which had the strongest "
        explanation += f"impact on the prediction."
        
        return explanation
    
    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance bar chart
        
        Args:
            shap_values: SHAP values to plot
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort features by importance
        importance = np.abs(shap_values)
        sorted_idx = np.argsort(importance)
        
        # Plot
        ax.barh(
            range(len(sorted_idx)),
            importance[sorted_idx],
            color='steelblue'
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([self.feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('Feature Importance')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        return fig
    
    def generate_summary_plot(
        self,
        test_data: np.ndarray,
        save_path: Optional[str] = None,
        max_display: int = 10
    ):
        """
        Generate SHAP summary plot
        
        Args:
            test_data: Test dataset
            save_path: Optional path to save figure
            max_display: Maximum features to display
        """
        test_tensor = torch.FloatTensor(test_data).to(self.device)
        
        try:
            shap_values = self.explainer.shap_values(test_tensor)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Reshape for summary plot
            if len(shap_values.shape) > 2:
                # Average over time dimension
                shap_values = np.mean(shap_values, axis=1)
                test_data_2d = np.mean(test_data, axis=1)
            else:
                test_data_2d = test_data
            
            # Create summary plot
            shap.summary_plot(
                shap_values,
                test_data_2d,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved SHAP summary plot to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating summary plot: {e}")


class TimeSeriesSHAP:
    """
    Specialized SHAP explainer for time series data
    Provides temporal feature attribution
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        background_data: np.ndarray,
        feature_names: List[str],
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.background_data = background_data
        self.feature_names = feature_names
    
    def explain_temporal(
        self,
        instance: np.ndarray
    ) -> Dict[str, Any]:
        """
        Explain temporal contributions
        
        Args:
            instance: Time series instance [seq_len, features]
            
        Returns:
            Temporal SHAP values and visualization data
        """
        seq_len, num_features = instance.shape
        
        # Calculate SHAP values for each time step
        temporal_shap = np.zeros((seq_len, num_features))
        
        instance_tensor = torch.FloatTensor(instance).unsqueeze(0).to(self.device)
        
        # Use gradient-based approximation for temporal attribution
        instance_tensor.requires_grad = True
        
        with torch.enable_grad():
            output = self.model(instance_tensor)
            if isinstance(output, tuple):
                output = output[0]
            
            # Get gradients
            self.model.zero_grad()
            output.backward()
            
            gradients = instance_tensor.grad.cpu().numpy()[0]
            temporal_shap = gradients * instance
        
        return {
            'temporal_shap_values': temporal_shap.tolist(),
            'time_steps': list(range(seq_len)),
            'feature_names': self.feature_names,
            'temporal_importance': np.sum(np.abs(temporal_shap), axis=1).tolist()
        }
    
    def plot_temporal_attribution(
        self,
        temporal_shap: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot temporal feature attribution heatmap"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(
            temporal_shap.T,
            aspect='auto',
            cmap='RdBu_r',
            interpolation='nearest'
        )
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Features')
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_yticklabels(self.feature_names)
        ax.set_title('Temporal Feature Attribution')
        
        plt.colorbar(im, ax=ax, label='SHAP Value')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig