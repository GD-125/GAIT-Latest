# File: src/preprocessing/normalizer.py
# Data normalization and scaling utilities

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from typing import Dict, Any, Optional, Tuple, Union
import logging
import joblib

logger = logging.getLogger(__name__)

class DataNormalizer:
    """Comprehensive data normalization and scaling"""
    
    def __init__(self):
        self.scalers = {}
        self.fitted_scalers = {}
        self.normalization_stats = {}
        
        # Available normalization methods
        self.methods = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
            'quantile_uniform': lambda: QuantileTransformer(output_distribution='uniform'),
            'quantile_normal': lambda: QuantileTransformer(output_distribution='normal'),
            'power': PowerTransformer,
            'zscore': 'custom_zscore',
            'unit_vector': 'custom_unit_vector',
            'decimal_scaling': 'custom_decimal_scaling'
        }
    
    def fit_transform(self, 
                     data: pd.DataFrame, 
                     method: str = 'standard',
                     columns: Optional[list] = None,
                     group_by: Optional[str] = None) -> pd.DataFrame:
        """
        Fit normalizer and transform data
        
        Args:
            data: Input DataFrame
            method: Normalization method
            columns: Specific columns to normalize (None for all numeric)
            group_by: Column to group by for separate normalization
            
        Returns:
            Normalized DataFrame
        """
        
        if method not in self.methods:
            logger.error(f"Unknown normalization method: {method}")
            return data
        
        normalized_data = data.copy()
        
        try:
            # Select columns to normalize
            if columns is None:
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in columns if col in data.columns]
            
            if not numeric_columns:
                logger.warning("No numeric columns found for normalization")
                return data
            
            logger.info(f"Normalizing {len(numeric_columns)} columns using {method} method")
            
            if group_by and group_by in data.columns:
                # Group-wise normalization
                groups = data[group_by].unique()
                
                for group in groups:
                    group_mask = data[group_by] == group
                    group_data = data.loc[group_mask, numeric_columns]
                    
                    if len(group_data) > 1:
                        normalized_group = self._apply_normalization(
                            group_data, method, f"{method}_{group}"
                        )
                        normalized_data.loc[group_mask, numeric_columns] = normalized_group
                    else:
                        logger.warning(f"Group {group} has only one sample, skipping normalization")
            else:
                # Global normalization
                normalized_columns = self._apply_normalization(
                    data[numeric_columns], method, method
                )
                normalized_data[numeric_columns] = normalized_columns
            
            # Store normalization statistics
            self._calculate_normalization_stats(data[numeric_columns], normalized_data[numeric_columns])
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            return data
    
    def transform(self, 
                 data: pd.DataFrame, 
                 method: str = 'standard',
                 columns: Optional[list] = None) -> pd.DataFrame:
        """
        Transform data using previously fitted normalizer
        
        Args:
            data: Input DataFrame to transform
            method: Normalization method (must be previously fitted)
            columns: Specific columns to normalize
            
        Returns:
            Normalized DataFrame
        """
        
        if method not in self.fitted_scalers:
            logger.error(f"Normalizer {method} not fitted. Call fit_transform first.")
            return data
        
        normalized_data = data.copy()
        
        try:
            # Select columns to normalize
            if columns is None:
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in columns if col in data.columns]
            
            if not numeric_columns:
                logger.warning("No numeric columns found for normalization")
                return data
            
            # Apply fitted transformation
            scaler = self.fitted_scalers[method]
            
            if isinstance(scaler, dict):
                # Custom method or group-wise scalers
                for key, group_scaler in scaler.items():
                    if hasattr(group_scaler, 'transform'):
                        transformed = group_scaler.transform(data[numeric_columns])
                        normalized_data[numeric_columns] = transformed
                    else:
                        # Custom method
                        normalized_data[numeric_columns] = self._apply_custom_method(
                            data[numeric_columns], method, group_scaler
                        )
            else:
                # sklearn scaler
                transformed = scaler.transform(data[numeric_columns])
                normalized_data[numeric_columns] = transformed
            
            logger.info(f"Applied {method} normalization to {len(numeric_columns)} columns")
            
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            return data
    
    def _apply_normalization(self, 
                           data: pd.DataFrame, 
                           method: str, 
                           scaler_key: str) -> pd.DataFrame:
        """Apply specific normalization method"""
        
        if method in ['zscore', 'unit_vector', 'decimal_scaling']:
            # Custom methods
            normalized_data = self._apply_custom_method(data, method)
            # Store parameters for custom methods
            self.fitted_scalers[scaler_key] = self._get_custom_params(data, method)
            
        else:
            # sklearn methods
            scaler_class = self.methods[method]
            
            if callable(scaler_class):
                scaler = scaler_class()
            else:
                scaler = scaler_class
            
            try:
                normalized_data = pd.DataFrame(
                    scaler.fit_transform(data),
                    columns=data.columns,
                    index=data.index
                )
                
                # Store fitted scaler
                self.fitted_scalers[scaler_key] = scaler
                
            except Exception as e:
                logger.error(f"Error applying {method} normalization: {str(e)}")
                return data
        
        return normalized_data
    
    def _apply_custom_method(self, 
                           data: pd.DataFrame, 
                           method: str, 
                           params: Optional[Dict] = None) -> pd.DataFrame:
        """Apply custom normalization methods"""
        
        normalized_data = data.copy()
        
        try:
            if method == 'zscore':
                # Z-score normalization: (x - mean) / std
                for col in data.columns:
                    if params and col in params:
                        mean_val, std_val = params[col]['mean'], params[col]['std']
                    else:
                        mean_val, std_val = data[col].mean(), data[col].std()
                    
                    if std_val > 1e-12:  # Avoid division by zero
                        normalized_data[col] = (data[col] - mean_val) / std_val
                    else:
                        normalized_data[col] = data[col] - mean_val
            
            elif method == 'unit_vector':
                # Unit vector normalization: x / ||x||
                for col in data.columns:
                    if params and col in params:
                        norm_val = params[col]['norm']
                    else:
                        norm_val = np.linalg.norm(data[col])
                    
                    if norm_val > 1e-12:
                        normalized_data[col] = data[col] / norm_val
                    else:
                        normalized_data[col] = data[col]
            
            elif method == 'decimal_scaling':
                # Decimal scaling: x / 10^j where j is the smallest integer such that max(|x|) < 1
                for col in data.columns:
                    if params and col in params:
                        scale_factor = params[col]['scale_factor']
                    else:
                        max_abs = np.max(np.abs(data[col]))
                        if max_abs > 0:
                            j = int(np.ceil(np.log10(max_abs)))
                            scale_factor = 10 ** j
                        else:
                            scale_factor = 1
                    
                    normalized_data[col] = data[col] / scale_factor
            
        except Exception as e:
            logger.error(f"Error in custom method {method}: {str(e)}")
            return data
        
        return normalized_data
    
    def _get_custom_params(self, data: pd.DataFrame, method: str) -> Dict[str, Dict[str, float]]:
        """Get parameters for custom normalization methods"""
        
        params = {}
        
        try:
            for col in data.columns:
                if method == 'zscore':
                    params[col] = {
                        'mean': float(data[col].mean()),
                        'std': float(data[col].std())
                    }
                
                elif method == 'unit_vector':
                    params[col] = {
                        'norm': float(np.linalg.norm(data[col]))
                    }
                
                elif method == 'decimal_scaling':
                    max_abs = np.max(np.abs(data[col]))
                    if max_abs > 0:
                        j = int(np.ceil(np.log10(max_abs)))
                        scale_factor = 10 ** j
                    else:
                        scale_factor = 1
                    
                    params[col] = {
                        'scale_factor': float(scale_factor)
                    }
        
        except Exception as e:
            logger.error(f"Error getting custom parameters: {str(e)}")
        
        return params
    
    def _calculate_normalization_stats(self, 
                                     original_data: pd.DataFrame, 
                                     normalized_data: pd.DataFrame):
        """Calculate and store normalization statistics"""
        
        try:
            stats = {}
            
            for col in original_data.columns:
                if col in normalized_data.columns:
                    stats[col] = {
                        'original_mean': float(original_data[col].mean()),
                        'original_std': float(original_data[col].std()),
                        'original_min': float(original_data[col].min()),
                        'original_max': float(original_data[col].max()),
                        'normalized_mean': float(normalized_data[col].mean()),
                        'normalized_std': float(normalized_data[col].std()),
                        'normalized_min': float(normalized_data[col].min()),
                        'normalized_max': float(normalized_data[col].max())
                    }
            
            self.normalization_stats = stats
            
        except Exception as e:
            logger.error(f"Error calculating normalization stats: {str(e)}")
    
    def inverse_transform(self, 
                         data: pd.DataFrame, 
                         method: str = 'standard',
                         columns: Optional[list] = None) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale
        
        Args:
            data: Normalized DataFrame
            method: Normalization method used
            columns: Specific columns to inverse transform
            
        Returns:
            Data in original scale
        """
        
        if method not in self.fitted_scalers:
            logger.error(f"Normalizer {method} not fitted")
            return data
        
        inverse_data = data.copy()
        
        try:
            # Select columns to inverse transform
            if columns is None:
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in columns if col in data.columns]
            
            if not numeric_columns:
                return data
            
            scaler = self.fitted_scalers[method]
            
            if isinstance(scaler, dict):
                # Custom method parameters
                if method == 'zscore':
                    for col in numeric_columns:
                        if col in scaler:
                            mean_val = scaler[col]['mean']
                            std_val = scaler[col]['std']
                            inverse_data[col] = data[col] * std_val + mean_val
                
                elif method == 'unit_vector':
                    for col in numeric_columns:
                        if col in scaler:
                            norm_val = scaler[col]['norm']
                            inverse_data[col] = data[col] * norm_val
                
                elif method == 'decimal_scaling':
                    for col in numeric_columns:
                        if col in scaler:
                            scale_factor = scaler[col]['scale_factor']
                            inverse_data[col] = data[col] * scale_factor
            else:
                # sklearn scaler
                if hasattr(scaler, 'inverse_transform'):
                    inverse_transformed = scaler.inverse_transform(data[numeric_columns])
                    inverse_data[numeric_columns] = inverse_transformed
                else:
                    logger.warning(f"Scaler {method} does not support inverse transform")
            
            logger.info(f"Applied inverse {method} transformation to {len(numeric_columns)} columns")
            
        except Exception as e:
            logger.error(f"Error in inverse_transform: {str(e)}")
        
        return inverse_data
    
    def get_normalization_report(self) -> Dict[str, Any]:
        """Generate comprehensive normalization report"""
        
        report = {
            'fitted_methods': list(self.fitted_scalers.keys()),
            'normalization_stats': self.normalization_stats,
            'available_methods': list(self.methods.keys())
        }
        
        # Add detailed statistics
        if self.normalization_stats:
            report['summary'] = {
                'num_normalized_columns': len(self.normalization_stats),
                'mean_reduction_factor': np.mean([
                    stats['original_std'] / (stats['normalized_std'] + 1e-12) 
                    for stats in self.normalization_stats.values()
                ]),
                'columns': list(self.normalization_stats.keys())
            }
        
        return report
    
    def save_normalizer(self, filepath: str, method: str = 'standard'):
        """Save fitted normalizer to file"""
        
        try:
            if method not in self.fitted_scalers:
                logger.error(f"Method {method} not fitted")
                return False
            
            save_data = {
                'scaler': self.fitted_scalers[method],
                'method': method,
                'stats': self.normalization_stats
            }
            
            joblib.dump(save_data, filepath)
            logger.info(f"Normalizer saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving normalizer: {str(e)}")
            return False
    
    def load_normalizer(self, filepath: str) -> bool:
        """Load fitted normalizer from file"""
        
        try:
            save_data = joblib.load(filepath)
            
            method = save_data['method']
            self.fitted_scalers[method] = save_data['scaler']
            self.normalization_stats = save_data.get('stats', {})
            
            logger.info(f"Normalizer {method} loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading normalizer: {str(e)}")
            return False
