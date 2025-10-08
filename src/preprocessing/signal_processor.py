# File: src/preprocessing/signal_processor.py
# Signal preprocessing for gait analysis

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pywt
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

class SignalProcessor:
    """Signal preprocessing for gait analysis data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.filter_params = {
            'lowpass_freq': 20,  # Hz
            'highpass_freq': 0.5,  # Hz
            'filter_order': 4
        }
    
    def process_data(self,
                    data: pd.DataFrame,
                    denoise: bool = True,
                    normalize: bool = True,
                    segment: bool = False,
                    extract_features: bool = False,
                    window_size: Optional[int] = None,
                    overlap: Optional[int] = None,
                    sampling_rate: int = 50) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline
        
        Args:
            data: Input sensor data
            denoise: Apply noise filtering
            normalize: Apply normalization
            segment: Create segments (optional)
            extract_features: Extract features (optional)
            window_size: Window size for segmentation
            overlap: Overlap size for segmentation
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with processed data and metadata
        """
        
        try:
            processed_data = data.copy()
            metadata = {
                'original_shape': data.shape,
                'processing_steps': []
            }
            
            # 1. Denoise
            if denoise:
                logger.info("Applying denoising filters...")
                processed_data = self.denoise(processed_data, sampling_rate)
                metadata['processing_steps'].append('denoise')
            
            # 2. Normalize
            if normalize:
                logger.info("Applying normalization...")
                processed_data = self.normalize(processed_data)
                metadata['processing_steps'].append('normalize')
            
            # 3. Handle missing values
            processed_data = self.handle_missing_values(processed_data)
            metadata['processing_steps'].append('handle_missing')
            
            metadata['final_shape'] = processed_data.shape
            
            result = {
                'processed_data': processed_data,
                'metadata': metadata
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {str(e)}")
            return {
                'processed_data': data,
                'metadata': {'error': str(e)}
            }
    
    def denoise(self, data: pd.DataFrame, sampling_rate: int = 50) -> pd.DataFrame:
        """
        Apply denoising filters to sensor data
        
        Args:
            data: Input DataFrame
            sampling_rate: Sampling frequency in Hz
            
        Returns:
            Denoised DataFrame
        """
        
        denoised_data = data.copy()
        
        try:
            # Get numeric columns only
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                signal_data = data[col].values
                
                # Skip if all NaN
                if np.all(np.isnan(signal_data)):
                    continue
                
                # Fill NaN temporarily for filtering
                signal_filled = pd.Series(signal_data).ffill().bfill().values
                
                # Apply bandpass filter
                filtered = self.bandpass_filter(
                    signal_filled,
                    self.filter_params['highpass_freq'],
                    self.filter_params['lowpass_freq'],
                    sampling_rate,
                    self.filter_params['filter_order']
                )
                
                # Apply wavelet denoising
                denoised = self.wavelet_denoise(filtered)
                
                denoised_data[col] = denoised
            
            logger.info(f"Denoised {len(numeric_cols)} columns")
            
        except Exception as e:
            logger.error(f"Error in denoising: {str(e)}")
            return data
        
        return denoised_data
    
    def bandpass_filter(self,
                       signal_data: np.ndarray,
                       lowcut: float,
                       highcut: float,
                       fs: int,
                       order: int = 4) -> np.ndarray:
        """
        Apply Butterworth bandpass filter
        
        Args:
            signal_data: Input signal
            lowcut: Low cutoff frequency
            highcut: High cutoff frequency
            fs: Sampling frequency
            order: Filter order
            
        Returns:
            Filtered signal
        """
        
        try:
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            
            # Design filter
            b, a = signal.butter(order, [low, high], btype='band')
            
            # Apply filter (zero-phase filtering)
            filtered_signal = signal.filtfilt(b, a, signal_data)
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Error in bandpass filter: {str(e)}")
            return signal_data
    
    def lowpass_filter(self,
                      signal_data: np.ndarray,
                      cutoff: float,
                      fs: int,
                      order: int = 4) -> np.ndarray:
        """
        Apply Butterworth lowpass filter
        
        Args:
            signal_data: Input signal
            cutoff: Cutoff frequency
            fs: Sampling frequency
            order: Filter order
            
        Returns:
            Filtered signal
        """
        
        try:
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            
            b, a = signal.butter(order, normal_cutoff, btype='low')
            filtered_signal = signal.filtfilt(b, a, signal_data)
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Error in lowpass filter: {str(e)}")
            return signal_data
    
    def highpass_filter(self,
                       signal_data: np.ndarray,
                       cutoff: float,
                       fs: int,
                       order: int = 4) -> np.ndarray:
        """
        Apply Butterworth highpass filter
        
        Args:
            signal_data: Input signal
            cutoff: Cutoff frequency
            fs: Sampling frequency
            order: Filter order
            
        Returns:
            Filtered signal
        """
        
        try:
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            
            b, a = signal.butter(order, normal_cutoff, btype='high')
            filtered_signal = signal.filtfilt(b, a, signal_data)
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Error in highpass filter: {str(e)}")
            return signal_data
    
    def wavelet_denoise(self,
                       signal_data: np.ndarray,
                       wavelet: str = 'db4',
                       level: Optional[int] = None) -> np.ndarray:
        """
        Apply wavelet denoising
        
        Args:
            signal_data: Input signal
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Denoised signal
        """
        
        try:
            if level is None:
                level = min(6, int(np.log2(len(signal_data))))
            
            if level < 1:
                return signal_data
            
            # Decompose signal
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)
            
            # Calculate threshold
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
            
            # Apply soft thresholding
            denoised_coeffs = [coeffs[0]]  # Keep approximation coefficients
            for detail_coeffs in coeffs[1:]:
                denoised_coeffs.append(pywt.threshold(detail_coeffs, threshold, mode='soft'))
            
            # Reconstruct signal
            denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
            
            # Handle length mismatch
            if len(denoised_signal) > len(signal_data):
                denoised_signal = denoised_signal[:len(signal_data)]
            elif len(denoised_signal) < len(signal_data):
                denoised_signal = np.pad(denoised_signal, (0, len(signal_data) - len(denoised_signal)), mode='edge')
            
            return denoised_signal
            
        except Exception as e:
            logger.error(f"Error in wavelet denoising: {str(e)}")
            return signal_data
    
    def normalize(self,
                 data: pd.DataFrame,
                 method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize data
        
        Args:
            data: Input DataFrame
            method: Normalization method ('zscore', 'minmax')
            
        Returns:
            Normalized DataFrame
        """
        
        normalized_data = data.copy()
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if method == 'zscore':
                for col in numeric_cols:
                    col_data = data[col].values
                    if not np.all(np.isnan(col_data)):
                        mean_val = np.nanmean(col_data)
                        std_val = np.nanstd(col_data)
                        if std_val > 1e-10:
                            normalized_data[col] = (col_data - mean_val) / std_val
                        else:
                            normalized_data[col] = col_data - mean_val
            
            elif method == 'minmax':
                for col in numeric_cols:
                    col_data = data[col].values
                    if not np.all(np.isnan(col_data)):
                        min_val = np.nanmin(col_data)
                        max_val = np.nanmax(col_data)
                        if max_val - min_val > 1e-10:
                            normalized_data[col] = (col_data - min_val) / (max_val - min_val)
                        else:
                            normalized_data[col] = 0
            
            logger.info(f"Normalized {len(numeric_cols)} columns using {method}")
            
        except Exception as e:
            logger.error(f"Error in normalization: {str(e)}")
            return data
        
        return normalized_data
    
    def handle_missing_values(self,
                             data: pd.DataFrame,
                             method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            data: Input DataFrame
            method: Method for handling missing values
            
        Returns:
            DataFrame with handled missing values
        """
        
        handled_data = data.copy()
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if method == 'interpolate':
                for col in numeric_cols:
                    handled_data[col] = handled_data[col].interpolate(method='linear', limit_direction='both')
            
            elif method == 'forward_fill':
                handled_data[numeric_cols] = handled_data[numeric_cols].ffill()
            
            elif method == 'backward_fill':
                handled_data[numeric_cols] = handled_data[numeric_cols].bfill()
            
            elif method == 'mean':
                for col in numeric_cols:
                    mean_val = handled_data[col].mean()
                    handled_data[col] = handled_data[col].fillna(mean_val)
            
            # Final cleanup - replace any remaining NaN with 0
            handled_data[numeric_cols] = handled_data[numeric_cols].fillna(0)
            
            logger.info(f"Handled missing values using {method}")
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return data
        
        return handled_data
    
    def remove_outliers(self,
                       data: pd.DataFrame,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove or replace outliers
        
        Args:
            data: Input DataFrame
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        
        cleaned_data = data.copy()
        
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = data[col].values
                
                if method == 'iqr':
                    Q1 = np.nanpercentile(col_data, 25)
                    Q3 = np.nanpercentile(col_data, 75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                elif method == 'zscore':
                    mean_val = np.nanmean(col_data)
                    std_val = np.nanstd(col_data)
                    lower_bound = mean_val - threshold * std_val
                    upper_bound = mean_val + threshold * std_val
                
                # Replace outliers with bounds
                cleaned_data[col] = np.clip(col_data, lower_bound, upper_bound)
            
            logger.info(f"Removed outliers from {len(numeric_cols)} columns using {method}")
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            return data
        
        return cleaned_data
    
    def resample_data(self,
                     data: pd.DataFrame,
                     target_rate: int,
                     current_rate: int,
                     time_column: Optional[str] = None) -> pd.DataFrame:
        """
        Resample data to target sampling rate
        
        Args:
            data: Input DataFrame
            target_rate: Target sampling rate in Hz
            current_rate: Current sampling rate in Hz
            time_column: Name of time column (if exists)
            
        Returns:
            Resampled DataFrame
        """
        
        try:
            if target_rate == current_rate:
                return data
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            resampled_data = {}
            
            # Calculate resampling ratio
            ratio = target_rate / current_rate
            new_length = int(len(data) * ratio)
            
            for col in numeric_cols:
                col_data = data[col].values
                # Use scipy's resample
                resampled_signal = signal.resample(col_data, new_length)
                resampled_data[col] = resampled_signal
            
            resampled_df = pd.DataFrame(resampled_data)
            
            logger.info(f"Resampled data from {current_rate}Hz to {target_rate}Hz")
            
            return resampled_df
            
        except Exception as e:
            logger.error(f"Error resampling data: {str(e)}")
            return data