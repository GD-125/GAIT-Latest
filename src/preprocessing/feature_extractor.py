# File: src/preprocessing/feature_extractor.py
# Advanced feature extraction for gait analysis

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import pywavelets as pywt
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Comprehensive feature extraction for gait analysis"""
    
    def __init__(self):
        self.feature_categories = {
            'time_domain': True,
            'frequency_domain': True, 
            'wavelet_domain': True,
            'statistical': True,
            'morphological': True,
            'nonlinear': True
        }
        
        self.extracted_features = {}
        self.feature_names = []
        
    def extract_comprehensive_features(self, 
                                     data: pd.DataFrame,
                                     sampling_rate: int = 50,
                                     categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract comprehensive feature set from sensor data
        
        Args:
            data: Input sensor DataFrame
            sampling_rate: Sampling frequency in Hz
            categories: List of feature categories to extract
            
        Returns:
            DataFrame with extracted features
        """
        
        if categories is None:
            categories = list(self.feature_categories.keys())
        
        logger.info(f"Extracting features: {categories}")
        
        # Identify sensor columns
        sensor_columns = self._identify_sensor_columns(data)
        
        if not sensor_columns:
            logger.error("No sensor columns found in data")
            return pd.DataFrame()
        
        all_features = {}
        
        try:
            for sensor_group, columns in sensor_columns.items():
                logger.debug(f"Processing {sensor_group} sensors: {columns}")
                
                for col in columns:
                    if col not in data.columns:
                        continue
                    
                    signal_data = data[col].dropna().values
                    
                    if len(signal_data) < 10:
                        logger.warning(f"Insufficient data for {col}, skipping")
                        continue
                    
                    # Extract features by category
                    if 'time_domain' in categories:
                        time_features = self._extract_time_domain_features(signal_data, col)
                        all_features.update(time_features)
                    
                    if 'frequency_domain' in categories:
                        freq_features = self._extract_frequency_domain_features(
                            signal_data, col, sampling_rate
                        )
                        all_features.update(freq_features)
                    
                    if 'wavelet_domain' in categories:
                        wavelet_features = self._extract_wavelet_features(signal_data, col)
                        all_features.update(wavelet_features)
                    
                    if 'statistical' in categories:
                        stat_features = self._extract_statistical_features(signal_data, col)
                        all_features.update(stat_features)
                    
                    if 'morphological' in categories:
                        morph_features = self._extract_morphological_features(signal_data, col)
                        all_features.update(morph_features)
                    
                    if 'nonlinear' in categories:
                        nonlinear_features = self._extract_nonlinear_features(signal_data, col)
                        all_features.update(nonlinear_features)
                
                # Extract cross-sensor features
                if len(columns) > 1 and 'statistical' in categories:
                    cross_features = self._extract_cross_sensor_features(data[columns], sensor_group)
                    all_features.update(cross_features)
            
            # Extract gait-specific features
            if 'morphological' in categories:
                gait_features = self._extract_gait_specific_features(data, sampling_rate)
                all_features.update(gait_features)
            
            # Create feature DataFrame
            feature_df = pd.DataFrame([all_features])
            self.feature_names = list(all_features.keys())
            
            logger.info(f"Extracted {len(all_features)} features")
            
            return feature_df
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            return pd.DataFrame()
    
    def _identify_sensor_columns(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify and group sensor columns"""
        
        sensor_groups = {
            'accelerometer': [],
            'gyroscope': [],
            'emg': [],
            'other': []
        }
        
        for col in data.columns:
            col_lower = col.lower()
            
            if 'accel' in col_lower:
                sensor_groups['accelerometer'].append(col)
            elif 'gyro' in col_lower:
                sensor_groups['gyroscope'].append(col)
            elif 'emg' in col_lower:
                sensor_groups['emg'].append(col)
            elif any(term in col_lower for term in ['force', 'pressure', 'angle']):
                sensor_groups['other'].append(col)
        
        # Remove empty groups
        sensor_groups = {k: v for k, v in sensor_groups.items() if v}
        
        return sensor_groups
    
    def _extract_time_domain_features(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """Extract time domain features"""
        
        features = {}
        
        try:
            # Basic statistics
            features[f'{prefix}_mean'] = float(np.mean(signal))
            features[f'{prefix}_std'] = float(np.std(signal))
            features[f'{prefix}_var'] = float(np.var(signal))
            features[f'{prefix}_min'] = float(np.min(signal))
            features[f'{prefix}_max'] = float(np.max(signal))
            features[f'{prefix}_range'] = float(np.ptp(signal))
            features[f'{prefix}_median'] = float(np.median(signal))
            
            # Percentiles
            features[f'{prefix}_p25'] = float(np.percentile(signal, 25))
            features[f'{prefix}_p75'] = float(np.percentile(signal, 75))
            features[f'{prefix}_iqr'] = features[f'{prefix}_p75'] - features[f'{prefix}_p25']
            
            # Energy and power
            features[f'{prefix}_energy'] = float(np.sum(signal**2))
            features[f'{prefix}_rms'] = float(np.sqrt(np.mean(signal**2)))
            features[f'{prefix}_abs_mean'] = float(np.mean(np.abs(signal)))
            
            # Shape characteristics
            features[f'{prefix}_skewness'] = float(stats.skew(signal))
            features[f'{prefix}_kurtosis'] = float(stats.kurtosis(signal))
            
            # Zero crossings
            zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
            features[f'{prefix}_zero_crossings'] = len(zero_crossings)
            features[f'{prefix}_zero_crossing_rate'] = len(zero_crossings) / len(signal)
            
            # Peak detection
            peaks, _ = signal.find_peaks(signal, distance=10)
            features[f'{prefix}_num_peaks'] = len(peaks)
            
            if len(peaks) > 1:
                peak_intervals = np.diff(peaks)
                features[f'{prefix}_mean_peak_interval'] = float(np.mean(peak_intervals))
                features[f'{prefix}_std_peak_interval'] = float(np.std(peak_intervals))
            else:
                features[f'{prefix}_mean_peak_interval'] = 0.0
                features[f'{prefix}_std_peak_interval'] = 0.0
            
            # Autocorrelation features
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find first minimum in autocorrelation (period estimation)
            if len(autocorr) > 10:
                min_idx = np.argmin(autocorr[5:25]) + 5  # Look in reasonable range
                features[f'{prefix}_autocorr_min_lag'] = float(min_idx)
                features[f'{prefix}_autocorr_min_value'] = float(autocorr[min_idx])
            else:
                features[f'{prefix}_autocorr_min_lag'] = 0.0
                features[f'{prefix}_autocorr_min_value'] = 0.0
            
            # Derivative features
            first_diff = np.diff(signal)
            features[f'{prefix}_diff_mean'] = float(np.mean(first_diff))
            features[f'{prefix}_diff_std'] = float(np.std(first_diff))
            features[f'{prefix}_diff_energy'] = float(np.sum(first_diff**2))
            
            second_diff = np.diff(first_diff)
            features[f'{prefix}_diff2_mean'] = float(np.mean(second_diff))
            features[f'{prefix}_diff2_std'] = float(np.std(second_diff))
            
        except Exception as e:
            logger.warning(f"Error extracting time domain features for {prefix}: {str(e)}")
            # Return zeros for failed features
            features.update({f'{prefix}_{feat}': 0.0 for feat in [
                'mean', 'std', 'var', 'min', 'max', 'range', 'median', 'p25', 'p75', 'iqr',
                'energy', 'rms', 'abs_mean', 'skewness', 'kurtosis', 'zero_crossings', 
                'zero_crossing_rate', 'num_peaks', 'mean_peak_interval', 'std_peak_interval',
                'autocorr_min_lag', 'autocorr_min_value', 'diff_mean', 'diff_std', 
                'diff_energy', 'diff2_mean', 'diff2_std'
            ]})
        
        return features
    
    def _extract_frequency_domain_features(self, signal: np.ndarray, prefix: str, 
                                         sampling_rate: int = 50) -> Dict[str, float]:
        """Extract frequency domain features"""
        
        features = {}
        
        try:
            # FFT
            n_samples = len(signal)
            fft_signal = fft(signal)
            frequencies = fftfreq(n_samples, 1/sampling_rate)
            
            # Keep only positive frequencies
            positive_freq_mask = frequencies > 0
            positive_frequencies = frequencies[positive_freq_mask]
            magnitude_spectrum = np.abs(fft_signal[positive_freq_mask])
            power_spectrum = magnitude_spectrum**2
            
            # Normalize
            power_spectrum = power_spectrum / np.sum(power_spectrum)
            
            # Spectral features
            features[f'{prefix}_spectral_centroid'] = float(
                np.sum(positive_frequencies * power_spectrum)
            )
            
            features[f'{prefix}_spectral_spread'] = float(
                np.sqrt(np.sum(((positive_frequencies - features[f'{prefix}_spectral_centroid'])**2) * power_spectrum))
            )
            
            features[f'{prefix}_spectral_rolloff'] = float(
                self._calculate_spectral_rolloff(positive_frequencies, power_spectrum)
            )
            
            features[f'{prefix}_spectral_flux'] = float(np.sum(np.diff(magnitude_spectrum)**2))
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(magnitude_spectrum)
            features[f'{prefix}_dominant_frequency'] = float(positive_frequencies[dominant_freq_idx])
            features[f'{prefix}_dominant_frequency_power'] = float(magnitude_spectrum[dominant_freq_idx])
            
            # Frequency band powers
            # Low frequency (0-2 Hz)
            low_freq_mask = (positive_frequencies >= 0) & (positive_frequencies <= 2)
            features[f'{prefix}_power_0_2hz'] = float(np.sum(power_spectrum[low_freq_mask]))
            
            # G