"""
File: tests/test_preprocessing.py
Unit tests for preprocessing components
"""

import pytest
import numpy as np
import pandas as pd

from src.preprocessing.signal_processor import SignalProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.normalizer import Normalizer
from src.preprocessing.segmentation import WindowSegmenter


class TestSignalProcessor:
    """Tests for signal processing"""
    
    def test_initialization(self, signal_processor):
        """Test signal processor initialization"""
        assert isinstance(signal_processor, SignalProcessor)
    
    def test_filter_signal(self, signal_processor, sample_gait_data):
        """Test signal filtering"""
        signal = sample_gait_data['accel_x'].values
        
        filtered = signal_processor.butterworth_filter(
            signal,
            cutoff=20,
            fs=100,
            order=4
        )
        
        assert len(filtered) == len(signal)
        assert not np.isnan(filtered).any()
    
    def test_remove_noise(self, signal_processor, sample_gait_data):
        """Test noise removal"""
        # Add noise to signal
        signal = sample_gait_data['accel_x'].values
        noisy_signal = signal + np.random.normal(0, 0.1, len(signal))
        
        denoised = signal_processor.wavelet_denoise(noisy_signal)
        
        assert len(denoised) == len(signal)
        
        # Check that denoised signal is closer to original
        mse_noisy = np.mean((noisy_signal - signal) ** 2)
        mse_denoised = np.mean((denoised - signal) ** 2)
        
        assert mse_denoised < mse_noisy
    
    def test_process_dataframe(self, signal_processor, sample_gait_data):
        """Test processing entire dataframe"""
        processed = signal_processor.process(sample_gait_data)
        
        assert isinstance(processed, pd.DataFrame)
        assert processed.shape == sample_gait_data.shape
        assert not processed.isna().any().any()


class TestFeatureExtractor:
    """Tests for feature extraction"""
    
    def test_time_domain_features(self, feature_extractor, sample_gait_data):
        """Test time domain feature extraction"""
        features = feature_extractor.extract_time_domain_features(sample_gait_data)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check for expected features
        expected_features = ['mean', 'std', 'max', 'min', 'rms']
        
        for expected in expected_features:
            assert any(expected in key for key in features.keys())
    
    def test_frequency_domain_features(self, feature_extractor):
        """Test frequency domain feature extraction"""
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
        
        features = feature_extractor.extract_frequency_features(
            signal,
            sampling_rate=100
        )
        
        assert 'dominant_frequency' in features
        assert 'spectral_energy' in features
        assert features['dominant_frequency'] == pytest.approx(5, rel=0.5)
    
    def test_statistical_features(self, feature_extractor):
        """Test statistical feature extraction"""
        signal = np.random.randn(1000)
        
        features = feature_extractor.extract_statistical_features(signal)
        
        assert 'mean' in features
        assert 'std' in features
        assert 'skewness' in features
        assert 'kurtosis' in features
        
        # Verify values
        assert features['mean'] == pytest.approx(np.mean(signal), rel=0.01)
        assert features['std'] == pytest.approx(np.std(signal), rel=0.01)


class TestNormalizer:
    """Tests for data normalization"""
    
    def test_zscore_normalization(self):
        """Test z-score normalization"""
        normalizer = Normalizer(method='zscore')
        
        data = np.random.randn(100, 9)
        
        # Fit and transform
        normalized = normalizer.fit_transform(data)
        
        # Check mean is close to 0 and std close to 1
        assert np.allclose(np.mean(normalized, axis=0), 0, atol=0.1)
        assert np.allclose(np.std(normalized, axis=0), 1, atol=0.1)
    
    def test_minmax_normalization(self):
        """Test min-max normalization"""
        normalizer = Normalizer(method='minmax')
        
        data = np.random.randn(100, 9)
        
        normalized = normalizer.fit_transform(data)
        
        # Check range is [0, 1]
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
    
    def test_robust_normalization(self):
        """Test robust normalization (with outliers)"""
        normalizer = Normalizer(method='robust')
        
        # Create data with outliers
        data = np.random.randn(100, 9)
        data[0, :] = 100  # Outlier
        
        normalized = normalizer.fit_transform(data)
        
        # Check normalization worked despite outliers
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()


class TestWindowSegmenter:
    """Tests for time series segmentation"""
    
    def test_fixed_window_segmentation(self):
        """Test fixed-size window segmentation"""
        segmenter = WindowSegmenter(window_size=100, step_size=50)
        
        data = np.random.randn(500, 9)
        
        windows = segmenter.segment(data)
        
        assert len(windows) > 0
        assert all(w.shape == (100, 9) for w in windows)
        
        # Calculate expected number of windows
        expected_windows = (500 - 100) // 50 + 1
        assert len(windows) == expected_windows
    
    def test_sliding_window(self):
        """Test sliding window with overlap"""
        segmenter = WindowSegmenter(window_size=100, step_size=25)
        
        data = np.random.randn(300, 9)
        
        windows = segmenter.segment(data)
        
        # With overlap, should have more windows
        assert len(windows) > 0
        
        # Check overlap
        if len(windows) > 1:
            overlap = segmenter.window_size - segmenter.step_size
            assert overlap == 75
    
    def test_edge_cases(self):
        """Test edge cases in segmentation"""
        segmenter = WindowSegmenter(window_size=100, step_size=100)
        
        # Data shorter than window
        short_data = np.random.randn(50, 9)
        windows = segmenter.segment(short_data, pad=True)
        
        assert len(windows) == 1
        assert windows[0].shape[0] == 100  # Padded


@pytest.mark.unit
class TestDataValidation:
    """Tests for data validation"""
    
    def test_required_columns(self, sample_gait_data, helpers):
        """Test validation of required columns"""
        helpers.assert_valid_dataframe(sample_gait_data)
    
    def test_missing_values_detection(self):
        """Test detection of missing values"""
        data = pd.DataFrame({
            'accel_x': [1, 2, np.nan, 4],
            'accel_y': [1, 2, 3, 4]
        })
        
        missing = data.isna().sum()
        
        assert missing['accel_x'] == 1
        assert missing['accel_y'] == 0
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        # Normal data with outliers
        data = np.concatenate([
            np.random.randn(95),
            np.array([10, -10, 15, -15, 20])  # Outliers
        ])
        
        # Z-score method
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = z_scores > 3
        
        assert np.sum(outliers) >= 3  # At least 3 outliers detected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
