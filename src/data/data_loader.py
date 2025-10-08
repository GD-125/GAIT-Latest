# File: src/data/data_loader.py
# Data loading and validation utilities

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import io
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading and validation for gait analysis"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        self.required_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        self.optional_columns = ['emg_signal', 'timestamp', 'subject_id', 'label']
    
    def load_file(self, file_obj) -> pd.DataFrame:
        """
        Load data from uploaded file object
        
        Args:
            file_obj: Streamlit file upload object
            
        Returns:
            DataFrame with loaded data
        """
        try:
            filename = file_obj.name
            file_extension = Path(filename).suffix.lower()
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Read file content
            if file_extension == '.csv':
                data = pd.read_csv(file_obj)
            elif file_extension in ['.xlsx', '.xls']:
                data = pd.read_excel(file_obj)
            else:
                raise ValueError(f"Unsupported format: {file_extension}")
            
            logger.info(f"Loaded data with shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data for gait analysis requirements
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Check if dataframe is empty
            if data.empty:
                results['is_valid'] = False
                results['errors'].append("Dataset is empty")
                return results
            
            # Check required columns
            missing_columns = [col for col in self.required_columns if col not in data.columns]
            if missing_columns:
                results['warnings'].append(f"Missing recommended columns: {missing_columns}")
            
            # Check data types and ranges
            sensor_columns = [col for col in self.required_columns if col in data.columns]
            
            for col in sensor_columns:
                # Check for numeric data
                if not pd.api.types.is_numeric_dtype(data[col]):
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                        nan_count = data[col].isna().sum()
                        if nan_count > 0:
                            results['warnings'].append(f"Converted {nan_count} non-numeric values to NaN in {col}")
                    except:
                        results['errors'].append(f"Column {col} contains non-numeric data")
                        results['is_valid'] = False
                
                # Check for reasonable sensor ranges
                if col.startswith('accel'):
                    # Accelerometer typically ±16g for gait analysis
                    outliers = ((data[col].abs() > 50) & (data[col].notna())).sum()
                    if outliers > 0:
                        results['warnings'].append(f"{col}: {outliers} values outside typical range (±50 m/s²)")
                
                elif col.startswith('gyro'):
                    # Gyroscope typically ±2000°/s for gait analysis
                    outliers = ((data[col].abs() > 35) & (data[col].notna())).sum()
                    if outliers > 0:
                        results['warnings'].append(f"{col}: {outliers} values outside typical range (±35 rad/s)")
            
            # Check timestamp if available
            if 'timestamp' in data.columns:
                try:
                    timestamps = pd.to_datetime(data['timestamp'])
                    duration = (timestamps.max() - timestamps.min()).total_seconds()
                    results['info']['duration'] = duration
                    
                    if duration < 30:
                        results['warnings'].append("Recording duration < 30 seconds (minimum recommended)")
                    
                    # Estimate sampling rate
                    time_diffs = timestamps.diff().dropna()
                    median_interval = time_diffs.median().total_seconds()
                    if median_interval > 0:
                        sampling_rate = 1 / median_interval
                        results['info']['estimated_sampling_rate'] = sampling_rate
                        
                        if sampling_rate < 50:
                            results['warnings'].append("Sampling rate < 50 Hz (minimum recommended for gait analysis)")
                
                except Exception as e:
                    results['warnings'].append(f"Could not process timestamps: {str(e)}")
            
            # Data quality checks
            total_samples = len(data)
            results['info']['total_samples'] = total_samples
            results['info']['columns'] = list(data.columns)
            
            # Missing data check
            missing_data = data.isnull().sum()
            if missing_data.sum() > 0:
                missing_percentage = (missing_data.sum() / (len(data) * len(data.columns))) * 100
                results['info']['missing_data_percentage'] = missing_percentage
                
                if missing_percentage > 10:
                    results['warnings'].append(f"High missing data percentage: {missing_percentage:.1f}%")
            
            # Check for constant values (sensor malfunction)
            for col in sensor_columns:
                if data[col].std() < 1e-6:
                    results['warnings'].append(f"Column {col} has very low variance (possible sensor malfunction)")
            
            logger.info(f"Data validation completed. Valid: {results['is_valid']}")
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            results['is_valid'] = False
            results['errors'].append(f"Validation error: {str(e)}")
        
        return results
    
    def generate_sample_data(self, 
                           n_samples: int = 1000,
                           sampling_rate: int = 50,
                           duration_seconds: int = 20) -> pd.DataFrame:
        """
        Generate sample gait data for testing
        
        Args:
            n_samples: Number of samples to generate
            sampling_rate: Sampling rate in Hz
            duration_seconds: Duration in seconds
            
        Returns:
            DataFrame with synthetic gait data
        """
        
        if n_samples != sampling_rate * duration_seconds:
            n_samples = sampling_rate * duration_seconds
        
        # Time vector
        time_vector = np.linspace(0, duration_seconds, n_samples)
        
        # Generate realistic gait patterns
        step_frequency = 1.8  # ~1.8 Hz typical step frequency
        
        # Accelerometer data (m/s²)
        # Vertical acceleration (main gait component)
        accel_z = 9.81 + 2.0 * np.sin(2 * np.pi * step_frequency * time_vector) + \
                  np.random.normal(0, 0.5, n_samples)
        
        # Anterior-posterior acceleration
        accel_x = 1.5 * np.sin(2 * np.pi * step_frequency * time_vector + np.pi/4) + \
                  np.random.normal(0, 0.3, n_samples)
        
        # Medio-lateral acceleration  
        accel_y = 0.8 * np.sin(4 * np.pi * step_frequency * time_vector) + \
                  np.random.normal(0, 0.2, n_samples)
        
        # Gyroscope data (rad/s)
        # Sagittal plane rotation (main walking component)
        gyro_x = 0.5 * np.sin(2 * np.pi * step_frequency * time_vector + np.pi/2) + \
                 np.random.normal(0, 0.1, n_samples)
        
        # Frontal plane rotation
        gyro_y = 0.3 * np.sin(2 * np.pi * step_frequency * time_vector) + \
                 np.random.normal(0, 0.05, n_samples)
        
        # Transverse plane rotation
        gyro_z = 0.2 * np.sin(4 * np.pi * step_frequency * time_vector + np.pi/3) + \
                 np.random.normal(0, 0.03, n_samples)
        
        # EMG signal (μV) - optional
        emg_signal = 50 + 20 * np.abs(np.sin(2 * np.pi * step_frequency * time_vector)) + \
                     np.random.normal(0, 5, n_samples)
        
        # Create timestamps
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=t) for t in time_vector]
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'accel_x': accel_x,
            'accel_y': accel_y, 
            'accel_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
            'emg_signal': emg_signal,
            'subject_id': ['SAMPLE_001'] * n_samples,
            'label': ['Normal'] * n_samples
        })
        
        logger.info(f"Generated sample data with {n_samples} samples at {sampling_rate} Hz")
        return data


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
                    segment: bool = True,
                    extract_features: bool = True,
                    window_size: Optional[int] = None,
                    overlap: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline
        
        Args:
            data: Input sensor data
            denoise: Apply noise filtering
            normalize: Apply normalization
            segment: Create segments
            extract_features: Extract features from segments
            window_size: Size of window for segmentation
            overlap: Overlap between windows
        
        Returns:
            Dictionary with processed data and features
        """
        # TODO: Implement preprocessing steps
        # For now, just return the input data as a placeholder
        return {"processed_data": data}