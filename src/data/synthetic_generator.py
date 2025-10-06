# File: src/data/synthetic_generator.py
# Synthetic medical sensor data generation for testing and training

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from scipy import signal
from scipy.stats import multivariate_normal
import random

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate realistic synthetic gait and medical sensor data"""
    
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        # Disease-specific parameters
        self.disease_parameters = {
            'Normal': {
                'step_frequency': (1.6, 2.0),  # Hz
                'step_variability': 0.05,
                'acceleration_amplitude': (2.0, 3.0),
                'gyroscope_amplitude': (0.5, 1.0),
                'tremor_frequency': None,
                'tremor_amplitude': 0.0,
                'asymmetry_factor': 1.0,
                'noise_level': 0.1
            },
            'Parkinson': {
                'step_frequency': (1.2, 1.6),  # Slower gait
                'step_variability': 0.15,  # Higher variability
                'acceleration_amplitude': (1.5, 2.5),  # Reduced amplitude
                'gyroscope_amplitude': (0.3, 0.8),
                'tremor_frequency': (4, 6),  # 4-6 Hz tremor
                'tremor_amplitude': 0.3,
                'asymmetry_factor': 1.2,  # Slight asymmetry
                'noise_level': 0.15,
                'freezing_episodes': True
            },
            'Huntington': {
                'step_frequency': (1.0, 1.8),  # Variable frequency
                'step_variability': 0.25,  # High variability
                'acceleration_amplitude': (1.8, 4.0),  # Irregular amplitude
                'gyroscope_amplitude': (0.8, 2.0),  # High rotational activity
                'tremor_frequency': None,
                'tremor_amplitude': 0.0,
                'asymmetry_factor': 1.5,  # Higher asymmetry
                'noise_level': 0.2,
                'chorea_pattern': True
            },
            'Ataxia': {
                'step_frequency': (1.1, 1.5),  # Slower, unsteady
                'step_variability': 0.30,  # Very high variability
                'acceleration_amplitude': (1.2, 2.8),
                'gyroscope_amplitude': (0.6, 1.5),
                'tremor_frequency': (2, 4),  # Lower frequency tremor
                'tremor_amplitude': 0.2,
                'asymmetry_factor': 1.8,  # High asymmetry
                'noise_level': 0.25,
                'coordination_impairment': True
            },
            'MultipleSclerosis': {
                'step_frequency': (1.3, 1.7),
                'step_variability': 0.20,  # Moderate variability
                'acceleration_amplitude': (1.6, 2.8),
                'gyroscope_amplitude': (0.4, 1.2),
                'tremor_frequency': (3, 5),  # Intention tremor
                'tremor_amplitude': 0.15,
                'asymmetry_factor': 1.3,
                'noise_level': 0.18,
                'spasticity_pattern': True
            }
        }
    
    def generate_dataset(self,
                        n_subjects: int = 100,
                        diseases: List[str] = None,
                        duration_range: Tuple[int, int] = (60, 120),
                        sampling_rate: int = 50,
                        include_demographics: bool = True) -> pd.DataFrame:
        """
        Generate a complete synthetic dataset with multiple subjects
        
        Args:
            n_subjects: Number of subjects to generate
            diseases: List of diseases to include
            duration_range: Recording duration range in seconds
            sampling_rate: Sampling frequency in Hz
            include_demographics: Whether to include demographic data
            
        Returns:
            Complete dataset with all subjects
        """
        
        if diseases is None:
            diseases = list(self.disease_parameters.keys())
        
        logger.info(f"Generating dataset with {n_subjects} subjects, diseases: {diseases}")
        
        all_data = []
        
        for subject_id in range(1, n_subjects + 1):
            # Random disease assignment
            disease = np.random.choice(diseases)
            
            # Random duration within range
            duration = np.random.randint(duration_range[0], duration_range[1] + 1)
            
            # Generate subject data
            subject_data = self.generate_subject_data(
                subject_id=f"S{subject_id:03d}",
                disease=disease,
                duration=duration,
                sampling_rate=sampling_rate,
                include_demographics=include_demographics
            )
            
            all_data.append(subject_data)
            
            if subject_id % 20 == 0:
                logger.info(f"Generated {subject_id}/{n_subjects} subjects")
        
        # Combine all subjects
        complete_dataset = pd.concat(all_data, ignore_index=True)
        
        logger.info(f"Dataset generation complete. Total samples: {len(complete_dataset)}")
        
        return complete_dataset
    
    def generate_subject_data(self,
                             subject_id: str,
                             disease: str = 'Normal',
                             duration: int = 60,
                             sampling_rate: int = 50,
                             include_demographics: bool = True) -> pd.DataFrame:
        """
        Generate data for a single subject
        
        Args:
            subject_id: Unique subject identifier
            disease: Disease condition
            duration: Recording duration in seconds
            sampling_rate: Sampling frequency in Hz
            include_demographics: Include demographic information
            
        Returns:
            DataFrame with subject's sensor data
        """
        
        n_samples = duration * sampling_rate
        time_vector = np.linspace(0, duration, n_samples)
        
        # Get disease parameters
        if disease not in self.disease_parameters:
            logger.warning(f"Unknown disease '{disease}', using Normal parameters")
            disease = 'Normal'
        
        params = self.disease_parameters[disease]
        
        # Generate base gait pattern
        gait_data = self._generate_gait_pattern(time_vector, params)
        
        # Add disease-specific modifications
        if disease == 'Parkinson':
            gait_data = self._add_parkinson_features(gait_data, time_vector, params)
        elif disease == 'Huntington':
            gait_data = self._add_huntington_features(gait_data, time_vector, params)
        elif disease == 'Ataxia':
            gait_data = self._add_ataxia_features(gait_data, time_vector, params)
        elif disease == 'MultipleSclerosis':
            gait_data = self._add_ms_features(gait_data, time_vector, params)
        
        # Create DataFrame
        timestamps = [datetime.now() + timedelta(seconds=t) for t in time_vector]
        
        df_data = {
            'timestamp': timestamps,
            'subject_id': [subject_id] * n_samples,
            'disease': [disease] * n_samples,
            'accel_x': gait_data['accel_x'],
            'accel_y': gait_data['accel_y'],
            'accel_z': gait_data['accel_z'],
            'gyro_x': gait_data['gyro_x'],
            'gyro_y': gait_data['gyro_y'],
            'gyro_z': gait_data['gyro_z']
        }
        
        # Add EMG if requested
        if np.random.random() > 0.3:  # 70% chance of having EMG
            df_data['emg_signal'] = self._generate_emg_signal(time_vector, params)
        
        # Add demographics
        if include_demographics:
            demographics = self._generate_demographics(disease)
            for key, value in demographics.items():
                df_data[key] = [value] * n_samples
        
        return pd.DataFrame(df_data)
    
    def _generate_gait_pattern(self, time_vector: np.ndarray, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate base gait pattern from accelerometer and gyroscope"""
        
        # Base step frequency
        step_freq_range = params['step_frequency']
        base_step_freq = np.random.uniform(step_freq_range[0], step_freq_range[1])
        
        # Step frequency variation over time
        freq_variation = params['step_variability']
        step_freq = base_step_freq + freq_variation * np.sin(0.1 * time_vector) * np.random.randn(len(time_vector)) * 0.1
        
        # Acceleration amplitudes
        accel_amp_range = params['acceleration_amplitude']
        accel_amplitude = np.random.uniform(accel_amp_range[0], accel_amp_range[1])
        
        # Generate accelerometer signals
        accel_x = accel_amplitude * np.sin(2 * np.pi * step_freq * time_vector + np.pi/4)
        accel_y = accel_amplitude * 0.6 * np.sin(4 * np.pi * step_freq * time_vector)
        accel_z = 9.81 + accel_amplitude * np.sin(2 * np.pi * step_freq * time_vector)  # Include gravity
        
        # Gyroscope amplitudes
        gyro_amp_range = params['gyroscope_amplitude']
        gyro_amplitude = np.random.uniform(gyro_amp_range[0], gyro_amp_range[1])
        
        # Generate gyroscope signals
        gyro_x = gyro_amplitude * np.sin(2 * np.pi * step_freq * time_vector + np.pi/2)
        gyro_y = gyro_amplitude * 0.4 * np.sin(2 * np.pi * step_freq * time_vector)
        gyro_z = gyro_amplitude * 0.3 * np.sin(4 * np.pi * step_freq * time_vector + np.pi/3)
        
        # Add asymmetry
        asymmetry = params.get('asymmetry_factor', 1.0)
        if asymmetry != 1.0:
            # Introduce left-right asymmetry
            asymmetry_pattern = 0.5 + 0.5 * np.sin(np.pi * step_freq * time_vector)
            accel_y *= (1 + 0.2 * (asymmetry - 1) * asymmetry_pattern)
            gyro_z *= (1 + 0.15 * (asymmetry - 1) * asymmetry_pattern)
        
        # Add noise
        noise_level = params['noise_level']
        accel_x += np.random.normal(0, noise_level, len(time_vector))
        accel_y += np.random.normal(0, noise_level, len(time_vector))
        accel_z += np.random.normal(0, noise_level, len(time_vector))
        gyro_x += np.random.normal(0, noise_level * 0.5, len(time_vector))
        gyro_y += np.random.normal(0, noise_level * 0.5, len(time_vector))
        gyro_z += np.random.normal(0, noise_level * 0.5, len(time_vector))
        
        return {
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z
        }
    
    def _add_parkinson_features(self, gait_data: Dict[str, np.ndarray], 
                               time_vector: np.ndarray, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Add Parkinson's disease specific features"""
        
        # Add 4-6 Hz tremor
        if params.get('tremor_frequency'):
            tremor_freq_range = params['tremor_frequency']
            tremor_freq = np.random.uniform(tremor_freq_range[0], tremor_freq_range[1])
            tremor_amp = params['tremor_amplitude']
            
            tremor_signal = tremor_amp * np.sin(2 * np.pi * tremor_freq * time_vector)
            
            # Add tremor mainly to x and y axes (resting tremor)
            gait_data['accel_x'] += tremor_signal
            gait_data['accel_y'] += tremor_signal * 0.8
            gait_data['gyro_z'] += tremor_signal * 0.3
        
        # Add freezing episodes
        if params.get('freezing_episodes'):
            n_episodes = np.random.poisson(2)  # Average 2 episodes
            for _ in range(n_episodes):
                freeze_start = np.random.randint(0, len(time_vector) - 100)
                freeze_duration = np.random.randint(20, 100)  # 0.4-2 seconds
                freeze_end = min(freeze_start + freeze_duration, len(time_vector))
                
                # Reduce movement during freezing
                gait_data['accel_x'][freeze_start:freeze_end] *= 0.1
                gait_data['accel_y'][freeze_start:freeze_end] *= 0.1
                gait_data['gyro_x'][freeze_start:freeze_end] *= 0.1
                gait_data['gyro_y'][freeze_start:freeze_end] *= 0.1
        
        # Reduce overall amplitude (bradykinesia)
        for key in gait_data:
            if key != 'accel_z':  # Don't reduce gravity component
                gait_data[key] *= 0.8
        
        return gait_data
    
    def _add_huntington_features(self, gait_data: Dict[str, np.ndarray], 
                                time_vector: np.ndarray, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Add Huntington's disease specific features (chorea)"""
        
        # Add choreic movements (irregular, involuntary)
        if params.get('chorea_pattern'):
            # Generate irregular bursts of activity
            n_bursts = np.random.poisson(10)  # Average 10 bursts
            
            for _ in range(n_bursts):
                burst_start = np.random.randint(0, len(time_vector) - 50)
                burst_duration = np.random.randint(10, 50)
                burst_end = min(burst_start + burst_duration, len(time_vector))
                
                # Add sudden irregular movements
                burst_amplitude = np.random.uniform(1.0, 3.0)
                burst_freq = np.random.uniform(2, 8)
                
                burst_signal = burst_amplitude * np.sin(2 * np.pi * burst_freq * time_vector[burst_start:burst_end])
                
                gait_data['accel_x'][burst_start:burst_end] += burst_signal
                gait_data['accel_y'][burst_start:burst_end] += burst_signal * 0.7
                gait_data['gyro_x'][burst_start:burst_end] += burst_signal * 0.5
        
        # Add high frequency variability
        variability = np.random.normal(0, 0.5, len(time_vector))
        for key in gait_data:
            gait_data[key] += variability * 0.3
        
        return gait_data
    
    def _add_ataxia_features(self, gait_data: Dict[str, np.ndarray], 
                            time_vector: np.ndarray, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Add ataxia specific features (coordination impairment)"""
        
        if params.get('coordination_impairment'):
            # Add coordination errors (phase delays between axes)
            phase_error = np.random.uniform(0, np.pi/2)
            
            # Introduce timing delays
            gait_data['accel_y'] = np.roll(gait_data['accel_y'], int(phase_error * 10))
            gait_data['gyro_x'] = np.roll(gait_data['gyro_x'], int(phase_error * 15))
            
            # Add irregular stepping patterns
            irregularity = 0.3 * np.random.randn(len(time_vector))
            gait_data['accel_x'] += irregularity
            gait_data['accel_z'] += irregularity * 0.5
        
        # Add low-frequency tremor during movement
        if params.get('tremor_frequency'):
            tremor_freq = np.random.uniform(params['tremor_frequency'][0], params['tremor_frequency'][1])
            tremor_amp = params['tremor_amplitude']
            
            # Intention tremor increases with movement
            movement_intensity = np.abs(gait_data['accel_x']) + np.abs(gait_data['accel_y'])
            normalized_intensity = movement_intensity / np.max(movement_intensity)
            
            tremor_signal = tremor_amp * normalized_intensity * np.sin(2 * np.pi * tremor_freq * time_vector)
            
            gait_data['accel_x'] += tremor_signal
            gait_data['accel_y'] += tremor_signal * 0.8
        
        return gait_data
    
    def _add_ms_features(self, gait_data: Dict[str, np.ndarray], 
                        time_vector: np.ndarray, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Add Multiple Sclerosis specific features"""
        
        if params.get('spasticity_pattern'):
            # Add spasticity episodes (sudden stiffness)
            n_episodes = np.random.poisson(3)  # Average 3 episodes
            
            for _ in range(n_episodes):
                spasm_start = np.random.randint(0, len(time_vector) - 80)
                spasm_duration = np.random.randint(30, 80)
                spasm_end = min(spasm_start + spasm_duration, len(time_vector))
                
                # During spasticity, movements become rigid and jerky
                spasm_factor = np.linspace(1.0, 2.0, spasm_end - spasm_start)
                
                gait_data['accel_x'][spasm_start:spasm_end] *= spasm_factor
                gait_data['gyro_x'][spasm_start:spasm_end] *= spasm_factor * 0.5
                
                # Add jerkiness
                jerk = np.random.uniform(-0.5, 0.5, spasm_end - spasm_start)
                gait_data['accel_y'][spasm_start:spasm_end] += jerk
        
        # Add fatigue effect (gradual amplitude reduction)
        fatigue_factor = np.linspace(1.0, 0.7, len(time_vector))
        for key in gait_data:
            if key != 'accel_z':  # Preserve gravity
                gait_data[key] *= fatigue_factor
        
        # Add intention tremor
        if params.get('tremor_frequency'):
            tremor_freq = np.random.uniform(params['tremor_frequency'][0], params['tremor_frequency'][1])
            tremor_amp = params['tremor_amplitude']
            
            # Tremor that appears during movement
            movement_mask = (np.abs(gait_data['accel_x']) > np.percentile(np.abs(gait_data['accel_x']), 70))
            tremor_signal = tremor_amp * np.sin(2 * np.pi * tremor_freq * time_vector)
            tremor_signal *= movement_mask.astype(float)
            
            gait_data['accel_x'] += tremor_signal
            gait_data['accel_y'] += tremor_signal * 0.6
        
        return gait_data
    
    def _generate_emg_signal(self, time_vector: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Generate synthetic EMG signal"""
        
        # Base EMG pattern related to gait
        step_freq_range = params['step_frequency']
        step_freq = np.random.uniform(step_freq_range[0], step_freq_range[1])
        
        # EMG follows muscle activation pattern
        base_emg = 50 + 30 * np.abs(np.sin(2 * np.pi * step_freq * time_vector))
        
        # Add muscle fatigue over time
        fatigue_factor = np.linspace(1.0, 1.2, len(time_vector))  # Slight increase over time
        base_emg *= fatigue_factor
        
        # Add noise (EMG is inherently noisy)
        noise_level = params.get('noise_level', 0.1)
        emg_noise = np.random.exponential(scale=10, size=len(time_vector))  # Exponential noise
        
        emg_signal = base_emg + emg_noise * noise_level * 50
        
        # Ensure positive values (rectified EMG)
        emg_signal = np.abs(emg_signal)
        
        # Add disease-specific EMG modifications
        disease = params.get('disease', 'Normal')
        
        if 'tremor_frequency' in params and params['tremor_frequency']:
            # Add tremor to EMG
            tremor_freq = np.random.uniform(params['tremor_frequency'][0], params['tremor_frequency'][1])
            tremor_emg = 15 * np.abs(np.sin(2 * np.pi * tremor_freq * time_vector))
            emg_signal += tremor_emg
        
        return emg_signal
    
    def _generate_demographics(self, disease: str) -> Dict[str, Any]:
        """Generate realistic demographic data"""
        
        # Age distributions vary by disease
        age_ranges = {
            'Normal': (20, 80),
            'Parkinson': (55, 85),
            'Huntington': (30, 65),
            'Ataxia': (25, 70),
            'MultipleSclerosis': (20, 60)
        }
        
        age_range = age_ranges.get(disease, (20, 80))
        age = np.random.randint(age_range[0], age_range[1])
        
        # Gender distribution (slightly more males in some neurological conditions)
        gender_probs = {
            'Normal': [0.5, 0.5],
            'Parkinson': [0.6, 0.4],  # Slightly more males
            'Huntington': [0.5, 0.5],
            'Ataxia': [0.55, 0.45],
            'MultipleSclerosis': [0.3, 0.7]  # More females
        }
        
        gender = np.random.choice(['M', 'F'], p=gender_probs.get(disease, [0.5, 0.5]))
        
        # Height and weight (correlated)
        if gender == 'M':
            height = np.random.normal(175, 8)  # cm
            weight = np.random.normal(80, 12)  # kg
        else:
            height = np.random.normal(162, 7)  # cm
            weight = np.random.normal(65, 10)  # kg
        
        # BMI
        bmi = weight / ((height / 100) ** 2)
        
        # Disease severity (for pathological conditions)
        severity = None
        if disease != 'Normal':
            severity = np.random.choice(['Mild', 'Moderate', 'Severe'], p=[0.4, 0.4, 0.2])
        
        # Disease duration (for pathological conditions)
        duration = None
        if disease != 'Normal':
            duration_ranges = {
                'Parkinson': (1, 15),
                'Huntington': (2, 20),
                'Ataxia': (1, 25),
                'MultipleSclerosis': (1, 30)
            }
            duration_range = duration_ranges.get(disease, (1, 10))
            duration = np.random.randint(duration_range[0], duration_range[1])
        
        # Medication status
        on_medication = False
        if disease != 'Normal' and np.random.random() > 0.3:  # 70% on medication
            on_medication = True
        
        demographics = {
            'age': int(age),
            'gender': gender,
            'height_cm': round(height, 1),
            'weight_kg': round(weight, 1),
            'bmi': round(bmi, 1),
            'disease_severity': severity,
            'disease_duration_years': duration,
            'on_medication': on_medication
        }
        
        return demographics
    
    def generate_balanced_dataset(self,
                                 total_subjects: int = 500,
                                 test_split: float = 0.2,
                                 validation_split: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate a balanced dataset with train/validation/test splits
        
        Args:
            total_subjects: Total number of subjects
            test_split: Proportion for test set
            validation_split: Proportion for validation set (from remaining data)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        
        diseases = list(self.disease_parameters.keys())
        subjects_per_disease = total_subjects // len(diseases)
        
        logger.info(f"Generating balanced dataset: {subjects_per_disease} subjects per disease")
        
        # Generate data for each disease
        all_subjects = []
        
        for disease in diseases:
            for i in range(subjects_per_disease):
                subject_data = self.generate_subject_data(
                    subject_id=f"{disease}_{i+1:03d}",
                    disease=disease,
                    duration=np.random.randint(60, 120),
                    sampling_rate=50,
                    include_demographics=True
                )
                all_subjects.append(subject_data)
        
        # Combine all data
        complete_data = pd.concat(all_subjects, ignore_index=True)
        
        # Split by subjects (not by samples)
        unique_subjects = complete_data['subject_id'].unique()
        np.random.shuffle(unique_subjects)
        
        n_test = int(len(unique_subjects) * test_split)
        n_val = int(len(unique_subjects) * validation_split)
        
        test_subjects = unique_subjects[:n_test]
        val_subjects = unique_subjects[n_test:n_test + n_val]
        train_subjects = unique_subjects[n_test + n_val:]
        
        # Create splits
        train_df = complete_data[complete_data['subject_id'].isin(train_subjects)].reset_index(drop=True)
        val_df = complete_data[complete_data['subject_id'].isin(val_subjects)].reset_index(drop=True)
        test_df = complete_data[complete_data['subject_id'].isin(test_subjects)].reset_index(drop=True)
        
        logger.info(f"Dataset splits created:")
        logger.info(f"  Train: {len(train_subjects)} subjects, {len(train_df)} samples")
        logger.info(f"  Validation: {len(val_subjects)} subjects, {len(val_df)} samples") 
        logger.info(f"  Test: {len(test_subjects)} subjects, {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def add_realistic_artifacts(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add realistic measurement artifacts and noise"""
        
        data_copy = data.copy()
        sensor_cols = [col for col in data.columns if any(sensor in col for sensor in ['accel', 'gyro', 'emg'])]
        
        for col in sensor_cols:
            # Add sensor drift
            if np.random.random() > 0.8:  # 20% chance of drift
                drift_rate = np.random.uniform(-0.001, 0.001)
                time_idx = np.arange(len(data))
                drift = drift_rate * time_idx
                data_copy[col] += drift
            
            # Add occasional spikes (sensor glitches)
            if np.random.random() > 0.9:  # 10% chance of spikes
                n_spikes = np.random.randint(1, 5)
                spike_indices = np.random.choice(len(data), n_spikes, replace=False)
                spike_amplitude = np.random.uniform(5, 20)
                data_copy.loc[spike_indices, col] += spike_amplitude * np.random.choice([-1, 1], n_spikes)
            
            # Add missing data segments
            if np.random.random() > 0.95:  # 5% chance of missing segments
                missing_start = np.random.randint(0, len(data) - 50)
                missing_length = np.random.randint(10, 50)
                missing_end = min(missing_start + missing_length, len(data))
                data_copy.loc[missing_start:missing_end, col] = np.nan
        
        return data_copy

# Example usage and testing
if __name__ == "__main__":
    # Initialize generator
    generator = SyntheticDataGenerator(random_seed=42)
    
    # Generate single subject data
    subject_data = generator.generate_subject_data(
        subject_id="TEST001",
        disease="Parkinson",
        duration=60,
        sampling_rate=50
    )
    
    print(f"Generated subject data shape: {subject_data.shape}")
    print(f"Columns: {list(subject_data.columns)}")
    print(f"Disease distribution:\n{subject_data['disease'].value_counts()}")
    
    # Generate balanced dataset
    train_df, val_df, test_df = generator.generate_balanced_dataset(
        total_subjects=50,  # Small for testing
        test_split=0.2,
        validation_split=0.2
    )
    
    print(f"\nDataset shapes:")
    print(f"Train: {train_df.shape}")
    print(f"Validation: {val_df.shape}")
    print(f"Test: {test_df.shape}")
    
    # Show disease distribution
    print(f"\nDisease distribution in training set:")
    print(train_df.groupby('subject_id')['disease'].first().value_counts())
                