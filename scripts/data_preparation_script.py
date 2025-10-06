"""
File: scripts/data_preparation.py
Data preparation and preprocessing pipeline
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

from src.data.data_loader import DataLoader
from src.data.data_validator import DataValidator
from src.preprocessing.signal_processor import SignalProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.normalizer import Normalizer
from src.preprocessing.segmentation import WindowSegmenter
from src.utils.logger import setup_logger

logger = setup_logger()


class DataPreparationPipeline:
    """
    Comprehensive data preparation pipeline
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.validator = DataValidator()
        self.signal_processor = SignalProcessor()
        self.feature_extractor = FeatureExtractor()
        self.normalizer = Normalizer(method=config.get('normalization', 'zscore'))
        self.segmenter = WindowSegmenter(
            window_size=config.get('window_size', 100),
            step_size=config.get('step_size', 50)
        )
        
        self.statistics = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'total_samples': 0,
            'total_windows': 0,
            'errors': []
        }
    
    def prepare_dataset(
        self,
        input_dir: str,
        output_dir: str,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ):
        """
        Prepare complete dataset with train/val/test splits
        
        Args:
            input_dir: Directory containing raw data files
            output_dir: Directory to save prepared data
            train_split: Proportion of training data
            val_split: Proportion of validation data
            test_split: Proportion of test data
        """
        logger.info("Starting data preparation pipeline...")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all data files
        data_files = list(input_path.glob('*.csv')) + list(input_path.glob('*.xlsx'))
        self.statistics['total_files'] = len(data_files)
        
        logger.info(f"Found {len(data_files)} data files")
        
        # Process all files
        all_windows = []
        all_labels = []
        
        for file_path in data_files:
            try:
                logger.info(f"Processing {file_path.name}...")
                
                # Load data
                df = self._load_file(file_path)
                
                # Validate
                validation = self.validator.validate_dataframe(df)
                if not validation['valid']:
                    logger.warning(f"Invalid data in {file_path.name}: {validation['errors']}")
                    self.statistics['invalid_files'] += 1
                    self.statistics['errors'].append({
                        'file': str(file_path),
                        'errors': validation['errors']
                    })
                    continue
                
                # Process
                processed = self._process_file(df)
                
                # Segment into windows
                windows, labels = self._segment_data(processed, df)
                
                all_windows.extend(windows)
                all_labels.extend(labels)
                
                self.statistics['valid_files'] += 1
                self.statistics['total_samples'] += len(df)
                self.statistics['total_windows'] += len(windows)
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                self.statistics['invalid_files'] += 1
                self.statistics['errors'].append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        # Convert to arrays
        X = np.array(all_windows)
        y = np.array(all_labels)
        
        logger.info(f"Total windows created: {len(X)}")
        logger.info(f"Data shape: {X.shape}")
        
        # Normalize
        logger.info("Normalizing data...")
        X_normalized = self._normalize_data(X)
        
        # Split data
        logger.info("Splitting into train/val/test sets...")
        splits = self._split_data(X_normalized, y, train_split, val_split, test_split)
        
        # Save splits
        self._save_splits(splits, output_path)
        
        # Save statistics
        self._save_statistics(output_path)
        
        logger.info("Data preparation completed successfully!")
        
        return self.statistics
    
    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Load data file"""
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _process_file(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process single data file"""
        # Signal processing
        processed = self.signal_processor.process(df)
        
        # Handle missing values
        processed = processed.fillna(method='ffill').fillna(method='bfill')
        
        return processed
    
    def _segment_data(
        self,
        df: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Segment data into windows"""
        # Get sensor columns
        sensor_cols = [
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'emg_1', 'emg_2', 'emg_3'
        ]
        
        # Extract sensor data
        sensor_data = df[sensor_cols].values
        
        # Segment
        windows = self.segmenter.segment(sensor_data)
        
        # Get labels if available
        labels = []
        if 'label' in original_df.columns:
            # For each window, take majority label
            for i in range(len(windows)):
                start_idx = i * self.segmenter.step_size
                end_idx = start_idx + self.segmenter.window_size
                window_labels = original_df['label'].iloc[start_idx:end_idx]
                
                # Convert string labels to numeric
                if window_labels.dtype == 'object':
                    label = 1 if window_labels.mode()[0] == 'gait' else 0
                else:
                    label = int(window_labels.mode()[0])
                
                labels.append(label)
        else:
            # No labels available, assign default
            labels = [0] * len(windows)
        
        return windows, labels
    
    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """Normalize data"""
        # Reshape for normalization
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Fit and transform
        X_normalized = self.normalizer.fit_transform(X_reshaped)
        
        # Reshape back
        X_normalized = X_normalized.reshape(original_shape)
        
        # Save normalization parameters
        self.normalizer.save_params('data/processed/normalization_params.json')
        
        return X_normalized
    
    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_split: float,
        val_split: float,
        test_split: float
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data into train/val/test sets"""
        # Ensure splits sum to 1
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6
        
        # Shuffle indices
        indices = np.random.permutation(len(X))
        
        # Calculate split points
        train_end = int(len(X) * train_split)
        val_end = train_end + int(len(X) * val_split)
        
        # Split
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        splits = {
            'train': (X[train_indices], y[train_indices]),
            'val': (X[val_indices], y[val_indices]),
            'test': (X[test_indices], y[test_indices])
        }
        
        logger.info(f"Train set: {len(train_indices)} samples")
        logger.info(f"Validation set: {len(val_indices)} samples")
        logger.info(f"Test set: {len(test_indices)} samples")
        
        return splits
    
    def _save_splits(self, splits: Dict, output_path: Path):
        """Save data splits"""
        for split_name, (X, y) in splits.items():
            split_dir = output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as numpy arrays
            np.save(split_dir / 'X.npy', X)
            np.save(split_dir / 'y.npy', y)
            
            logger.info(f"Saved {split_name} split to {split_dir}")
            
            # Save as CSV for inspection
            df = pd.DataFrame({
                'label': y,
                'sample_id': range(len(y))
            })
            df.to_csv(split_dir / 'labels.csv', index=False)
    
    def _save_statistics(self, output_path: Path):
        """Save processing statistics"""
        stats_path = output_path / 'preparation_stats.json'
        
        with open(stats_path, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("DATA PREPARATION SUMMARY")
        print("="*50)
        print(f"Total files processed: {self.statistics['total_files']}")
        print(f"Valid files: {self.statistics['valid_files']}")
        print(f"Invalid files: {self.statistics['invalid_files']}")
        print(f"Total samples: {self.statistics['total_samples']}")
        print(f"Total windows: {self.statistics['total_windows']}")
        print("="*50 + "\n")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing raw data files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save prepared data')
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Proportion of training data')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Proportion of validation data')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Proportion of test data')
    parser.add_argument('--window_size', type=int, default=100,
                       help='Window size for segmentation')
    parser.add_argument('--step_size', type=int, default=50,
                       help='Step size for segmentation')
    parser.add_argument('--normalization', type=str, default='zscore',
                       choices=['zscore', 'minmax', 'robust'],
                       help='Normalization method')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'window_size': args.window_size,
        'step_size': args.step_size,
        'normalization': args.normalization
    }
    
    # Initialize pipeline
    pipeline = DataPreparationPipeline(config)
    
    # Prepare dataset
    pipeline.prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )


if __name__ == "__main__":
    main()
