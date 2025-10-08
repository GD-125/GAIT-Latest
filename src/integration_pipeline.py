# File: src/integration_pipeline.py
# Complete integration pipeline from raw data to predictions

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

from src.preprocessing.signal_processor import SignalProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.preprocessing.normalizer import DataNormalizer
from src.preprocessing.segmentation import SignalSegmenter
from src.models.gait_detector import GaitDetector
from src.models.disease_classifier import DiseaseClassifier

logger = logging.getLogger(__name__)


class FEAIPipeline:
    """
    Complete FE-AI analysis pipeline
    Raw Data → Preprocessing → Gait Detection → Disease Classification
    """
    
    def __init__(self,
                 gait_model_path: Optional[str] = None,
                 disease_model_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        # Initialize preprocessing modules
        self.signal_processor = SignalProcessor()
        self.feature_extractor = FeatureExtractor()
        self.normalizer = DataNormalizer()
        self.segmenter = SignalSegmenter()
        
        # Initialize models
        self.gait_detector = GaitDetector(model_path=gait_model_path, device=device)
        self.disease_classifier = DiseaseClassifier(model_path=disease_model_path, device=device)
        
        logger.info("FE-AI Pipeline initialized successfully")
    
    def process_raw_data(self,
                        data: pd.DataFrame,
                        sampling_rate: int = 50,
                        denoise: bool = True,
                        normalize: bool = True) -> pd.DataFrame:
        """
        Preprocess raw sensor data
        
        Args:
            data: Raw sensor DataFrame with columns [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            sampling_rate: Sampling frequency in Hz
            denoise: Apply denoising filters
            normalize: Apply normalization
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            logger.info("Starting data preprocessing...")
            
            # Process signal
            processed_result = self.signal_processor.process_data(
                data=data,
                denoise=denoise,
                normalize=normalize,
                sampling_rate=sampling_rate
            )
            
            processed_data = processed_result['processed_data']
            
            logger.info(f"Preprocessing complete: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return data
    
    def detect_gait(self,
                   data: pd.DataFrame,
                   confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Detect gait patterns in preprocessed data
        
        Args:
            data: Preprocessed sensor data
            confidence_threshold: Confidence threshold for gait detection
            
        Returns:
            Gait detection results
        """
        try:
            logger.info("Running gait detection...")
            
            gait_results = self.gait_detector.predict(
                data=data,
                confidence_threshold=confidence_threshold
            )
            
            logger.info(f"Gait detection complete: {gait_results['gait_segments']} gait segments found")
            return gait_results
            
        except Exception as e:
            logger.error(f"Error in gait detection: {str(e)}")
            return {
                'error': str(e),
                'gait_segments': 0,
                'non_gait_segments': 0
            }
    
    def classify_disease(self,
                        gait_results: Dict[str, Any],
                        diseases: Optional[list] = None,
                        ensemble_weight: float = 0.7) -> Dict[str, Any]:
        """
        Classify neurological disease from gait patterns
        
        Args:
            gait_results: Output from gait detection
            diseases: List of diseases to classify (None for all)
            ensemble_weight: Weight for transformer model in ensemble
            
        Returns:
            Disease classification results
        """
        try:
            logger.info("Running disease classification...")
            
            disease_results = self.disease_classifier.predict(
                gait_predictions=gait_results,
                diseases=diseases,
                ensemble_weight=ensemble_weight
            )
            
            logger.info(f"Disease classification complete: {disease_results['top_prediction']['disease']}")
            return disease_results
            
        except Exception as e:
            logger.error(f"Error in disease classification: {str(e)}")
            return {
                'error': str(e),
                'top_prediction': {'disease': 'Unknown', 'confidence': 0.0}
            }
    
    def analyze_complete(self,
                        data: pd.DataFrame,
                        sampling_rate: int = 50,
                        confidence_threshold: float = 0.8,
                        diseases: Optional[list] = None) -> Dict[str, Any]:
        """
        Complete analysis pipeline from raw data to disease prediction
        
        Args:
            data: Raw sensor DataFrame
            sampling_rate: Sampling frequency
            confidence_threshold: Gait detection threshold
            diseases: Diseases to classify
            
        Returns:
            Complete analysis results
        """
        logger.info("=" * 60)
        logger.info("Starting Complete FE-AI Analysis Pipeline")
        logger.info("=" * 60)
        
        results = {
            'preprocessing': {},
            'gait_detection': {},
            'disease_classification': {},
            'status': 'running'
        }
        
        try:
            # Step 1: Preprocess data
            processed_data = self.process_raw_data(
                data=data,
                sampling_rate=sampling_rate,
                denoise=True,
                normalize=True
            )
            results['preprocessing'] = {
                'status': 'completed',
                'shape': processed_data.shape,
                'columns': list(processed_data.columns)
            }
            
            # Step 2: Detect gait
            gait_results = self.detect_gait(
                data=processed_data,
                confidence_threshold=confidence_threshold
            )
            results['gait_detection'] = gait_results
            
            # Step 3: Classify disease
            disease_results = self.classify_disease(
                gait_results=gait_results,
                diseases=diseases
            )
            results['disease_classification'] = disease_results
            
            results['status'] = 'completed'
            logger.info("=" * 60)
            logger.info("FE-AI Analysis Pipeline Completed Successfully")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {str(e)}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable analysis report
        
        Args:
            results: Complete analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "=" * 70)
        report.append("FE-AI GAIT ANALYSIS AND DISEASE DETECTION REPORT")
        report.append("=" * 70 + "\n")
        
        # Preprocessing Summary
        if 'preprocessing' in results:
            report.append("1. DATA PREPROCESSING")
            report.append("-" * 70)
            prep = results['preprocessing']
            report.append(f"   Status: {prep.get('status', 'N/A')}")
            report.append(f"   Data Shape: {prep.get('shape', 'N/A')}")
            report.append("")
        
        # Gait Detection Summary
        if 'gait_detection' in results:
            report.append("2. GAIT DETECTION RESULTS")
            report.append("-" * 70)
            gait = results['gait_detection']
            report.append(f"   Gait Segments: {gait.get('gait_segments', 0)}")
            report.append(f"   Non-Gait Segments: {gait.get('non_gait_segments', 0)}")
            report.append(f"   Average Confidence: {gait.get('avg_confidence', 0):.2%}")
            report.append(f"   Processing Time: {gait.get('processing_time', 0):.2f}s")
            report.append("")
        
        # Disease Classification Summary
        if 'disease_classification' in results:
            report.append("3. DISEASE CLASSIFICATION RESULTS")
            report.append("-" * 70)
            disease = results['disease_classification']
            
            if 'top_prediction' in disease:
                top = disease['top_prediction']
                report.append(f"   Top Prediction: {top['disease']}")
                report.append(f"   Confidence: {top['confidence']:.2%}")
                report.append("")
            
            if 'all_predictions' in disease:
                report.append("   All Disease Probabilities:")
                for disease_name, prob in disease['all_predictions'].items():
                    report.append(f"      {disease_name:30s}: {prob:.2%}")
                report.append("")
            
            report.append(f"   Processing Time: {disease.get('processing_time', 0):.2f}s")
        
        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    import torch
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'accel_x': np.random.randn(1000),
        'accel_y': np.random.randn(1000),
        'accel_z': 9.8 + np.random.randn(1000) * 0.5,
        'gyro_x': np.random.randn(1000) * 0.1,
        'gyro_y': np.random.randn(1000) * 0.1,
        'gyro_z': np.random.randn(1000) * 0.1,
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='20ms')
    })
    
    # Initialize pipeline
    pipeline = FEAIPipeline()
    
    # Run complete analysis
    results = pipeline.analyze_complete(
        data=sample_data,
        sampling_rate=50,
        confidence_threshold=0.8
    )
    
    # Generate report
    report = pipeline.generate_report(results)
    print(report)