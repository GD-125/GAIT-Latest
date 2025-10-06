# File: scripts/train_models.py
# Model training script

#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.gait_detector import GaitDetector
from src.models.disease_classifier import DiseaseClassifier
from src.preprocessing.signal_processor import SignalProcessor
from src.utils.logger import setup_logger
import logging

def main():
    """Train FE-AI models"""
    
    parser = argparse.ArgumentParser(description="Train FE-AI Models")
    parser.add_argument("--model", required=True,
                       choices=["gait_detector", "disease_classifier", "both"],
                       help="Model to train")
    parser.add_argument("--train-data", required=True,
                       help="Training data file path")
    parser.add_argument("--val-data",
                       help="Validation data file path")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--save-path", default="data/models/",
                       help="Path to save trained models")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("ModelTraining")
    logger.info(f"Starting model training: {args.model}")
    
    try:
        # Load training data
        logger.info(f"Loading training data from {args.train_data}")
        train_df = pd.read_csv(args.train_data)
        
        val_df = None
        if args.val_data:
            logger.info(f"Loading validation data from {args.val_data}")
            val_df = pd.read_csv(args.val_data)
        
        # Initialize signal processor
        processor = SignalProcessor()
        
        # Process training data
        logger.info("Processing training data...")
        train_processed = processor.process_data(
            train_df,
            denoise=True,
            normalize=True,
            segment=True,
            extract_features=True,
            window_size=5,
            overlap=25
        )
        
        val_processed = None
        if val_df is not None:
            logger.info("Processing validation data...")
            val_processed = processor.process_data(
                val_df,
                denoise=True,
                normalize=True,
                segment=True,
                extract_features=True,
                window_size=5,
                overlap=25
            )
        
        # Train models
        save_path = Path(args.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if args.model in ["gait_detector", "both"]:
            logger.info("Training gait detector...")
            train_gait_detector(
                train_processed, val_processed, 
                args, save_path, logger
            )
        
        if args.model in ["disease_classifier", "both"]:
            logger.info("Training disease classifier...")
            train_disease_classifier(
                train_processed, val_processed,
                args, save_path, logger
            )
        
        logger.info("Model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return False

def train_gait_detector(train_processed, val_processed, args, save_path, logger):
    """Train gait detection model"""
    
    # Prepare gait detection data
    X_train = np.array(train_processed.get('segments', []))
    y_train = np.random.choice([0, 1], len(X_train), p=[0.3, 0.7])  # Mock labels
    
    X_val, y_val = None, None
    if val_processed:
        X_val = np.array(val_processed.get('segments', []))
        y_val = np.random.choice([0, 1], len(X_val), p=[0.3, 0.7])
    
    # Initialize and train model
    detector = GaitDetector()
    
    history = detector.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save model
    model_path = save_path / "gait_detector_v2.1.pth"
    detector.save_model(str(model_path))
    logger.info(f"Gait detector saved to {model_path}")
    
    # Evaluate model
    if X_val is not None and y_val is not None:
        metrics = detector.evaluate(X_val, y_val)
        logger.info(f"Gait detector performance: {metrics}")

def train_disease_classifier(train_processed, val_processed, args, save_path, logger):
    """Train disease classification model"""
    
    # Prepare disease classification data
    features_df = train_processed.get('features', pd.DataFrame())
    X_train = features_df.values
    diseases = ['Parkinson', 'Huntington', 'Ataxia', 'MS', 'Normal']
    y_train = np.random.choice(diseases, len(X_train))
    
    X_val, y_val = None, None
    if val_processed:
        val_features = val_processed.get('features', pd.DataFrame())
        X_val = val_features.values
        y_val = np.random.choice(diseases, len(X_val))
    
    # Initialize and train model
    classifier = DiseaseClassifier()
    
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save model
    model_path = save_path / "disease_classifier_v2.1.pkl"
    classifier.save_model(str(model_path))
    logger.info(f"Disease classifier saved to {model_path}")
    
    # Evaluate model
    if X_val is not None and y_val is not None:
        metrics = classifier.evaluate(X_val, y_val)
        logger.info(f"Disease classifier performance: {metrics}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
