# File: scripts/generate_sample_data.py
# Generate sample datasets for testing and demonstration

#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.synthetic_generator import SyntheticDataGenerator
from src.utils.logger import setup_logger
import logging

def main():
    """Generate sample datasets"""
    
    parser = argparse.ArgumentParser(description="Generate FE-AI Sample Datasets")
    parser.add_argument("--output", required=True,
                       help="Output file path (CSV format)")
    parser.add_argument("--subjects", type=int, default=100,
                       help="Number of subjects to generate")
    parser.add_argument("--diseases", nargs="+",
                       default=["Normal", "Parkinson", "Huntington", "Ataxia", "MultipleSclerosis"],
                       help="Diseases to include")
    parser.add_argument("--duration", type=int, nargs=2, default=[60, 120],
                       help="Duration range in seconds (min max)")
    parser.add_argument("--sampling-rate", type=int, default=50,
                       help="Sampling rate in Hz")
    parser.add_argument("--balanced", action="store_true",
                       help="Create balanced dataset with train/val/test splits")
    parser.add_argument("--artifacts", action="store_true",
                       help="Add realistic measurement artifacts")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger("SampleDataGen")
    logger.info("Starting sample data generation...")
    
    try:
        # Initialize generator
        generator = SyntheticDataGenerator(random_seed=42)
        
        if args.balanced:
            # Generate balanced dataset with splits
            logger.info(f"Generating balanced dataset with {args.subjects} subjects")
            
            train_df, val_df, test_df = generator.generate_balanced_dataset(
                total_subjects=args.subjects,
                test_split=0.2,
                validation_split=0.2
            )
            
            # Save splits
            output_path = Path(args.output)
            base_name = output_path.stem
            extension = output_path.suffix
            
            train_path = output_path.parent / f"{base_name}_train{extension}"
            val_path = output_path.parent / f"{base_name}_val{extension}"
            test_path = output_path.parent / f"{base_name}_test{extension}"
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            logger.info(f"Balanced dataset saved:")
            logger.info(f"  Training: {train_path} ({len(train_df)} samples)")
            logger.info(f"  Validation: {val_path} ({len(val_df)} samples)")
            logger.info(f"  Test: {test_path} ({len(test_df)} samples)")
            
        else:
            # Generate single dataset
            logger.info(f"Generating dataset with {args.subjects} subjects")
            
            dataset = generator.generate_dataset(
                n_subjects=args.subjects,
                diseases=args.diseases,
                duration_range=tuple(args.duration),
                sampling_rate=args.sampling_rate,
                include_demographics=True
            )
            
            # Add artifacts if requested
            if args.artifacts:
                logger.info("Adding realistic measurement artifacts...")
                dataset = generator.add_realistic_artifacts(dataset)
            
            # Save dataset
            dataset.to_csv(args.output, index=False)
            
            logger.info(f"Dataset saved: {args.output}")
            logger.info(f"  Total samples: {len(dataset):,}")
            logger.info(f"  Subjects: {len(dataset['subject_id'].unique())}")
            logger.info(f"  Disease distribution:")
            for disease, count in dataset.groupby('subject_id')['disease'].first().value_counts().items():
                logger.info(f"    {disease}: {count}")
        
        logger.info("Sample data generation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Sample data generation failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
