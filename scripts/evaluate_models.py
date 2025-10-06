"""
File: scripts/evaluate_models.py
Comprehensive model evaluation script
"""

import argparse
import logging
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.gait_detector import GaitDetector
from src.models.disease_classifier import DiseaseClassifier
from src.data.data_loader import DataLoader as GaitDataLoader
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger()


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations
    """
    
    def __init__(self, model_path: str, model_type: str, config: Config):
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = self._load_model()
        
        # Results storage
        self.results = {
            'model_path': str(model_path),
            'model_type': model_type,
            'device': self.device,
            'metrics': {},
            'predictions': [],
            'labels': []
        }
    
    def _load_model(self):
        """Load trained model from checkpoint"""
        logger.info(f"Loading model from {self.model_path}")
        
        if self.model_type == 'gait_detector':
            model = GaitDetector(input_dim=9, hidden_dim=128)
        elif self.model_type == 'disease_classifier':
            model = DiseaseClassifier(num_classes=5)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
        return model
    
    def evaluate(self, test_data_path: str):
        """
        Evaluate model on test dataset
        
        Args:
            test_data_path: Path to test data
        """
        logger.info(f"Loading test data from {test_data_path}")
        
        # Load test data
        data_loader = GaitDataLoader(test_data_path)
        test_loader = data_loader.get_dataloader(batch_size=32, shuffle=False)
        
        # Run evaluation
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_data)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Convert to probabilities
                if self.model_type == 'gait_detector':
                    # Binary classification
                    probs = outputs.squeeze()
                    preds = (probs > 0.5).float()
                else:
                    # Multi-class classification
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        # Store results
        self.results['predictions'] = all_predictions
        self.results['labels'] = all_labels
        self.results['probabilities'] = all_probabilities
        
        # Calculate metrics
        self._calculate_metrics()
        
        # Generate visualizations
        self._generate_visualizations()
        
        return self.results
    
    def _calculate_metrics(self):
        """Calculate evaluation metrics"""
        predictions = np.array(self.results['predictions'])
        labels = np.array(self.results['labels'])
        probabilities = np.array(self.results['probabilities'])
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        self.results['metrics']['accuracy'] = float(accuracy)
        self.results['metrics']['precision'] = float(precision)
        self.results['metrics']['recall'] = float(recall)
        self.results['metrics']['f1_score'] = float(f1)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        self.results['metrics']['confusion_matrix'] = cm.tolist()
        
        # ROC-AUC for binary classification
        if self.model_type == 'gait_detector':
            try:
                roc_auc = roc_auc_score(labels, probabilities)
                self.results['metrics']['roc_auc'] = float(roc_auc)
                
                # ROC curve data
                fpr, tpr, thresholds = roc_curve(labels, probabilities)
                self.results['metrics']['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        # Classification report
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        self.results['metrics']['classification_report'] = report
        
        logger.info(f"Evaluation Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
    
    def _generate_visualizations(self):
        """Generate evaluation visualizations"""
        output_dir = Path('outputs/evaluation')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        predictions = np.array(self.results['predictions'])
        labels = np.array(self.results['labels'])
        
        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
        
        # ROC Curve (for binary classification)
        if self.model_type == 'gait_detector' and 'roc_curve' in self.results['metrics']:
            roc_data = self.results['metrics']['roc_curve']
            
            plt.figure(figsize=(10, 8))
            plt.plot(roc_data['fpr'], roc_data['tpr'], 
                    label=f"ROC (AUC = {self.results['metrics']['roc_auc']:.3f})",
                    linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved ROC curve to {output_dir / 'roc_curve.png'}")
        
        # Prediction distribution
        plt.figure(figsize=(10, 6))
        if self.model_type == 'gait_detector':
            plt.hist(self.results['probabilities'], bins=50, alpha=0.7)
            plt.xlabel('Prediction Probability')
            plt.ylabel('Count')
            plt.title('Prediction Distribution')
        else:
            plt.hist(predictions, bins=range(6), alpha=0.7)
            plt.xlabel('Predicted Class')
            plt.ylabel('Count')
            plt.title('Prediction Distribution')
        plt.savefig(output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved prediction distribution to {output_dir / 'prediction_distribution.png'}")
    
    def save_results(self, output_path: str):
        """Save evaluation results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def generate_report(self, output_path: str):
        """Generate comprehensive evaluation report"""
        output_path = Path(output_path)
        
        report = f"""
# Model Evaluation Report

## Model Information
- Model Type: {self.model_type}
- Model Path: {self.model_path}
- Device: {self.device}

## Performance Metrics

### Overall Performance
- **Accuracy**: {self.results['metrics']['accuracy']:.4f}
- **Precision**: {self.results['metrics']['precision']:.4f}
- **Recall**: {self.results['metrics']['recall']:.4f}
- **F1-Score**: {self.results['metrics']['f1_score']:.4f}

"""
        
        if 'roc_auc' in self.results['metrics']:
            report += f"- **ROC-AUC**: {self.results['metrics']['roc_auc']:.4f}\n\n"
        
        report += """
### Confusion Matrix
```
"""
        cm = np.array(self.results['metrics']['confusion_matrix'])
        report += str(cm)
        report += """
```

### Classification Report
"""
        
        for class_name, metrics in self.results['metrics']['classification_report'].items():
            if isinstance(metrics, dict):
                report += f"\n#### {class_name}\n"
                report += f"- Precision: {metrics.get('precision', 0):.4f}\n"
                report += f"- Recall: {metrics.get('recall', 0):.4f}\n"
                report += f"- F1-Score: {metrics.get('f1-score', 0):.4f}\n"
                report += f"- Support: {metrics.get('support', 0)}\n"
        
        report += """
## Visualizations

See the following files for detailed visualizations:
- Confusion Matrix: `outputs/evaluation/confusion_matrix.png`
- Prediction Distribution: `outputs/evaluation/prediction_distribution.png`
"""
        
        if self.model_type == 'gait_detector':
            report += "- ROC Curve: `outputs/evaluation/roc_curve.png`\n"
        
        report += """
## Conclusion

"""
        accuracy = self.results['metrics']['accuracy']
        
        if accuracy >= 0.95:
            report += "The model demonstrates **excellent** performance on the test dataset.\n"
        elif accuracy >= 0.90:
            report += "The model demonstrates **good** performance on the test dataset.\n"
        elif accuracy >= 0.80:
            report += "The model demonstrates **acceptable** performance on the test dataset.\n"
        else:
            report += "The model performance is **below expectations** and may require improvement.\n"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['gait_detector', 'disease_classifier'],
                       help='Type of model to evaluate')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.model_type, config)
    
    # Run evaluation
    logger.info("Starting model evaluation...")
    results = evaluator.evaluate(args.test_data)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_results(output_dir / 'results.json')
    evaluator.generate_report(output_dir / 'evaluation_report.md')
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Accuracy:  {results['metrics']['accuracy']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall:    {results['metrics']['recall']:.4f}")
    print(f"F1-Score:  {results['metrics']['f1_score']:.4f}")
    if 'roc_auc' in results['metrics']:
        print(f"ROC-AUC:   {results['metrics']['roc_auc']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
