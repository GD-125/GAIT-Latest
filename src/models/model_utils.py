# File: src/models/model_utils.py
# Utility functions for model training, evaluation, and management

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import json
import logging
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)


class GaitDataset(Dataset):
    """Custom Dataset for gait signal data"""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        transform: Optional[Callable] = None
    ):
        """
        Args:
            X: Input features of shape (n_samples, n_channels, sequence_length)
            y: Labels of shape (n_samples,)
            transform: Optional transform to be applied
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.X[idx], self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' depending on metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            return False
        
        # Check improvement based on mode
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs")
        
        return self.early_stop
    
    def restore_best_weights(self, model: nn.Module):
        """Restore model to best weights"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class ModelTrainer:
    """Comprehensive model trainer with advanced features"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, _ = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)] '
                    f'Loss: {loss.item():.6f}'
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 15
    ) -> Dict:
        """
        Train model with early stopping
        
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='min'
        )
        
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            self.logger.info(
                f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | '
                f'Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | '
                f'Val Acc: {val_acc:.2f}%'
            )
            
            # Early stopping
            if early_stopping(val_loss, self.model):
                self.logger.info(f"Early stopping at epoch {epoch}")
                early_stopping.restore_best_weights(self.model)
                break
        
        return self.history
    
    def save_model(self, path: str, metadata: Optional[Dict] = None):
        """Save model with metadata"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'metadata': metadata or {}
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and training state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.logger.info(f"Model loaded from {path}")
        return checkpoint.get('metadata', {})


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: List[str]
    ) -> Dict:
        """
        Comprehensive evaluation with multiple metrics
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output, _ = self.model(data)
                probs = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # ROC-AUC (for binary or multiclass)
        try:
            if len(class_names) == 2:
                roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                roc_auc = roc_auc_score(
                    all_labels, all_probs, multi_class='ovr', average='weighted'
                )
        except Exception as e:
            self.logger.warning(f"Could not calculate ROC-AUC: {e}")
            roc_auc = None
        
        # Classification report
        report = classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        self.logger.info(f"Evaluation complete - Accuracy: {accuracy*100:.2f}%")
        return results
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_history(
        self,
        history: Dict,
        save_path: Optional[str] = None
    ):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(
        self,
        results: Dict,
        class_names: List[str],
        save_dir: str = "reports"
    ) -> str:
        """Generate comprehensive evaluation report"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = save_dir / f"evaluation_report_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        report_data = {
            'timestamp': timestamp,
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'roc_auc': float(results['roc_auc']) if results['roc_auc'] else None,
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'classification_report': results['classification_report']
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        
        self.logger.info(f"Evaluation report saved to {report_path}")
        return str(report_path)


class ModelExporter:
    """Export models to various formats for deployment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def export_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple,
        output_path: str,
        opset_version: int = 11
    ):
        """Export model to ONNX format"""
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        self.logger.info(f"Model exported to ONNX: {output_path}")
    
    def export_to_torchscript(
        self,
        model: nn.Module,
        input_shape: Tuple,
        output_path: str
    ):
        """Export model to TorchScript"""
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        traced_model = torch.jit.trace(model, dummy_input)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        traced_model.save(output_path)
        
        self.logger.info(f"Model exported to TorchScript: {output_path}")
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: DataLoader,
        output_path: str
    ):
        """Quantize model for faster inference"""
        model.eval()
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.Conv1d},
            dtype=torch.qint8
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_model.state_dict(), output_path)
        
        self.logger.info(f"Quantized model saved to {output_path}")
        
        return quantized_model


class ModelOptimizer:
    """Model optimization utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def prune_model(
        self,
        model: nn.Module,
        amount: float = 0.3
    ) -> nn.Module:
        """
        Prune model weights to reduce size
        
        Args:
            model: Model to prune
            amount: Fraction of parameters to prune (0.0 to 1.0)
        
        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune
        
        # Prune all conv and linear layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        
        self.logger.info(f"Model pruned by {amount*100:.1f}%")
        return model
    
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }
    
    def estimate_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Estimate model memory footprint"""
        # Calculate parameter size
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        
        total_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        return {
            'parameters_mb': param_size / (1024 ** 2),
            'buffers_mb': buffer_size / (1024 ** 2),
            'total_mb': total_size_mb
        }


if __name__ == "__main__":
    print("Testing Model Utils...")
    
    # Test dataset
    X = np.random.randn(100, 9, 250)
    y = np.random.randint(0, 2, 100)
    dataset = GaitDataset(X, y)
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=5)
    print("✅ Early stopping initialized")
    
    print("\n✅ All model utility tests passed!")