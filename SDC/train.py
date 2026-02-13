"""
Training Module for Sesame Disease Classification

Implements training loop, validation, and model checkpointing.
Based on paper specifications:
- 40 epochs
- SGD optimizer with appropriate learning rate
- Cross-entropy loss
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime


class Trainer:
    """
    Trainer class for sesame disease classification model.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler,
                 device: str = 'cuda',
                 num_epochs: int = 40,
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './logs',
                 class_names: list = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            num_epochs: Number of training epochs
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            class_names: List of class names
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard writer
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=self.log_dir / timestamp)
        
        # Class names
        self.class_names = class_names or ['bacterial_blight', 'phyllody', 'healthy']
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best validation accuracy for model checkpointing
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # Current epoch
        self.current_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': 100. * correct / total
            })
            
            # Log to tensorboard (every 10 batches)
            if batch_idx % 10 == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                self.writer.add_scalar('Train/BatchAcc', 100. * correct / total, global_step)
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float, Dict]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy, per_class_metrics)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class metrics
        class_correct = [0] * len(self.class_names)
        class_total = [0] * len(self.class_names)
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs} [Val]')
        
        with torch.no_grad():
            for images, labels in pbar:
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class statistics
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / total,
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        # Calculate per-class accuracy
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            if class_total[i] > 0:
                class_acc = 100. * class_correct[i] / class_total[i]
                per_class_metrics[class_name] = {
                    'accuracy': class_acc,
                    'correct': class_correct[i],
                    'total': class_total[i]
                }
        
        return epoch_loss, epoch_acc, per_class_metrics
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'class_names': self.class_names
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_filepath)
            print(f"✓ Best model saved with validation accuracy: {self.best_val_acc:.2f}%")
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, resume_from: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            resume_from: Path to checkpoint to resume from
        """
        # Resume from checkpoint if provided
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        print(f"\n{'='*50}")
        print(f"Starting training on {self.device}")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"{'='*50}\n")
        
        # Training loop
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, per_class_metrics = self.validate_epoch()
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # Log per-class metrics
            for class_name, metrics in per_class_metrics.items():
                self.writer.add_scalar(f'ClassAccuracy/{class_name}', 
                                      metrics['accuracy'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Per-class accuracy:")
            for class_name, metrics in per_class_metrics.items():
                print(f"    {class_name}: {metrics['accuracy']:.2f}% "
                      f"({metrics['correct']}/{metrics['total']})")
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # Save regular checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', is_best=False)
            
            # Always save best model
            if is_best:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', is_best=True)
        
        # Training complete
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch + 1}")
        print(f"{'='*50}\n")
        
        # Close tensorboard writer
        self.writer.close()
        
        # Save final model
        self.save_checkpoint('final_model.pth', is_best=False)


def create_optimizer(model: nn.Module, 
                     optimizer_name: str = 'sgd',
                     lr: float = 0.001,
                     momentum: float = 0.9,
                     weight_decay: float = 1e-4) -> optim.Optimizer:
    """
    Create optimizer.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('sgd', 'adam', 'adamw')
        lr: Learning rate
        momentum: Momentum (for SGD)
        weight_decay: Weight decay for regularization
        
    Returns:
        Optimizer
    """
    if optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer,
                    scheduler_name: str = 'cosine',
                    num_epochs: int = 40,
                    **kwargs):
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: Name of scheduler ('cosine', 'step', 'plateau')
        num_epochs: Total number of epochs
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_name.lower() == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_name.lower() == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name.lower() == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 5),
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


if __name__ == "__main__":
    print("Training module loaded successfully!")
    print("Use this module to train the sesame disease classification model.")
