"""
Main Training Script for Sesame Disease Classification

Usage:
    python main.py --config config.yaml
    
Or with command-line arguments:
    python main.py --data_dir ./data --batch_size 32 --epochs 40
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path

# Import project modules
from sesame_disease_model import create_model
from dataset import SesameDataModule
from augmentation import get_augmentation_pipeline
from train import Trainer, create_optimizer, create_scheduler
from evaluate import ModelEvaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train Sesame Disease Classification Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Number of output classes')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam', 'adamw'],
                       help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau'],
                       help='Learning rate scheduler')
    
    # Loss arguments
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights for imbalanced data')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file (overrides CLI args)')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update args with config values
        for key, value in config.items():
            setattr(args, key, value)
    
    return args


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    print("="*60)
    print("Sesame Disease Classification - Training")
    print("="*60)
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create augmentation pipelines
    print("\nCreating augmentation pipelines...")
    train_transform = get_augmentation_pipeline('train', image_size=(299, 299))
    val_transform = get_augmentation_pipeline('val', image_size=(299, 299))
    
    # Create data module
    print("Loading datasets...")
    data_module = SesameDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        use_weighted_sampling=True
    )
    
    data_module.setup()
    
    # Get class names and weights
    class_names = data_module.get_class_names()
    print(f"Class names: {class_names}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        num_classes=args.num_classes,
        dropout_rate=args.dropout,
        device=device
    )
    
    # Create criterion (loss function)
    if args.use_class_weights:
        class_weights = data_module.get_class_weights()
        if class_weights is not None:
            class_weights = class_weights.to(device)
            print(f"Using class weights: {class_weights}")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    print("\nCreating optimizer...")
    optimizer = create_optimizer(
        model,
        optimizer_name=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    print("Creating learning rate scheduler...")
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        num_epochs=args.epochs
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        class_names=class_names
    )
    
    # Train model
    print("\nStarting training...")
    trainer.train(resume_from=args.resume)
    
    # Evaluate on test set
    if data_module.test_loader:
        print("\nEvaluating on test set...")
        evaluator = ModelEvaluator(
            model=model,
            device=device,
            class_names=class_names
        )
        
        test_results = evaluator.evaluate(data_module.test_loader)
        
        # Save confusion matrix
        save_path = Path(args.checkpoint_dir) / 'confusion_matrix.png'
        evaluator.plot_confusion_matrix(
            test_results['confusion_matrix'],
            save_path=str(save_path)
        )
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
