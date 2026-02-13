"""
Utility functions for sesame disease classification project.

Includes helpers for:
- Visualization
- Model inspection
- Data statistics
- Performance analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import Counter

from sesame_disease_model import SesameDiseaseCNN


def plot_training_history(history_path: str, save_path: str = None):
    """
    Plot training history from checkpoint.
    
    Args:
        history_path: Path to checkpoint containing training history
        save_path: Path to save plot
    """
    # Load checkpoint
    checkpoint = torch.load(history_path, map_location='cpu')
    history = checkpoint['history']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot training and validation accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot learning rate
    axes[1, 0].plot(history['lr'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot accuracy gap (overfitting indicator)
    acc_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(acc_gap, linewidth=2, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap (%)')
    axes[1, 1].set_title('Training-Validation Accuracy Gap')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_model_architecture(model_path: str = None):
    """
    Analyze and print model architecture details.
    
    Args:
        model_path: Path to model checkpoint (optional)
    """
    # Create or load model
    if model_path:
        model = SesameDiseaseCNN()
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = SesameDiseaseCNN()
    
    print("="*60)
    print("Model Architecture Analysis")
    print("="*60)
    
    # Count parameters by layer type
    layer_params = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0]
        if layer_type not in layer_params:
            layer_params[layer_type] = 0
        layer_params[layer_type] += param.numel()
    
    print("\nParameters by layer type:")
    total_params = 0
    for layer_type, num_params in sorted(layer_params.items()):
        print(f"  {layer_type}: {num_params:,}")
        total_params += num_params
    
    print(f"\nTotal parameters: {total_params:,}")
    
    # Calculate model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    print(f"\nModel size:")
    print(f"  Parameters: {param_size / 1024**2:.2f} MB")
    print(f"  Buffers: {buffer_size / 1024**2:.2f} MB")
    print(f"  Total: {total_size / 1024**2:.2f} MB")
    
    # Layer-by-layer breakdown
    print("\nLayer-by-layer breakdown:")
    print(f"{'Layer':<30} {'Parameters':<15} {'Shape':<30}")
    print("-"*75)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<30} {param.numel():<15,} {str(list(param.shape)):<30}")
    
    print("="*60)


def visualize_predictions(images: List[np.ndarray],
                         true_labels: List[str],
                         pred_labels: List[str],
                         confidences: List[float],
                         class_names: List[str],
                         save_path: str = None):
    """
    Visualize a grid of predictions.
    
    Args:
        images: List of images
        true_labels: List of true class names
        pred_labels: List of predicted class names
        confidences: List of confidence scores
        class_names: List of all class names
        save_path: Path to save visualization
    """
    n_images = len(images)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx in range(len(axes)):
        if idx < n_images:
            ax = axes[idx]
            ax.imshow(images[idx])
            
            # Color code: green for correct, red for incorrect
            color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
            
            title = f"True: {true_labels[idx]}\n"
            title += f"Pred: {pred_labels[idx]}\n"
            title += f"Conf: {confidences[idx]:.3f}"
            
            ax.set_title(title, color=color, fontweight='bold')
            ax.axis('off')
        else:
            axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_metrics_per_threshold(y_true: np.ndarray,
                                    y_probs: np.ndarray,
                                    thresholds: np.ndarray = None) -> Dict:
    """
    Calculate metrics at different confidence thresholds.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        thresholds: Confidence thresholds to evaluate
        
    Returns:
        Dictionary with metrics at each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)
    
    results = {
        'thresholds': [],
        'accuracies': [],
        'coverage': []
    }
    
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    
    for threshold in thresholds:
        # Filter predictions above threshold
        mask = confidences >= threshold
        
        if mask.sum() == 0:
            continue
        
        filtered_true = y_true[mask]
        filtered_pred = y_pred[mask]
        
        # Calculate metrics
        accuracy = (filtered_true == filtered_pred).mean()
        coverage = mask.sum() / len(y_true)
        
        results['thresholds'].append(threshold)
        results['accuracies'].append(accuracy)
        results['coverage'].append(coverage)
    
    return results


def export_model_summary(model_path: str, output_file: str):
    """
    Export model summary to JSON file.
    
    Args:
        model_path: Path to model checkpoint
        output_file: Path to output JSON file
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model = SesameDiseaseCNN()
    
    # Gather information
    summary = {
        'model_name': 'SesameDiseaseCNN',
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'input_size': [299, 299, 3],
        'num_classes': 3,
        'class_names': checkpoint.get('class_names', ['bacterial_blight', 'phyllody', 'healthy']),
        'best_val_acc': checkpoint.get('best_val_acc', None),
        'final_epoch': checkpoint.get('epoch', None),
    }
    
    # Add training history if available
    if 'history' in checkpoint:
        history = checkpoint['history']
        summary['training_history'] = {
            'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'final_train_acc': history['train_acc'][-1] if history['train_acc'] else None,
            'final_val_acc': history['val_acc'][-1] if history['val_acc'] else None,
        }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Model summary exported to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Utility functions')
    parser.add_argument('--action', type=str, required=True,
                       choices=['plot_history', 'analyze_model', 'export_summary'],
                       help='Action to perform')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    if args.action == 'plot_history':
        if not args.model_path:
            print("Error: --model_path required for plot_history")
        else:
            plot_training_history(args.model_path, args.output)
    
    elif args.action == 'analyze_model':
        analyze_model_architecture(args.model_path)
    
    elif args.action == 'export_summary':
        if not args.model_path or not args.output:
            print("Error: --model_path and --output required for export_summary")
        else:
            export_model_summary(args.model_path, args.output)
