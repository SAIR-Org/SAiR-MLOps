"""
Dataset and DataLoader Module for Sesame Disease Classification

Handles loading and preprocessing of sesame leaf images.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import json


class SesameDataset(Dataset):
    """
    PyTorch Dataset for sesame disease images.
    
    Expected directory structure:
    data_dir/
        bacterial_blight/
            image1.jpg
            image2.jpg
            ...
        phyllody/
            image1.jpg
            image2.jpg
            ...
        healthy/
            image1.jpg
            image2.jpg
            ...
    """
    
    def __init__(self, 
                 data_dir: str,
                 class_names: List[str] = None,
                 transform: Optional[Callable] = None,
                 mode: str = 'train'):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing class folders
            class_names: List of class names (subdirectory names)
            transform: Transformation/augmentation to apply
            mode: One of 'train', 'val', 'test'
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mode = mode
        
        # Default class names from paper
        if class_names is None:
            self.class_names = ['bacterial_blight', 'phyllody', 'healthy']
        else:
            self.class_names = class_names
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Load all image paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} does not exist!")
                continue
            
            # Get all image files
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg')) + \
                         list(class_dir.glob('*.png'))
            
            for img_path in image_files:
                self.image_paths.append(str(img_path))
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"Loaded {len(self.image_paths)} images for {mode} set")
        print(f"Class distribution: {self.get_class_distribution()}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image_tensor, label)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get label
        label = self.labels[idx]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            # Default: just convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of samples per class."""
        distribution = {}
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            count = self.labels.count(class_idx)
            distribution[class_name] = count
        return distribution
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            Tensor of class weights
        """
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        class_weights = total_samples / (len(self.class_names) * class_counts)
        return torch.FloatTensor(class_weights)
    
    def get_sample_weights(self) -> List[float]:
        """
        Calculate per-sample weights for WeightedRandomSampler.
        
        Returns:
            List of sample weights
        """
        class_weights = self.get_class_weights().numpy()
        sample_weights = [class_weights[label] for label in self.labels]
        return sample_weights


def create_data_splits(data_dir: str,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       random_seed: int = 42,
                       output_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Create train/val/test splits from data directory.
    
    Args:
        data_dir: Root directory containing class folders
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        output_dir: Directory to save split information
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing file paths
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    data_dir = Path(data_dir)
    class_names = ['bacterial_blight', 'phyllody', 'healthy']
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for class_name in class_names:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue
        
        # Get all image files
        image_files = list(class_dir.glob('*.jpg')) + \
                     list(class_dir.glob('*.jpeg')) + \
                     list(class_dir.glob('*.png'))
        
        image_files = [str(f) for f in image_files]
        
        # First split: train and temp (val+test)
        train_files, temp_files = train_test_split(
            image_files,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
            shuffle=True
        )
        
        # Second split: val and test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_files, test_files = train_test_split(
            temp_files,
            test_size=(1 - val_size),
            random_state=random_seed,
            shuffle=True
        )
        
        splits['train'].extend(train_files)
        splits['val'].extend(val_files)
        splits['test'].extend(test_files)
    
    # Print split statistics
    print(f"Split statistics:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val: {len(splits['val'])} images")
    print(f"  Test: {len(splits['test'])} images")
    
    # Save splits to file if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'data_splits.json', 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"Splits saved to {output_dir / 'data_splits.json'}")
    
    return splits


class SesameDataModule:
    """
    Wrapper class to manage data loading for train/val/test.
    """
    
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None,
                 use_weighted_sampling: bool = True):
        """
        Initialize data module.
        
        Args:
            data_dir: Root directory with train/val/test subdirectories
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            train_transform: Transform for training data
            val_transform: Transform for validation data
            test_transform: Transform for test data
            use_weighted_sampling: Whether to use weighted sampling for imbalanced data
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_weighted_sampling = use_weighted_sampling
        
        # Store transforms
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Initialize dataloaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def setup(self):
        """Setup datasets and dataloaders."""
        # Create datasets
        train_dir = self.data_dir / 'train'
        val_dir = self.data_dir / 'val'
        test_dir = self.data_dir / 'test'
        
        if train_dir.exists():
            self.train_dataset = SesameDataset(
                train_dir,
                transform=self.train_transform,
                mode='train'
            )
        
        if val_dir.exists():
            self.val_dataset = SesameDataset(
                val_dir,
                transform=self.val_transform,
                mode='val'
            )
        
        if test_dir.exists():
            self.test_dataset = SesameDataset(
                test_dir,
                transform=self.test_transform,
                mode='test'
            )
        
        # Create dataloaders
        if self.train_dataset:
            if self.use_weighted_sampling:
                # Use weighted sampling for imbalanced datasets
                sample_weights = self.train_dataset.get_sample_weights()
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    sampler=sampler,
                    num_workers=self.num_workers,
                    pin_memory=True
                )
            else:
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True
                )
        
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        if self.train_dataset:
            return self.train_dataset.class_names
        elif self.val_dataset:
            return self.val_dataset.class_names
        elif self.test_dataset:
            return self.test_dataset.class_names
        return []
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Get class weights from training dataset."""
        if self.train_dataset:
            return self.train_dataset.get_class_weights()
        return None


if __name__ == "__main__":
    print("Testing dataset and dataloader...")
    
    # This is a test - you would replace with your actual data directory
    # Example usage shown below
    
    print("""
    Example usage:
    
    from augmentation import get_augmentation_pipeline
    
    # Create augmentation pipelines
    train_aug = get_augmentation_pipeline('train')
    val_aug = get_augmentation_pipeline('val')
    test_aug = get_augmentation_pipeline('test')
    
    # Create data module
    data_module = SesameDataModule(
        data_dir='./data',  # Should contain train/val/test subdirectories
        batch_size=32,
        num_workers=4,
        train_transform=train_aug,
        val_transform=val_aug,
        test_transform=test_aug
    )
    
    # Setup datasets and dataloaders
    data_module.setup()
    
    # Access dataloaders
    for batch_idx, (images, labels) in enumerate(data_module.train_loader):
        print(f"Batch {batch_idx}: images shape = {images.shape}, labels shape = {labels.shape}")
        break
    """)
