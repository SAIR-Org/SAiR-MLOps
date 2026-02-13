"""
Data Augmentation Module for Sesame Disease Classification

Implements augmentation techniques from the paper:
- Rotation: [-45, 45] degrees
- Flipping: horizontal and vertical
- Stretching: scale factor [1.2, 1.5]
- Translation: [-50, 50] pixels
- Shearing: [-30, 30] degrees
"""

import numpy as np
import cv2
import torch
from torchvision import transforms
from typing import Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SesameDataAugmentation:
    """
    Augmentation pipeline for sesame disease dataset.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (299, 299), 
                 apply_augmentation: bool = True):
        """
        Initialize augmentation pipeline.
        
        Args:
            image_size: Target image size (height, width)
            apply_augmentation: Whether to apply augmentation
        """
        self.image_size = image_size
        self.apply_augmentation = apply_augmentation
        
        # Define augmentation pipeline using Albumentations
        if apply_augmentation:
            self.transform = A.Compose([
                # Rotation: [-45, 45] degrees with uniform random generation
                A.Rotate(limit=45, p=0.7, border_mode=cv2.BORDER_CONSTANT),
                
                # Flipping: horizontal and vertical
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                
                # Stretching: scale factor [1.2, 1.5]
                A.RandomScale(scale_limit=(0.2, 0.5), p=0.5),
                
                # Translation: [-50, 50] pixels
                A.ShiftScaleRotate(
                    shift_limit=50/image_size[0],  # normalized by image size
                    scale_limit=0,
                    rotate_limit=0,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT
                ),
                
                # Shearing: [-30, 30] degrees
                A.Affine(
                    shear=(-30, 30),
                    p=0.5,
                    mode=cv2.BORDER_CONSTANT
                ),
                
                # Additional augmentations for robustness
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.3),
                
                # Color augmentations
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, 
                                 saturation=0.2, hue=0.1, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=20, 
                                        sat_shift_limit=30, 
                                        val_shift_limit=20, p=1.0),
                ], p=0.3),
                
                # Resize to target size
                A.Resize(height=image_size[0], width=image_size[1]),
                
                # Normalize using ImageNet statistics
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                
                # Convert to PyTorch tensor
                ToTensorV2()
            ])
        else:
            # No augmentation, just resize and normalize
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Apply augmentation to image.
        
        Args:
            image: Input image (H, W, C) in RGB format, range [0, 255]
            
        Returns:
            Augmented tensor (C, H, W)
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Apply augmentation
        augmented = self.transform(image=image)
        
        return augmented['image']


class TrainingAugmentation:
    """
    Heavy augmentation for training set.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (299, 299)):
        self.image_size = image_size
        
        self.transform = A.Compose([
            # Geometric transformations
            A.Rotate(limit=45, p=0.8, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomScale(scale_limit=(0.2, 0.5), p=0.6),
            A.ShiftScaleRotate(
                shift_limit=50/image_size[0],
                scale_limit=0,
                rotate_limit=0,
                p=0.6,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.Affine(shear=(-30, 30), p=0.5, mode=cv2.BORDER_CONSTANT),
            
            # Perspective and distortion
            A.OneOf([
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
            ], p=0.3),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.4),
            
            # Color transformations
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, 
                             saturation=0.3, hue=0.15, p=1.0),
                A.HueSaturationValue(hue_shift_limit=25, 
                                    sat_shift_limit=40, 
                                    val_shift_limit=25, p=1.0),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, 
                          b_shift_limit=20, p=1.0),
            ], p=0.5),
            
            # Brightness and contrast
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, 
                                          contrast_limit=0.3, p=1.0),
                A.CLAHE(clip_limit=4.0, p=1.0),
            ], p=0.4),
            
            # Random erasing (cutout)
            A.CoarseDropout(
                max_holes=8,
                max_height=20,
                max_width=20,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=0.3
            ),
            
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        augmented = self.transform(image=image)
        return augmented['image']


class ValidationAugmentation:
    """
    Minimal augmentation for validation set (only resize and normalize).
    """
    
    def __init__(self, image_size: Tuple[int, int] = (299, 299)):
        self.image_size = image_size
        
        self.transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        augmented = self.transform(image=image)
        return augmented['image']


class TestTimeAugmentation:
    """
    Test-time augmentation (TTA) for improved inference accuracy.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (299, 299)):
        self.image_size = image_size
        
        # Define multiple TTA transforms
        self.transforms = [
            # Original
            A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # Horizontal flip
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # Vertical flip
            A.Compose([
                A.VerticalFlip(p=1.0),
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            # Rotate 90
            A.Compose([
                A.Rotate(limit=(90, 90), p=1.0),
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
        ]
    
    def __call__(self, image: np.ndarray) -> list:
        """
        Apply all TTA transforms.
        
        Returns:
            List of augmented tensors
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        augmented_images = []
        for transform in self.transforms:
            aug = transform(image=image)
            augmented_images.append(aug['image'])
        
        return augmented_images


def get_augmentation_pipeline(mode: str = 'train', 
                              image_size: Tuple[int, int] = (299, 299)):
    """
    Factory function to get appropriate augmentation pipeline.
    
    Args:
        mode: One of 'train', 'val', 'test', 'tta'
        image_size: Target image size
        
    Returns:
        Augmentation pipeline
    """
    if mode == 'train':
        return TrainingAugmentation(image_size)
    elif mode == 'val':
        return ValidationAugmentation(image_size)
    elif mode == 'test':
        return ValidationAugmentation(image_size)
    elif mode == 'tta':
        return TestTimeAugmentation(image_size)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'train', 'val', 'test', 'tta'")


if __name__ == "__main__":
    # Test augmentation pipelines
    print("Testing augmentation pipelines...")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Test training augmentation
    train_aug = get_augmentation_pipeline('train')
    train_tensor = train_aug(dummy_image)
    print(f"Training augmentation output shape: {train_tensor.shape}")
    print(f"Training augmentation output dtype: {train_tensor.dtype}")
    print(f"Training augmentation output range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
    
    # Test validation augmentation
    val_aug = get_augmentation_pipeline('val')
    val_tensor = val_aug(dummy_image)
    print(f"\nValidation augmentation output shape: {val_tensor.shape}")
    
    # Test TTA
    tta = get_augmentation_pipeline('tta')
    tta_tensors = tta(dummy_image)
    print(f"\nTTA generated {len(tta_tensors)} augmented versions")
    
    print("\nAugmentation tests passed!")
