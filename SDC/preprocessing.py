"""
Image Preprocessing Module for Sesame Disease Classification
Implements preprocessing pipeline from the paper:
1. Median filtering for noise removal
2. Contrast stretching for enhancement
3. SegNet-based semantic segmentation for leaf extraction
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from scipy.ndimage import median_filter
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Handles image preprocessing including filtering, enhancement, and normalization.
    """
    
    def __init__(self, target_size=(299, 299)):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
    
    def median_filter_denoise(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filter to remove noise while preserving edges.
        
        The paper uses median filtering to retain important data in field images
        which may contain dust, water droplets, or other particles.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            kernel_size: Size of the median filter kernel
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 3:
            # Apply to each channel separately
            filtered = np.zeros_like(image)
            for c in range(image.shape[2]):
                filtered[:, :, c] = median_filter(image[:, :, c], size=kernel_size)
            return filtered
        else:
            return median_filter(image, size=kernel_size)
    
    def contrast_stretching(self, image: np.ndarray, 
                           lower_percentile: float = 2.0,
                           upper_percentile: float = 98.0) -> np.ndarray:
        """
        Apply contrast stretching to enhance image quality.
        
        Following equation from paper:
        S = (r - c) * ((b - a) / (d - c)) + a
        
        Where:
        - r: original pixel value
        - c, d: lower and upper limits in original image
        - a, b: desired output range (0, 255 for 8-bit images)
        
        Args:
            image: Input image
            lower_percentile: Lower percentile for stretching
            upper_percentile: Upper percentile for stretching
            
        Returns:
            Contrast-stretched image
        """
        # Calculate percentile limits
        c = np.percentile(image, lower_percentile)
        d = np.percentile(image, upper_percentile)
        
        # Define output range
        a, b = 0, 255
        
        # Apply contrast stretching formula
        stretched = (image - c) * ((b - a) / (d - c + 1e-8)) + a
        
        # Clip values to valid range
        stretched = np.clip(stretched, a, b).astype(np.uint8)
        
        return stretched
    
    def rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to grayscale.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Grayscale image (H, W)
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, image: np.ndarray, mean: Optional[list] = None, 
                       std: Optional[list] = None) -> np.ndarray:
        """
        Normalize image using mean and std.
        
        Args:
            image: Input image (H, W, C) in range [0, 255]
            mean: Mean for each channel (default: ImageNet mean)
            std: Std for each channel (default: ImageNet std)
            
        Returns:
            Normalized image
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]  # ImageNet mean
        if std is None:
            std = [0.229, 0.224, 0.225]   # ImageNet std
        
        # Convert to float and scale to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        for c in range(3):
            image[:, :, c] = (image[:, :, c] - mean[c]) / std[c]
        
        return image
    
    def preprocess_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for semantic segmentation.
        
        Pipeline:
        1. Median filtering
        2. Contrast stretching
        3. Resize
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Preprocessed image ready for segmentation
        """
        # Apply median filter
        filtered = self.median_filter_denoise(image)
        
        # Apply contrast stretching
        enhanced = self.contrast_stretching(filtered)
        
        # Resize
        resized = self.resize_image(enhanced)
        
        return resized
    
    def preprocess_for_classification(self, image: np.ndarray, 
                                     apply_filtering: bool = True) -> torch.Tensor:
        """
        Full preprocessing pipeline for classification.
        
        Args:
            image: RGB image (H, W, 3) in range [0, 255]
            apply_filtering: Whether to apply median filtering and contrast stretching
            
        Returns:
            Preprocessed tensor (1, 3, H, W) ready for model input
        """
        if apply_filtering:
            # Apply median filter
            image = self.median_filter_denoise(image)
            
            # Apply contrast stretching
            image = self.contrast_stretching(image)
        
        # Resize
        image = self.resize_image(image)
        
        # Normalize
        image = self.normalize_image(image)
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Add batch dimension (1, C, H, W)
        tensor = tensor.unsqueeze(0)
        
        return tensor


class SimpleSegNet(nn.Module):
    """
    Simplified SegNet-inspired architecture for leaf segmentation.
    
    The paper uses SegNet semantic segmentation with VGG16 backbone.
    This is a simplified version for binary segmentation (leaf vs background).
    """
    
    def __init__(self):
        super(SimpleSegNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_encoder_block(3, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        # Decoder
        self.dec4 = self._make_decoder_block(512, 256)
        self.dec3 = self._make_decoder_block(256, 128)
        self.dec2 = self._make_decoder_block(128, 64)
        self.dec1 = self._make_decoder_block(64, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, 2, kernel_size=1)  # 2 classes: background, leaf
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxUnpool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Encoder with pooling indices
        x1, indices1, size1 = self._encode_block(x, self.enc1)
        x2, indices2, size2 = self._encode_block(x1, self.enc2)
        x3, indices3, size3 = self._encode_block(x2, self.enc3)
        x4, indices4, size4 = self._encode_block(x3, self.enc4)
        
        # Decoder with unpooling
        x = self._decode_block(x4, indices4, size4, self.dec4)
        x = self._decode_block(x, indices3, size3, self.dec3)
        x = self._decode_block(x, indices2, size2, self.dec2)
        x = self._decode_block(x, indices1, size1, self.dec1)
        
        # Final classification
        x = self.final(x)
        
        return x
    
    def _encode_block(self, x, encoder):
        for layer in encoder[:-1]:
            x = layer(x)
        # Last layer is MaxPool2d with return_indices
        size = x.size()
        x, indices = encoder[-1](x)
        return x, indices, size
    
    def _decode_block(self, x, indices, size, decoder):
        # Unpool
        x = decoder[-1](x, indices, output_size=size)
        # Apply conv layers
        for layer in decoder[:-1]:
            x = layer(x)
        return x


class LeafSegmenter:
    """
    Wrapper class for leaf segmentation using SegNet.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize segmenter.
        
        Args:
            model_path: Path to pretrained segmentation model weights
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = SimpleSegNet().to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.eval()
    
    def load_model(self, model_path: str):
        """Load pretrained model weights."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded segmentation model from {model_path}")
    
    def segment_leaf(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment leaf from background.
        
        Args:
            image: Preprocessed RGB image (H, W, 3)
            
        Returns:
            Tuple of (segmented_image, binary_mask)
            - segmented_image: RGB image with background removed
            - binary_mask: Binary mask (1=leaf, 0=background)
        """
        # Convert to tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Create binary mask
        binary_mask = (pred == 1).astype(np.uint8)
        
        # Apply mask to original image
        segmented = image.copy()
        for c in range(3):
            segmented[:, :, c] = segmented[:, :, c] * binary_mask
        
        return segmented, binary_mask


if __name__ == "__main__":
    # Test preprocessor
    preprocessor = ImagePreprocessor()
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    print("Testing preprocessing pipeline...")
    
    # Test preprocessing for segmentation
    preprocessed = preprocessor.preprocess_for_segmentation(dummy_image)
    print(f"Preprocessed for segmentation shape: {preprocessed.shape}")
    
    # Test preprocessing for classification
    tensor = preprocessor.preprocess_for_classification(dummy_image)
    print(f"Preprocessed for classification shape: {tensor.shape}")
    
    print("\nPreprocessing tests passed!")
