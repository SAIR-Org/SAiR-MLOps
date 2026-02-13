"""
Sesame Disease Classification Model - PyTorch Implementation
Based on: "Sesame Plant Disease Classification Using Deep Convolution Neural Networks"
Authors: Nibret et al., 2025

This module implements the custom CNN architecture described in the paper for classifying:
- Bacterial Blight Infected
- Phyllody Infected  
- Healthy Sesame Leaves
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SesameDiseaseCNN(nn.Module):
    """
    Custom CNN architecture for sesame disease classification.
    
    Architecture follows the paper's specifications:
    - Input: 299x299x3 RGB images
    - 10 convolutional layers with varying filter sizes (1x1, 3x3, 5x5, 7x7)
    - Mixed pooling (max and average)
    - Batch normalization after each conv layer
    - ReLU activation
    - Dropout for regularization
    - Output: 3 classes (Bacterial Blight, Phyllody, Healthy)
    
    Total parameters: ~1.1M (much smaller than InceptionV3: 24M, Xception: 23M)
    """
    
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(SesameDiseaseCNN, self).__init__()
        
        # First convolutional block (1x1 conv)
        self.conv1_1x1 = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 299x299 -> 149x149
        
        # Second convolutional block (3x3 conv)
        self.conv2_3x3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 149x149 -> 74x74
        
        # Third convolutional block (5x5 conv)
        self.conv3_5x5 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 74x74 -> 37x37
        
        # Fourth convolutional block (7x7 conv)
        self.conv4_7x7 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding='same')
        self.bn4 = nn.BatchNorm2d(64)
        
        # Fifth convolutional block (5x5 conv)
        self.conv5_5x5 = nn.Conv2d(64, 96, kernel_size=5, stride=1, padding='same')
        self.bn5 = nn.BatchNorm2d(96)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)  # 37x37 -> 18x18
        
        # Sixth convolutional block (3x3 conv)
        self.conv6_3x3 = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding='same')
        self.bn6 = nn.BatchNorm2d(128)
        
        # Seventh convolutional block (1x1 conv)
        self.conv7_1x1 = nn.Conv2d(128, 160, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(160)
        
        # Eighth convolutional block (3x3 conv)
        self.conv8_3x3 = nn.Conv2d(160, 192, kernel_size=3, stride=1, padding='same')
        self.bn8 = nn.BatchNorm2d(192)
        self.pool8 = nn.AvgPool2d(kernel_size=2, stride=2)  # 18x18 -> 9x9
        
        # Ninth convolutional block (3x3 conv)
        self.conv9_3x3 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding='same')
        self.bn9 = nn.BatchNorm2d(192)
        
        # Tenth convolutional block (1x1 conv)
        self.conv10_1x1 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.bn10 = nn.BatchNorm2d(192)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate flattened size: 192 channels * 9 * 9
        self.flattened_size = 192 * 9 * 9
        
        # Fully connected layer
        self.fc = nn.Linear(self.flattened_size, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 299, 299)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Block 1: 1x1 conv + max pool
        x = F.relu(self.bn1(self.conv1_1x1(x)))
        x = self.pool1(x)
        
        # Block 2: 3x3 conv + avg pool
        x = F.relu(self.bn2(self.conv2_3x3(x)))
        x = self.pool2(x)
        
        # Block 3: 5x5 conv + max pool
        x = F.relu(self.bn3(self.conv3_5x5(x)))
        x = self.pool3(x)
        
        # Block 4: 7x7 conv
        x = F.relu(self.bn4(self.conv4_7x7(x)))
        
        # Block 5: 5x5 conv + avg pool
        x = F.relu(self.bn5(self.conv5_5x5(x)))
        x = self.pool5(x)
        
        # Block 6: 3x3 conv
        x = F.relu(self.bn6(self.conv6_3x3(x)))
        
        # Block 7: 1x1 conv
        x = F.relu(self.bn7(self.conv7_1x1(x)))
        
        # Block 8: 3x3 conv + avg pool
        x = F.relu(self.bn8(self.conv8_3x3(x)))
        x = self.pool8(x)
        
        # Block 9: 3x3 conv
        x = F.relu(self.bn9(self.conv9_3x3(x)))
        
        # Block 10: 1x1 conv
        x = F.relu(self.bn10(self.conv10_1x1(x)))
        
        # Dropout
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x
    
    def get_num_parameters(self):
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_classes=3, dropout_rate=0.5, device='cuda'):
    """
    Factory function to create and initialize the model.
    
    Args:
        num_classes: Number of output classes (default: 3)
        dropout_rate: Dropout probability (default: 0.5)
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Initialized model moved to specified device
    """
    model = SesameDiseaseCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    
    print(f"Model created with {model.get_num_parameters():,} trainable parameters")
    print(f"Model loaded on device: {device}")
    
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)
    
    # Test with dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 299, 299).to(device)
    
    print(f"\nInput shape: {dummy_input.shape}")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output logits:\n{output}")
    
    # Apply softmax to get probabilities
    probs = F.softmax(output, dim=1)
    print(f"\nClass probabilities:\n{probs}")
