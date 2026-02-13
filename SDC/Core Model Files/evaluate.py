"""
Evaluation and Inference Module for Sesame Disease Classification

Includes:
- Model evaluation on test set
- Confusion matrix and classification metrics
- Single image inference
- Batch inference
- Test-time augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ModelEvaluator:
    """
    Evaluator class for model testing and inference.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda',
                 class_names: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device to run inference on
            class_names: List of class names
        """
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.class_names = class_names or ['bacterial_blight', 'phyllody', 'healthy']
    
    def evaluate(self, test_loader) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions = []
        all_labels = []
        all_probs = []
        
        print("Evaluating model on test set...")
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.class_names))
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_probs': y_probs
        }
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Test Results")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {accuracy*100:.2f}%")
        print(f"\nPer-class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {precision[i]:.4f}")
            print(f"    Recall: {recall[i]:.4f}")
            print(f"    F1-Score: {f1[i]:.4f}")
            print(f"    Support: {support[i]}")
        print(f"\nClassification Report:")
        print(report)
        print(f"{'='*50}\n")
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def predict_single_image(self, 
                            image: np.ndarray,
                            transform=None,
                            return_probs: bool = True) -> Dict:
        """
        Predict class for a single image.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            transform: Transformation to apply
            return_probs: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Apply transform if provided
        if transform:
            image_tensor = transform(image)
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
        else:
            # Default preprocessing
            image_resized = cv2.resize(image, (299, 299))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
            image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = F.softmax(output, dim=1)
            confidence, predicted = probs.max(1)
        
        # Get results
        pred_class_idx = predicted.item()
        pred_class_name = self.class_names[pred_class_idx]
        confidence_score = confidence.item()
        
        result = {
            'predicted_class': pred_class_name,
            'predicted_class_idx': pred_class_idx,
            'confidence': confidence_score
        }
        
        if return_probs:
            class_probs = {
                self.class_names[i]: probs[0, i].item()
                for i in range(len(self.class_names))
            }
            result['class_probabilities'] = class_probs
        
        return result
    
    def predict_with_tta(self,
                        image: np.ndarray,
                        tta_transform) -> Dict:
        """
        Predict with test-time augmentation.
        
        Args:
            image: Input image (H, W, 3) in RGB format
            tta_transform: TTA transformation
            
        Returns:
            Dictionary with averaged prediction results
        """
        # Get augmented versions
        augmented_images = tta_transform(image)
        
        # Predict on each version
        all_probs = []
        
        with torch.no_grad():
            for aug_image in augmented_images:
                aug_image = aug_image.unsqueeze(0).to(self.device)
                output = self.model(aug_image)
                probs = F.softmax(output, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)[0]
        
        # Get prediction
        pred_class_idx = np.argmax(avg_probs)
        pred_class_name = self.class_names[pred_class_idx]
        confidence_score = avg_probs[pred_class_idx]
        
        result = {
            'predicted_class': pred_class_name,
            'predicted_class_idx': pred_class_idx,
            'confidence': confidence_score,
            'class_probabilities': {
                self.class_names[i]: float(avg_probs[i])
                for i in range(len(self.class_names))
            }
        }
        
        return result
    
    def predict_batch(self,
                     images: List[np.ndarray],
                     transform=None,
                     batch_size: int = 32) -> List[Dict]:
        """
        Predict classes for a batch of images.
        
        Args:
            images: List of images (H, W, 3) in RGB format
            transform: Transformation to apply
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # Prepare batch
            if transform:
                batch_tensors = [transform(img) for img in batch_images]
            else:
                batch_tensors = []
                for img in batch_images:
                    img_resized = cv2.resize(img, (299, 299))
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1)
                    batch_tensors.append(img_tensor)
            
            # Stack into batch
            batch = torch.stack(batch_tensors).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                confidences, predictions = probs.max(1)
            
            # Store results
            for j in range(len(batch_images)):
                pred_class_idx = predictions[j].item()
                pred_class_name = self.class_names[pred_class_idx]
                confidence_score = confidences[j].item()
                
                result = {
                    'predicted_class': pred_class_name,
                    'predicted_class_idx': pred_class_idx,
                    'confidence': confidence_score,
                    'class_probabilities': {
                        self.class_names[k]: probs[j, k].item()
                        for k in range(len(self.class_names))
                    }
                }
                results.append(result)
        
        return results


class InferencePipeline:
    """
    Complete inference pipeline including preprocessing.
    """
    
    def __init__(self,
                 model_path: str,
                 device: str = 'cuda',
                 class_names: List[str] = None,
                 use_preprocessing: bool = True):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            class_names: List of class names
            use_preprocessing: Whether to apply preprocessing
        """
        from sesame_disease_model import SesameDiseaseCNN
        from preprocessing import ImagePreprocessor
        from augmentation import ValidationAugmentation
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names or ['bacterial_blight', 'phyllody', 'healthy']
        
        # Load model
        self.model = SesameDiseaseCNN(num_classes=len(self.class_names))
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor and transform
        self.use_preprocessing = use_preprocessing
        if use_preprocessing:
            self.preprocessor = ImagePreprocessor(target_size=(299, 299))
        
        self.transform = ValidationAugmentation(image_size=(299, 299))
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            self.model,
            device=device,
            class_names=self.class_names
        )
        
        print(f"Model loaded from {model_path}")
        print(f"Running on {self.device}")
    
    def predict(self, 
                image_path: str,
                return_probs: bool = True) -> Dict:
        """
        Predict disease class for an image file.
        
        Args:
            image_path: Path to image file
            return_probs: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing if enabled
        if self.use_preprocessing:
            image = self.preprocessor.preprocess_for_segmentation(image)
        
        # Predict
        result = self.evaluator.predict_single_image(
            image,
            transform=self.transform,
            return_probs=return_probs
        )
        
        return result
    
    def predict_array(self,
                     image: np.ndarray,
                     return_probs: bool = True) -> Dict:
        """
        Predict disease class for an image array.
        
        Args:
            image: Image array (H, W, 3) in RGB format
            return_probs: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Apply preprocessing if enabled
        if self.use_preprocessing:
            image = self.preprocessor.preprocess_for_segmentation(image)
        
        # Predict
        result = self.evaluator.predict_single_image(
            image,
            transform=self.transform,
            return_probs=return_probs
        )
        
        return result


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("Use this module to evaluate trained models and make predictions.")
