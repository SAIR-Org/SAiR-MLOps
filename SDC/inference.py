"""
Simple command-line inference script for sesame disease classification.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --image path/to/image.jpg --model checkpoints/best_model.pth
    python inference.py --batch path/to/images/folder/
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm
import json

from evaluate import InferencePipeline


def load_image(image_path: str) -> np.ndarray:
    """Load and convert image to RGB."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def predict_single(pipeline: InferencePipeline, image_path: str, verbose: bool = True):
    """Predict disease for a single image."""
    if verbose:
        print(f"\nProcessing: {image_path}")
    
    result = pipeline.predict(image_path, return_probs=True)
    
    if verbose:
        print(f"  Predicted class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Class probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"    {class_name}: {prob:.4f}")
    
    return result


def predict_batch(pipeline: InferencePipeline, 
                 image_dir: str,
                 output_file: str = None,
                 verbose: bool = True):
    """Predict diseases for all images in a directory."""
    image_dir = Path(image_dir)
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(ext))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    results = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            result = predict_single(pipeline, str(image_path), verbose=False)
            result['image_path'] = str(image_path)
            result['image_name'] = image_path.name
            results.append(result)
            
            if verbose:
                print(f"{image_path.name}: {result['predicted_class']} "
                      f"({result['confidence']:.4f})")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*60}")
    print("Batch Prediction Summary")
    print(f"{'='*60}")
    print(f"Total images processed: {len(results)}")
    
    # Count predictions per class
    class_counts = {}
    for result in results:
        class_name = result['predicted_class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\nPredictions by class:")
    for class_name, count in sorted(class_counts.items()):
        percentage = 100 * count / len(results)
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Calculate average confidence per class
    class_confidences = {}
    for result in results:
        class_name = result['predicted_class']
        if class_name not in class_confidences:
            class_confidences[class_name] = []
        class_confidences[class_name].append(result['confidence'])
    
    print("\nAverage confidence by class:")
    for class_name, confidences in sorted(class_confidences.items()):
        avg_conf = np.mean(confidences)
        print(f"  {class_name}: {avg_conf:.4f}")
    
    # Save results to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Sesame Disease Classification - Inference Script'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image file'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='Path to directory containing images for batch prediction'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    
    parser.add_argument(
        '--no-preprocessing',
        action='store_true',
        help='Disable preprocessing (median filtering, contrast stretching)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for batch prediction results (JSON format)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed results'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.batch:
        parser.error("Either --image or --batch must be specified")
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first or specify correct path with --model")
        return
    
    # Initialize inference pipeline
    print("Loading model...")
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    if device != args.device:
        print(f"Warning: {args.device} not available, using {device}")
    
    pipeline = InferencePipeline(
        model_path=str(model_path),
        device=device,
        use_preprocessing=not args.no_preprocessing
    )
    
    print(f"Model loaded successfully on {device}")
    
    # Single image prediction
    if args.image:
        predict_single(pipeline, args.image, verbose=args.verbose)
    
    # Batch prediction
    if args.batch:
        predict_batch(
            pipeline,
            args.batch,
            output_file=args.output,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
