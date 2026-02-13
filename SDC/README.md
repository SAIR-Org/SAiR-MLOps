# Sesame Plant Disease Classification

PyTorch implementation of the sesame disease classification model from the paper:

**"Sesame Plant Disease Classification Using Deep Convolution Neural Networks"**  
*Nibret et al., Applied Sciences, 2025*

## Overview

This project implements a deep CNN for classifying sesame plant diseases into three categories:
- **Bacterial Blight** - Caused by bacterial infection
- **Phyllody** - Caused by phytoplasma
- **Healthy** - No disease

The model achieves **96.67% testing accuracy**, **97.78% validation accuracy**, and **98% training accuracy** on the augmented dataset.

## Key Features

- ✅ Custom CNN architecture with ~1.1M parameters (much smaller than InceptionV3 and Xception)
- ✅ Median filtering and contrast stretching preprocessing
- ✅ SegNet-based semantic segmentation for leaf extraction
- ✅ Comprehensive data augmentation (rotation, flipping, scaling, translation, shearing)
- ✅ Production-ready FastAPI inference server
- ✅ TensorBoard logging and visualization
- ✅ Model checkpointing and resume training
- ✅ Test-time augmentation (TTA) for improved accuracy
- ✅ Class imbalance handling with weighted sampling

## Project Structure

```
├── sesame_disease_model.py   # Model architecture
├── preprocessing.py           # Image preprocessing and segmentation
├── augmentation.py            # Data augmentation pipelines
├── dataset.py                 # Dataset and DataLoader
├── train.py                   # Training loop and utilities
├── evaluate.py                # Evaluation and inference
├── main.py                    # Main training script
├── api.py                     # FastAPI inference server
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd sesame-disease-classification
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA support (if using GPU)
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision
```

## Data Preparation

### Expected Directory Structure

```
data/
├── train/
│   ├── bacterial_blight/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── phyllody/
│   │   └── ...
│   └── healthy/
│       └── ...
├── val/
│   ├── bacterial_blight/
│   ├── phyllody/
│   └── healthy/
└── test/
    ├── bacterial_blight/
    ├── phyllody/
    └── healthy/
```

### Creating Data Splits

If you have a single directory with all images organized by class:

```python
from dataset import create_data_splits

# Create train/val/test splits (70/15/15)
splits = create_data_splits(
    data_dir='./raw_data',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    output_dir='./data'
)
```

## Training

### Basic Training

```bash
python main.py --data_dir ./data --batch_size 32 --epochs 40
```

### Training with Config File

```bash
python main.py --config config.yaml
```

### Resume Training

```bash
python main.py --config config.yaml --resume checkpoints/checkpoint_epoch_20.pth
```

### Training with Custom Parameters

```bash
python main.py \
    --data_dir ./data \
    --batch_size 32 \
    --epochs 40 \
    --lr 0.001 \
    --optimizer sgd \
    --scheduler cosine \
    --use_class_weights \
    --device cuda
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

## Evaluation

### Evaluate on Test Set

```python
import torch
from evaluate import ModelEvaluator, InferencePipeline
from dataset import SesameDataModule
from augmentation import get_augmentation_pipeline

# Load trained model
pipeline = InferencePipeline(
    model_path='checkpoints/best_model.pth',
    device='cuda'
)

# Create test dataloader
val_transform = get_augmentation_pipeline('val')
data_module = SesameDataModule(
    data_dir='./data',
    batch_size=32,
    test_transform=val_transform
)
data_module.setup()

# Evaluate
evaluator = pipeline.evaluator
results = evaluator.evaluate(data_module.test_loader)

# Plot confusion matrix
evaluator.plot_confusion_matrix(
    results['confusion_matrix'],
    save_path='confusion_matrix.png'
)
```

## Inference

### Single Image Prediction

```python
from evaluate import InferencePipeline

# Load model
pipeline = InferencePipeline(
    model_path='checkpoints/best_model.pth',
    device='cuda',
    use_preprocessing=True
)

# Predict
result = pipeline.predict('path/to/image.jpg')

print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Class probabilities: {result['class_probabilities']}")
```

### Batch Prediction

```python
import cv2
import numpy as np

# Load images
images = [cv2.imread(f'image{i}.jpg') for i in range(10)]
images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

# Predict batch
results = pipeline.evaluator.predict_batch(images, batch_size=32)

for i, result in enumerate(results):
    print(f"Image {i}: {result['predicted_class']} "
          f"({result['confidence']:.4f})")
```

### Test-Time Augmentation (TTA)

```python
from augmentation import TestTimeAugmentation

# Create TTA transform
tta = TestTimeAugmentation(image_size=(299, 299))

# Predict with TTA
result = pipeline.evaluator.predict_with_tta(image, tta_transform=tta)
```

## Production Deployment

### FastAPI Server

#### Start the API server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

#### Example API Usage

**Single Image Prediction:**

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("sesame_leaf.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(result)
# Output:
# {
#     "predicted_class": "bacterial_blight",
#     "confidence": 0.9845,
#     "class_probabilities": {
#         "bacterial_blight": 0.9845,
#         "phyllody": 0.0123,
#         "healthy": 0.0032
#     },
#     "message": "Prediction successful"
# }
```

**Batch Prediction:**

```python
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
    ("files", open("image3.jpg", "rb"))
]

response = requests.post("http://localhost:8000/predict/batch", files=files)
results = response.json()
```

**Health Check:**

```bash
curl http://localhost:8000/health
```

**Get Model Info:**

```bash
curl http://localhost:8000/model/info
```

### Docker Deployment

#### Create Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Run API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Build and run:

```bash
docker build -t sesame-disease-api .
docker run -p 8000:8000 -v $(pwd)/checkpoints:/app/checkpoints sesame-disease-api
```

## Model Performance

Based on the paper's results:

| Dataset | Training Accuracy | Validation Accuracy | Testing Accuracy |
|---------|------------------|-------------------|------------------|
| Original Images | 94.0% | 92.2% | 90.0% |
| Preprocessed & Segmented | 96.0% | 95.6% | 94.4% |
| **Augmented Images** | **98.0%** | **97.8%** | **96.7%** |

### Comparison with Other Models

| Model | Parameters | Testing Accuracy | Training Time |
|-------|-----------|-----------------|---------------|
| **Proposed Model** | **1.1M** | **96.7%** | **23 min** |
| InceptionV3 | 24M | 88.9% | 6h 33min |
| Xception | 23M | 93.3% | 5h 57min |

### Per-Class Metrics (Augmented Dataset)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Bacterial Blight | 0.97 | 0.97 | 0.97 |
| Phyllody | 0.97 | 0.97 | 0.97 |
| Healthy | 0.97 | 0.97 | 0.97 |

## Model Export

### Export to ONNX

```python
import torch
from sesame_disease_model import create_model

# Load model
model = create_model(device='cpu')
checkpoint = torch.load('checkpoints/best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 299, 299)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "sesame_disease_model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

### Use ONNX Model

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("sesame_disease_model.onnx")

# Prepare input
input_data = np.random.randn(1, 3, 299, 299).astype(np.float32)

# Run inference
outputs = session.run(None, {"input": input_data})
predictions = outputs[0]
```

## Troubleshooting

### Out of Memory Error

Reduce batch size:
```bash
python main.py --batch_size 16
```

### Slow Training

- Enable GPU acceleration
- Reduce number of workers: `--num_workers 2`
- Use mixed precision training (add to train.py)

### Class Imbalance

Enable weighted sampling:
```bash
python main.py --use_class_weights
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{nibret2025sesame,
  title={Sesame Plant Disease Classification Using Deep Convolution Neural Networks},
  author={Nibret, Eyerusalem Alebachew and Mequanenit, Azanu Mirolgn and Ayalew, Aleka Melese and Kusrini, Kusrini and Mart{\'i}nez-B{\'e}jar, Rodrigo},
  journal={Applied Sciences},
  volume={15},
  number={4},
  pages={2124},
  year={2025},
  publisher={MDPI}
}
```

## License

[Add your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

- Original paper authors: Nibret et al.
- Research centers: Gondar and Humera Agriculture Research Centers
- Data collection location: Metema sesame fields