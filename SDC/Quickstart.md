# Quick Start Guide - Sesame Disease Classification

This guide will help you get started with the sesame disease classification model.

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- 4GB+ RAM
- 10GB+ free disk space

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install PyTorch with CUDA (if you have GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Prepare Your Data

Organize your images in this structure:

```
data/
├── train/
│   ├── bacterial_blight/
│   ├── phyllody/
│   └── healthy/
├── val/
│   ├── bacterial_blight/
│   ├── phyllody/
│   └── healthy/
└── test/
    ├── bacterial_blight/
    ├── phyllody/
    └── healthy/
```

### Step 3: Train the Model

```bash
# Train with default settings
python main.py --data_dir ./data --epochs 40 --batch_size 32

# Or use the config file
python main.py --config config.yaml
```

### Step 4: Evaluate the Model

```bash
# Single image prediction
python inference.py --image path/to/sesame_leaf.jpg

# Batch prediction
python inference.py --batch path/to/images/folder/ --output results.json
```

### Step 5: Deploy API (Optional)

```bash
# Start FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000

# Or use Docker
docker-compose up -d
```

## 📊 Expected Results

Based on the paper, you should achieve approximately:

- **Training Accuracy**: 98%
- **Validation Accuracy**: 97.78%
- **Testing Accuracy**: 96.67%

Training time: ~23 minutes per 40 epochs on a modern GPU

## 🔧 Common Commands

### Training

```bash
# Basic training
python main.py --data_dir ./data

# Training with custom parameters
python main.py --data_dir ./data --lr 0.001 --batch_size 64 --epochs 50

# Resume training from checkpoint
python main.py --resume checkpoints/checkpoint_epoch_20.pth
```

### Inference

```bash
# Single image
python inference.py --image test.jpg

# Batch inference with output file
python inference.py --batch ./test_images/ --output predictions.json

# CPU inference
python inference.py --image test.jpg --device cpu
```

### Monitoring

```bash
# View training progress
tensorboard --logdir logs

# Analyze model
python utils.py --action analyze_model --model_path checkpoints/best_model.pth

# Plot training history
python utils.py --action plot_history --model_path checkpoints/best_model.pth --output history.png
```

### API Usage

```bash
# Start server
uvicorn api:app --host 0.0.0.0 --port 8000

# Test with curl
curl -X POST "http://localhost:8000/predict" -F "file=@test.jpg"

# Get model info
curl http://localhost:8000/model/info
```

## 🐛 Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size
python main.py --batch_size 16

# Or use CPU
python main.py --device cpu
```

### CUDA Not Available

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install correct PyTorch version
# Visit: https://pytorch.org/get-started/locally/
```

### Slow Training

- Use GPU if available
- Reduce `--num_workers` if CPU bottleneck
- Increase `--batch_size` if GPU has memory

### Model Not Found

```bash
# Check if model exists
ls -lh checkpoints/

# Train model if needed
python main.py --data_dir ./data
```

## 📚 Next Steps

1. **Fine-tune hyperparameters**: Experiment with learning rate, batch size, etc.
2. **Try different augmentations**: Modify `augmentation.py`
3. **Add more classes**: Update `config.yaml` and retrain
4. **Deploy to cloud**: Use Docker and deploy to AWS/GCP/Azure
5. **Mobile deployment**: Export to ONNX or TorchScript

## 📖 Full Documentation

For detailed documentation, see [README.md](README.md)

## 🤝 Support

- Check [README.md](README.md) for detailed usage
- Open an issue on GitHub
- Contact: [your-email@example.com]

## 📄 Citation

```bibtex
@article{nibret2025sesame,
  title={Sesame Plant Disease Classification Using Deep Convolution Neural Networks},
  author={Nibret et al.},
  journal={Applied Sciences},
  year={2025}
}
```