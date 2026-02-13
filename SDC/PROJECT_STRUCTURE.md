# Sesame Disease Classification - Project Structure

This document provides a complete overview of the project structure and file descriptions.

## 📁 Project Structure

```
sesame-disease-classification/
│
├── Core Model Files
│   ├── sesame_disease_model.py    # CNN architecture (1.1M parameters)
│   ├── preprocessing.py            # Image preprocessing & SegNet segmentation
│   ├── augmentation.py             # Data augmentation pipelines
│   ├── dataset.py                  # PyTorch Dataset & DataLoader
│   ├── train.py                    # Training loop & utilities
│   └── evaluate.py                 # Evaluation & inference
│
├── Application Files
│   ├── main.py                     # Main training script
│   ├── inference.py                # CLI inference tool
│   ├── api.py                      # FastAPI production server
│   └── utils.py                    # Utility functions
│
├── Configuration
│   ├── config.yaml                 # Training configuration
│   └── requirements.txt            # Python dependencies
│
├── Deployment
│   ├── Dockerfile                  # Docker container config
│   └── docker-compose.yml          # Docker Compose setup
│
├── Documentation
│   ├── README.md                   # Complete documentation
│   ├── QUICKSTART.md               # Quick start guide
│   └── PROJECT_STRUCTURE.md        # This file
│
└── Runtime Directories (created during use)
    ├── data/                       # Training/validation/test data
    ├── checkpoints/                # Saved model checkpoints
    └── logs/                       # TensorBoard logs
```

## 📄 File Descriptions

### Core Model Files

#### `sesame_disease_model.py`
- **Purpose**: Defines the custom CNN architecture
- **Key Components**:
  - `SesameDiseaseCNN`: Main model class with 10 convolutional layers
  - `create_model()`: Factory function for model creation
- **Parameters**: ~1.1M (much smaller than InceptionV3: 24M, Xception: 23M)
- **Input**: 299×299×3 RGB images
- **Output**: 3 classes (bacterial_blight, phyllody, healthy)

#### `preprocessing.py`
- **Purpose**: Image preprocessing and segmentation
- **Key Components**:
  - `ImagePreprocessor`: Median filtering, contrast stretching
  - `SimpleSegNet`: Semantic segmentation for leaf extraction
  - `LeafSegmenter`: Wrapper for segmentation inference
- **Methods**: Follows paper's preprocessing pipeline exactly

#### `augmentation.py`
- **Purpose**: Data augmentation for training robustness
- **Key Components**:
  - `TrainingAugmentation`: Heavy augmentation for training
  - `ValidationAugmentation`: Minimal augmentation for validation
  - `TestTimeAugmentation`: TTA for improved inference
- **Augmentations**: Rotation, flipping, scaling, translation, shearing

#### `dataset.py`
- **Purpose**: Data loading and management
- **Key Components**:
  - `SesameDataset`: PyTorch Dataset for sesame images
  - `SesameDataModule`: Complete data pipeline manager
  - `create_data_splits()`: Split data into train/val/test
- **Features**: Weighted sampling for imbalanced datasets

#### `train.py`
- **Purpose**: Training loop and optimization
- **Key Components**:
  - `Trainer`: Main training class with validation
  - `create_optimizer()`: SGD/Adam/AdamW optimizers
  - `create_scheduler()`: Learning rate schedulers
- **Features**: TensorBoard logging, checkpointing, resume training

#### `evaluate.py`
- **Purpose**: Model evaluation and inference
- **Key Components**:
  - `ModelEvaluator`: Comprehensive evaluation metrics
  - `InferencePipeline`: Complete inference pipeline
- **Metrics**: Accuracy, precision, recall, F1-score, confusion matrix

### Application Files

#### `main.py`
- **Purpose**: Main entry point for training
- **Usage**: `python main.py --config config.yaml`
- **Features**: CLI arguments, config file support, automatic evaluation

#### `inference.py`
- **Purpose**: Command-line inference tool
- **Usage**: 
  - Single: `python inference.py --image test.jpg`
  - Batch: `python inference.py --batch ./images/`
- **Features**: JSON output, batch processing, verbose mode

#### `api.py`
- **Purpose**: Production REST API
- **Usage**: `uvicorn api:app --host 0.0.0.0 --port 8000`
- **Endpoints**:
  - `POST /predict`: Single image prediction
  - `POST /predict/batch`: Batch prediction
  - `GET /health`: Health check
  - `GET /model/info`: Model information
- **Features**: CORS support, error handling, request validation

#### `utils.py`
- **Purpose**: Utility functions and tools
- **Functions**:
  - `plot_training_history()`: Visualize training progress
  - `analyze_model_architecture()`: Model analysis
  - `export_model_summary()`: Export model info to JSON

### Configuration

#### `config.yaml`
- **Purpose**: Centralized configuration
- **Sections**:
  - Data configuration
  - Model hyperparameters
  - Training settings
  - Augmentation parameters
  - Device settings

#### `requirements.txt`
- **Purpose**: Python package dependencies
- **Key Packages**:
  - PyTorch 2.0+
  - Albumentations (augmentation)
  - FastAPI (API server)
  - TensorBoard (logging)
  - OpenCV (image processing)

### Deployment

#### `Dockerfile`
- **Purpose**: Container image definition
- **Base**: Python 3.10-slim
- **Exposes**: Port 8000
- **Includes**: Health check, environment setup

#### `docker-compose.yml`
- **Purpose**: Multi-container orchestration
- **Services**:
  - `sesame-api`: Main API service
  - `nginx`: Optional reverse proxy
- **Features**: GPU support (optional), volume mounts

## 🔄 Data Flow

### Training Flow
```
Raw Images → Preprocessing → Augmentation → Model → Loss → Optimizer → Updated Weights
                                              ↓
                                         Validation
                                              ↓
                                        Checkpointing
```

### Inference Flow
```
Input Image → Preprocessing → Segmentation → Model → Softmax → Prediction
```

### API Flow
```
HTTP Request → FastAPI → Image Processing → Model Inference → JSON Response
```

## 🎯 Key Design Decisions

1. **Model Architecture**: Custom CNN with mixed pooling for better feature extraction
2. **Preprocessing**: Median filtering + contrast stretching as per paper
3. **Augmentation**: Heavy augmentation for training, minimal for validation
4. **Training**: SGD optimizer with cosine annealing scheduler
5. **Production**: FastAPI for scalability and Docker for deployment

## 📊 Performance Characteristics

### Model Metrics
- **Parameters**: 1.1M (22× smaller than InceptionV3)
- **Input Size**: 299×299×3
- **Training Time**: ~23 minutes for 40 epochs (GPU)
- **Inference Time**: ~10ms per image (GPU), ~100ms (CPU)

### Accuracy (from paper)
- **Training**: 98.0%
- **Validation**: 97.78%
- **Testing**: 96.67%

### Resource Requirements
- **Training**: 4GB GPU memory, 8GB RAM
- **Inference**: 2GB GPU memory, 4GB RAM
- **Disk Space**: ~500MB for model + datasets

## 🔧 Extension Points

1. **Add New Classes**: Modify `num_classes` in config
2. **Custom Preprocessing**: Extend `ImagePreprocessor` class
3. **New Augmentations**: Add to `augmentation.py`
4. **Different Optimizers**: Extend `create_optimizer()`
5. **Custom Metrics**: Add to `ModelEvaluator`

## 📝 Best Practices

1. **Always use config file** for reproducibility
2. **Monitor training** with TensorBoard
3. **Save checkpoints** regularly during training
4. **Use validation set** for hyperparameter tuning
5. **Test on held-out data** for final evaluation
6. **Version control** your experiments

## 🚀 Production Checklist

- [ ] Train model with full dataset
- [ ] Evaluate on test set
- [ ] Export best model checkpoint
- [ ] Test API locally
- [ ] Create Docker image
- [ ] Set up monitoring
- [ ] Configure scaling
- [ ] Deploy to production
- [ ] Set up CI/CD pipeline

## 📚 Additional Resources

- **Paper**: Nibret et al., "Sesame Plant Disease Classification Using Deep CNNs", Applied Sciences, 2025
- **PyTorch Docs**: https://pytorch.org/docs/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Albumentations**: https://albumentations.ai/

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues or questions:
- Check documentation first
- Search existing issues
- Open a new issue with details
- Contact: [your-email@example.com]
