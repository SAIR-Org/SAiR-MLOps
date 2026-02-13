"""
FastAPI Inference Server for Sesame Disease Classification

Production-ready API for model inference.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
import io
from PIL import Image
import torch
import logging
from pathlib import Path

from evaluate import InferencePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Sesame Disease Classification API",
    description="API for classifying sesame plant diseases using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference pipeline
inference_pipeline = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_class: str
    confidence: float
    class_probabilities: dict
    message: str = "Prediction successful"


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global inference_pipeline
    
    try:
        # Path to trained model
        model_path = Path("checkpoints/best_model.pth")
        
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Using default path.")
            model_path = Path("checkpoints/final_model.pth")
        
        if not model_path.exists():
            logger.error("No trained model found! Please train the model first.")
            raise FileNotFoundError("No trained model found")
        
        # Initialize inference pipeline
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading model on device: {device}")
        
        inference_pipeline = InferencePipeline(
            model_path=str(model_path),
            device=device,
            class_names=['bacterial_blight', 'phyllody', 'healthy'],
            use_preprocessing=True
        )
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=inference_pipeline is not None,
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=inference_pipeline is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict disease class for uploaded image.
    
    Args:
        file: Uploaded image file (JPG, PNG)
        
    Returns:
        Prediction results with class and confidence
    """
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Make prediction
        result = inference_pipeline.predict_array(
            image_array,
            return_probs=True
        )
        
        logger.info(f"Prediction: {result['predicted_class']} "
                   f"(confidence: {result['confidence']:.4f})")
        
        return PredictionResponse(
            predicted_class=result['predicted_class'],
            confidence=float(result['confidence']),
            class_probabilities=result['class_probabilities'],
            message="Prediction successful"
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict disease classes for multiple uploaded images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of prediction results
    """
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per request")
    
    results = []
    
    for file in files:
        # Validate file type
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Make prediction
            result = inference_pipeline.predict_array(
                image_array,
                return_probs=True
            )
            
            results.append(PredictionResponse(
                predicted_class=result['predicted_class'],
                confidence=float(result['confidence']),
                class_probabilities=result['class_probabilities'],
                message="Prediction successful"
            ))
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append(PredictionResponse(
                predicted_class="error",
                confidence=0.0,
                class_probabilities={},
                message=f"Error: {str(e)}"
            ))
    
    return results


@app.get("/classes")
async def get_classes():
    """Get available disease classes."""
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": inference_pipeline.class_names,
        "num_classes": len(inference_pipeline.class_names)
    }


@app.get("/model/info")
async def get_model_info():
    """Get model information."""
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count parameters
    num_params = sum(p.numel() for p in inference_pipeline.model.parameters())
    trainable_params = sum(p.numel() for p in inference_pipeline.model.parameters() 
                          if p.requires_grad)
    
    return {
        "model_name": "SesameDiseaseCNN",
        "num_parameters": num_params,
        "trainable_parameters": trainable_params,
        "input_size": [299, 299, 3],
        "classes": inference_pipeline.class_names,
        "device": str(inference_pipeline.device)
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
