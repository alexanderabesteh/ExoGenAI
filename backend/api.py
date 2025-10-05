"""
FastAPI backend for ExoGenAI exoplanet detection system
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import numpy as np
import io
from hybrid_pipeline import HybridExoplanetDetector
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="ExoGenAI API",
    description="Hybrid AI system for exoplanet detection from light curves",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector (load model once at startup)
detector = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global detector
    model_path = Path("checkpoints/best_model.pt")
    
    if not model_path.exists():
        print("WARNING: Model not found. API will run in demo mode.")
        detector = HybridExoplanetDetector(device='cpu')
    else:
        print(f"Loading model from {model_path}...")
        detector = HybridExoplanetDetector(
            model_path=str(model_path),
            device='cuda'  # Change to 'cpu' if no GPU
        )
        print("Model loaded successfully!")


# Response models
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float
    pipeline_details: Dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model_loaded": detector is not None and detector.ml_model is not None,
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return {
        "status": "online",
        "model_loaded": detector is not None and detector.ml_model is not None,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict exoplanet from uploaded light curve CSV
    
    Expected CSV format:
    - Two columns: time, flux
    - OR: Single column with flux values
    - OR: First row is label, rest are flux values (Kaggle format)
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read CSV file
        contents = await file.read()
        
        # Try to parse as CSV
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
        # Extract flux data
        # Case 1: Has 'flux' column
        if 'flux' in df.columns:
            flux = df['flux'].values
        # Case 2: Has 'FLUX' column
        elif 'FLUX' in df.columns:
            flux = df['FLUX'].values
        # Case 3: Two columns (assume second is flux)
        elif len(df.columns) == 2:
            flux = df.iloc[:, 1].values
        # Case 4: One column
        elif len(df.columns) == 1:
            flux = df.iloc[:, 0].values
        # Case 5: Many columns (Kaggle format - first column is label, rest is flux)
        else:
            # Skip first column (label) if it exists
            flux = df.iloc[0, 1:].values
        
        # Remove NaNs
        flux = flux[~np.isnan(flux)]
        
        if len(flux) == 0:
            raise HTTPException(status_code=400, detail="No valid flux data found")
        
        # Run detection
        result = detector.detect(flux)
        
        # Format response
        return {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities'],
            "processing_time_ms": result['processing_time_ms'],
            "pipeline_details": {
                "rule_filter": result['stage_1_filter'],
                "ml_model": result['stage_2_ml'],
                "final_decision": result['final_decision']
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-with-plot")
async def predict_with_plot(file: UploadFile = File(...)):
    """
    Predict exoplanet and return data for plotting
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Extract flux
        if 'flux' in df.columns:
            time = df['time'].values if 'time' in df.columns else np.arange(len(df))
            flux = df['flux'].values
        elif len(df.columns) == 2:
            time = df.iloc[:, 0].values
            flux = df.iloc[:, 1].values
        else:
            flux = df.iloc[0, 1:].values
            time = np.arange(len(flux))
        
        flux_clean = flux[~np.isnan(flux)]
        
        # Run detection
        result = detector.detect(flux_clean)
        
        # Return prediction + plot data
        return {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "probabilities": result['probabilities'],
            "processing_time_ms": result['processing_time_ms'],
            "plot_data": {
                "time": time.tolist(),
                "flux": flux.tolist()
            },
            "pipeline_details": {
                "rule_filter": result['stage_1_filter'],
                "ml_model": result['stage_2_ml']
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """
    Batch prediction for multiple light curves
    
    Expected CSV format: Each row is a separate light curve
    First column can be label (optional), rest are flux values
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        results = []
        
        # Process each row
        for idx, row in df.iterrows():
            # Skip first column if it looks like a label
            if row.iloc[0] in [1, 2]:  # Kaggle format labels
                flux = row.iloc[1:].values
            else:
                flux = row.values
            
            flux = flux[~np.isnan(flux)]
            
            if len(flux) > 0:
                result = detector.detect(flux)
                results.append({
                    "index": idx,
                    "prediction": result['prediction'],
                    "confidence": result['confidence'],
                    "probabilities": result['probabilities']
                })
        
        return {
            "total_processed": len(results),
            "results": results,
            "statistics": detector.get_stats()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return detector.get_stats()


@app.post("/reset-stats")
async def reset_stats():
    """Reset pipeline statistics"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    detector.reset_stats()
    return {"message": "Statistics reset successfully"}


# Development/testing endpoints
@app.get("/test")
async def test():
    """Test endpoint with synthetic data"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Generate synthetic light curve with transit
    flux = np.ones(2048)
    for t0 in [500, 1000, 1500]:
        transit_mask = np.abs(np.arange(2048) - t0) < 20
        flux[transit_mask] -= 0.01
    flux += np.random.normal(0, 0.001, 2048)
    
    result = detector.detect(flux)
    
    return {
        "message": "Test prediction on synthetic transit",
        "result": result
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting ExoGenAI API server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )