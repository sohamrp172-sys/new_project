"""
FastAPI application for model serving and predictions.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from typing import Optional

from src.models.registry import ModelRegistry

# Global model variable
model = None
model_features = None
model_n_features = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global model, model_n_features

    registry = ModelRegistry()
    best_path = registry.get_best_model()
    if not best_path or not os.path.exists(best_path):
        print(f"Warning: No best model found in registry or path invalid: {best_path}")
        print("API will start but predictions may fail without a trained model")
    else:
        try:
            model = joblib.load(best_path)
            model_n_features = model.n_features_in_
            print(f"✓ Model loaded from registry at {best_path} (expects {model_n_features} features)")
        except Exception as e:
            print(f"Error loading model: {e}")
    yield
    # shutdown: nothing to clean up


# Initialize FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts customer churn using trained ML models",
    version="1.0.0",
    lifespan=lifespan
)


class PredictionInput(BaseModel):
    """Request schema for predictions.
    
    Accepts the same 28 features the model was trained on.
    All fields except the core three are optional and default to 0.
    """
    # Core features (always required)
    tenure: float = 0.0
    monthlycharges: float = 0.0
    totalcharges: float = 0.0
    # Remaining 25 engineered/selected features (optional, default 0)
    contract: float = 0.0
    maritalstatus: float = 0.0
    prizmcode: float = 0.0
    num_complaints: float = 0.0
    customerid: float = 0.0
    homeownership: float = 0.0
    hascreditcard: float = 0.0
    agehh2: float = 0.0
    customerid_squared: float = 0.0
    childreninhh: float = 0.0
    buysviamailorder: float = 0.0
    respondstomailoffers: float = 0.0
    uniquesubs: float = 0.0
    customer_satisfaction: float = 0.0
    newcellphoneuser: float = 0.0
    activesubs: float = 0.0
    servicearea: float = 0.0
    num_service_calls: float = 0.0
    handsetwebcapable: float = 0.0
    handsetprice: float = 0.0
    monthlyrevenue: float = 0.0
    retentioncalls: float = 0.0
    currentequipmentdays: float = 0.0
    agehh1: float = 0.0

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 24,
                "monthlycharges": 65.50,
                "totalcharges": 1574.50
            }
        }
    }


class PredictionOutput(BaseModel):
    """Response schema for predictions."""
    churn_probability: float
    churn_prediction: int
    prediction_label: str


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_n_features": model_n_features
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make prediction for customer churn.
    
    Args:
        input_data: Customer data for prediction
        
    Returns:
        Prediction with probability and label
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train and save the model first."
        )
    
    try:
        # Build feature vector in the same order as training
        feature_order = [
            'contract', 'maritalstatus', 'prizmcode', 'num_complaints', 'customerid',
            'homeownership', 'hascreditcard', 'agehh2', 'customerid_squared', 'childreninhh',
            'buysviamailorder', 'respondstomailoffers', 'uniquesubs', 'customer_satisfaction',
            'newcellphoneuser', 'activesubs', 'servicearea', 'num_service_calls',
            'handsetwebcapable', 'handsetprice', 'monthlyrevenue', 'retentioncalls',
            'currentequipmentdays', 'agehh1', 'tenure', 'monthlycharges', 'totalcharges',
            'customerid_squared'
        ]

        data_dict = input_data.model_dump()
        # Build array matching model's n_features_in_
        features = np.array([[data_dict.get(f, 0.0) for f in feature_order[:model_n_features]]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Return response
        return PredictionOutput(
            churn_probability=float(probability),
            churn_prediction=int(prediction),
            prediction_label="Churn" if prediction == 1 else "No Churn"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model-info")
async def model_info():
    """Get information about loaded model."""
    if model is None:
        return {"model_loaded": False, "message": "No model loaded"}

    registry = ModelRegistry()
    best_path = registry.get_best_model()
    return {
        "model_loaded": True,
        "model_type": type(model).__name__,
        "model_path": best_path or "unknown",
        "n_features": model_n_features
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Telco Churn Prediction API...")
    print("API will be available at http://localhost:8000")
    print("Docs available at http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
