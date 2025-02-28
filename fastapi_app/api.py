from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import uvicorn
import joblib
import numpy as np
import os

app = FastAPI()

# Define the available models
MODEL_DIR = "models"
AVAILABLE_MODELS = {
    "decision_tree": os.path.join(MODEL_DIR, "decision_tree_model.pkl"),
    "random_forest": os.path.join(MODEL_DIR, "random_forest_model.pkl"),
}

# Load models into a dictionary
models = {}
for model_name, model_path in AVAILABLE_MODELS.items():
    if os.path.exists(model_path):
        try:
            models[model_name] = joblib.load(model_path)
            print(f"✅ Model '{model_name}' loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")

# Define input validation schema
class FeaturesInput(BaseModel):
    model_name: str  # The model to use
    features: list[float]  # List of numerical values

    @validator("model_name")
    def validate_model_name(cls, value):
        """Ensure the provided model name is available"""
        if value not in AVAILABLE_MODELS:
            raise ValueError(f"Model '{value}' is not available. Choose from {list(AVAILABLE_MODELS.keys())}")
        return value

    @validator("features")
    def validate_features(cls, value):
        """Ensure all features are numerical"""
        if not all(isinstance(i, (int, float)) for i in value):
            raise ValueError("All features must be numbers (int or float)")
        return value

@app.post("/predict/")
def predict(data: FeaturesInput):
    """Predict churn based on input features and selected model"""
    model_name = data.model_name

    # Check if model is loaded
    if model_name not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not loaded or missing.")

    # Make prediction (without reshaping input)
    prediction = models[model_name].predict([data.features])
    
    return {"model_used": model_name, "prediction": int(prediction[0])}


def start_fastapi():
    """Start FastAPI server."""
    print("🚀 Starting FastAPI server...")
    uvicorn.run("fastapi_app.api:app", host="0.0.0.0", port=8000, reload=True)
