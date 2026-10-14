from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import predict_consumer, load_artifacts, FEATURE_COLS

app = FastAPI(
    title="Electricity Theft Detection API",
    description="Detects abnormal electricity consumption using Random Forest + SHAP",
    version="1.0.0"
)

# Allow requests from HuggingFace Spaces and any browser (static frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your HF Space URL in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
model, scaler = load_artifacts(config["paths"]["models_dir"])

class ConsumerFeatures(BaseModel):
    mean_consumption:      float = Field(..., ge=0, description="Mean daily consumption (kWh)")
    std_consumption:       float = Field(..., ge=0, description="Std of daily consumption")
    max_consumption:       float = Field(..., ge=0, description="Max daily consumption")
    min_consumption:       float = Field(..., ge=0, description="Min daily consumption")
    cv_consumption:        float = Field(..., ge=0, description="Coefficient of variation")
    peak_offpeak_ratio:    float = Field(..., ge=0, description="Peak to off-peak ratio")
    weekend_weekday_ratio: float = Field(..., ge=0, description="Weekend to weekday ratio")
    mean_to_peak_ratio:    float = Field(..., ge=0, le=1, description="Mean to peak ratio")
    load_factor:           float = Field(..., ge=0, le=1, description="Load factor")
    sudden_change_count:   float = Field(..., ge=0, description="Count of sudden changes")
    rolling_mean_7d:       float = Field(..., ge=0, description="7-day rolling mean")
    rolling_std_7d:        float = Field(..., ge=0, description="7-day rolling std")

class PredictionResponse(BaseModel):
    prediction:   str
    probability:  float
    risk_score:   float
    risk_level:   str
    top_features: list

@app.get("/")
def root():
    return {"message": "Electricity Theft Detection API is running. Visit /docs for usage."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: ConsumerFeatures):
    try:
        result = predict_consumer(features.dict(), model, scaler, config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
