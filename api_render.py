"""
api_render.py — Self-contained FastAPI for Render cloud deployment.
Loads the pre-trained Random Forest model and scaler from models/.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="Electricity Theft Detection API",
    description="Detects abnormal electricity consumption using Random Forest + SHAP | SGCC Dataset | B III M.Sc. AI & ML",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_COLS = [
    "mean_consumption", "std_consumption", "max_consumption", "min_consumption",
    "cv_consumption", "peak_offpeak_ratio", "weekend_weekday_ratio",
    "mean_to_peak_ratio", "load_factor", "sudden_change_count",
    "rolling_mean_7d", "rolling_std_7d",
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
model  = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


class ConsumerFeatures(BaseModel):
    mean_consumption:      float = Field(..., ge=0, example=7.0)
    std_consumption:       float = Field(..., ge=0, example=4.0)
    max_consumption:       float = Field(..., ge=0, example=28.0)
    min_consumption:       float = Field(..., ge=0, example=0.5)
    cv_consumption:        float = Field(..., ge=0, example=1.46)
    peak_offpeak_ratio:    float = Field(..., ge=0, example=15.0)
    weekend_weekday_ratio: float = Field(..., ge=0, example=1.3)
    mean_to_peak_ratio:    float = Field(..., ge=0, le=1, example=0.25)
    load_factor:           float = Field(..., ge=0, le=1, example=0.25)
    sudden_change_count:   float = Field(..., ge=0, example=115.0)
    rolling_mean_7d:       float = Field(..., ge=0, example=6.5)
    rolling_std_7d:        float = Field(..., ge=0, example=1.3)


class PredictionResponse(BaseModel):
    prediction:   str
    probability:  float
    risk_score:   float
    risk_level:   str
    top_features: list


@app.get("/")
def root():
    return {
        "message": "Electricity Theft Detection API",
        "docs": "/docs",
        "status": "running",
        "model": "Random Forest (F1=0.3254, ROC-AUC=0.7703)",
        "dataset": "SGCC — 42,372 consumers × 1,035 daily readings",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ConsumerFeatures):
    try:
        X = np.array([[getattr(features, col) for col in FEATURE_COLS]])
        X_scaled = scaler.transform(X)
        proba = float(model.predict_proba(X_scaled)[0][1])
        risk_score = round(proba * 100, 1)

        if proba < 0.4:
            risk_level = "Low"
        elif proba < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        prediction = "Potential Electricity Theft" if proba >= 0.5 else "Normal Consumer"
        top_features = [FEATURE_COLS[i] for i in np.argsort(model.feature_importances_)[::-1][:3]]

        return PredictionResponse(
            prediction=prediction,
            probability=proba,
            risk_score=risk_score,
            risk_level=risk_level,
            top_features=top_features,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
