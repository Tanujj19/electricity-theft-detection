import numpy as np
import pandas as pd
import joblib
import yaml
import os
import warnings
warnings.filterwarnings("ignore")

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        import yaml
        return yaml.safe_load(f)

def load_artifacts(models_dir):
    model  = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    return model, scaler

FEATURE_COLS = [
    "mean_consumption", "std_consumption", "max_consumption", "min_consumption",
    "cv_consumption", "peak_offpeak_ratio", "weekend_weekday_ratio",
    "mean_to_peak_ratio", "load_factor", "sudden_change_count",
    "rolling_mean_7d", "rolling_std_7d"
]

def compute_risk_level(probability, config):
    low  = config["risk_score"]["low_threshold"]
    med  = config["risk_score"]["medium_threshold"]
    if probability < low:
        return "Low"
    elif probability < med:
        return "Medium"
    return "High"

def predict_consumer(features: dict, model, scaler, config):
    """
    features: dict with keys matching FEATURE_COLS
    Returns prediction dict with label, probability, risk score, risk level, top features.
    """
    X = pd.DataFrame([features])[FEATURE_COLS]
    X_sc = scaler.transform(X)

    proba      = model.predict_proba(X_sc)[0][1]
    prediction = int(proba >= 0.5)
    label      = "Potential Electricity Theft" if prediction == 1 else "Normal Consumer"
    risk_score = round(proba * 100, 1)
    risk_level = compute_risk_level(proba, config)

    # top 3 features by RF feature importance
    importances = model.feature_importances_
    top_idx     = np.argsort(importances)[::-1][:3]
    top_features = [FEATURE_COLS[i] for i in top_idx]

    return {
        "prediction":   label,
        "probability":  round(float(proba), 4),
        "risk_score":   risk_score,
        "risk_level":   risk_level,
        "top_features": top_features
    }

if __name__ == "__main__":
    config = load_config()
    model, scaler = load_artifacts(config["paths"]["models_dir"])

    # Example: a suspicious consumer
    sample = {
        "mean_consumption":      25.0,
        "std_consumption":       18.0,
        "max_consumption":       80.0,
        "min_consumption":        0.5,
        "cv_consumption":         0.72,
        "peak_offpeak_ratio":    40.0,
        "weekend_weekday_ratio":  0.85,
        "mean_to_peak_ratio":     0.31,
        "load_factor":            0.31,
        "sudden_change_count":   90.0,
        "rolling_mean_7d":       28.0,
        "rolling_std_7d":         9.0,
    }

    result = predict_consumer(sample, model, scaler, config)
    print("\n--- Prediction Result ---")
    for k, v in result.items():
        print(f"  {k}: {v}")
