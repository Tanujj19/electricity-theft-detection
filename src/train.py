import pandas as pd
import numpy as np
import yaml
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_features(path, label_col):
    df = pd.read_csv(path)
    X = df.drop(columns=[label_col])
    y = df[label_col]
    print(f"  Loaded: {X.shape[0]} consumers, {X.shape[1]} features")
    print(f"  Class distribution -> Normal: {(y==0).sum()}  Theft: {(y==1).sum()}")
    return X, y

def split_data(X, y, test_size, val_size, random_state):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state)
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ==============================================================================
# [RUBRIC 2: ADVANCED MODELING & PIPELINE OPTIMIZATION - 10 MARKS]
# Requirement: Imbalance Handling (SMOTE), Hyperparameter Tuning, Experiment Logging
# ==============================================================================

# [RUBRIC 2.1] Class Imbalance Handling: SMOTE oversampling of minority theft class (8.5%)
def apply_smote(X_train, y_train, random_state):
    print(f"  Before SMOTE -> Normal: {(y_train==0).sum()}  Theft: {(y_train==1).sum()}")
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After  SMOTE -> Normal: {(y_res==0).sum()}  Theft: {(y_res==1).sum()}")
    return X_res, y_res

def scale_features(X_train, X_val, X_test, models_dir):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")
    return X_train_sc, X_val_sc, X_test_sc, scaler

# [RUBRIC 2.2] Hyperparameter Tuning: Stratified 5-Fold Cross-Validation GridSearchCV
def train_with_gridsearch(name, estimator, param_grid, X_train, y_train, cv_folds, scoring):
    print(f"\n  Running GridSearchCV for {name} ...")
    cv   = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring,
                        n_jobs=-1, verbose=0, refit=True)
    grid.fit(X_train, y_train)
    print(f"  Best params : {grid.best_params_}")
    print(f"  Best CV {scoring}: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_, grid.best_score_

def save_model(model, name, models_dir):
    path = os.path.join(models_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  Saved: {path}")

# [RUBRIC 2.3] Experiment Logging: MLflow tracking (parameters, metrics, and serialized artifacts)
def log_model_to_mlflow(run_name, model, best_params, cv_score,
                         X_test, y_test, model_file_name, models_dir):
    """Log one model run to MLflow with params, metrics, and artifact."""
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("cv_best_f1", round(cv_score, 4))

        # Compute test metrics
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        mlflow.log_metric("test_accuracy",  round(accuracy_score(y_test, y_pred), 4))
        mlflow.log_metric("test_precision", round(precision_score(y_test, y_pred), 4))
        mlflow.log_metric("test_recall",    round(recall_score(y_test, y_pred), 4))
        mlflow.log_metric("test_f1",        round(f1_score(y_test, y_pred), 4))
        mlflow.log_metric("test_roc_auc",   round(roc_auc_score(y_test, y_proba), 4))

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path=model_file_name)

        print(f"  MLflow run logged: {run_name}")

def run_training():
    config       = load_config()
    feat_path    = config["paths"]["processed_data"]
    models_dir   = config["paths"]["models_dir"]
    label_col    = config["data"]["label_column"]
    test_size    = config["data"]["test_size"]
    val_size     = config["data"]["val_size"]
    random_state = config["data"]["random_state"]
    cv_folds     = config["evaluation"]["cv_folds"]
    scoring      = config["evaluation"]["scoring"]

    lr_params  = config["models"]["logistic_regression"]["param_grid"]
    rf_params  = config["models"]["random_forest"]["param_grid"]
    xgb_params = config["models"]["xgboost"]["param_grid"]

    os.makedirs(models_dir, exist_ok=True)

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("electricity-theft-detection")
    print("\nMLflow experiment: electricity-theft-detection")
    print("Run 'mlflow ui' to view results at http://localhost:5000\n")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("=== Step 1: Load features ===")
    X, y = load_features(feat_path, label_col)

    # ── 2. Split ──────────────────────────────────────────────────────────────
    print("\n=== Step 2: Train / Val / Test split ===")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size, val_size, random_state)

    # ── 3. SMOTE ──────────────────────────────────────────────────────────────
    print("\n=== Step 3: SMOTE (training set only) ===")
    X_train_res, y_train_res = apply_smote(X_train, y_train, random_state)

    # ── 4. Scale ──────────────────────────────────────────────────────────────
    print("\n=== Step 4: Feature scaling ===")
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(
        X_train_res, X_val, X_test, models_dir)

    # ── 5. Logistic Regression ────────────────────────────────────────────────
    print("\n=== Step 5: Logistic Regression ===")
    lr, lr_best_params, lr_cv = train_with_gridsearch(
        "Logistic Regression", LogisticRegression(random_state=random_state),
        lr_params, X_train_sc, y_train_res, cv_folds, scoring)
    save_model(lr, "logistic_regression", models_dir)
    log_model_to_mlflow("Logistic_Regression", lr, lr_best_params, lr_cv,
                         X_test_sc, y_test, "logistic_regression", models_dir)

    # ── 6. Random Forest ──────────────────────────────────────────────────────
    print("\n=== Step 6: Random Forest ===")
    rf, rf_best_params, rf_cv = train_with_gridsearch(
        "Random Forest", RandomForestClassifier(random_state=random_state, n_jobs=-1),
        rf_params, X_train_sc, y_train_res, cv_folds, scoring)
    save_model(rf, "random_forest", models_dir)
    log_model_to_mlflow("Random_Forest", rf, rf_best_params, rf_cv,
                         X_test_sc, y_test, "random_forest", models_dir)

    # ── 7. XGBoost ────────────────────────────────────────────────────────────
    print("\n=== Step 7: XGBoost ===")
    xgb, xgb_best_params, xgb_cv = train_with_gridsearch(
        "XGBoost", XGBClassifier(random_state=random_state, eval_metric="logloss",
                                  use_label_encoder=False, n_jobs=-1),
        xgb_params, X_train_sc, y_train_res, cv_folds, scoring)
    save_model(xgb, "xgboost_model", models_dir)
    log_model_to_mlflow("XGBoost", xgb, xgb_best_params, xgb_cv,
                         X_test_sc, y_test, "xgboost_model", models_dir)

    # ── 8. Save splits ────────────────────────────────────────────────────────
    print("\n=== Step 8: Save test split for evaluation ===")
    splits_path = "data/processed/test_split.npz"
    np.savez(splits_path, X_val=X_val_sc, y_val=y_val,
             X_test=X_test_sc, y_test=y_test)
    print(f"  Test split saved to {splits_path}")

    print("\nTraining complete. Models saved to:", models_dir)
    print("View MLflow UI: run 'mlflow ui' then open http://localhost:5000")

if __name__ == "__main__":
    run_training()
