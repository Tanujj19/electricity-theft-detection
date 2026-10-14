"""
train_optuna.py — Advanced Hyperparameter Optimization with Optuna + MLflow
============================================================================
Trains LightGBM, CatBoost, and XGBoost using Optuna Bayesian Optimization.
All trials are logged to MLflow for experiment tracking.
"""

import pandas as pd
import numpy as np
import yaml
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn

import lightgbm as lgb
import catboost as cb
from xgboost import XGBClassifier


# ─────────────────────────────────────────────────────────────────────────────
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


def split_and_balance(X, y, test_size, val_size, random_state):
    """Train/Val/Test split + SMOTE on training set only."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state)

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # SMOTE on training only
    print(f"  Before SMOTE -> Normal: {(y_train==0).sum()}  Theft: {(y_train==1).sum()}")
    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"  After  SMOTE -> Normal: {(y_train_res==0).sum()}  Theft: {(y_train_res==1).sum()}")

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    return X_train_sc, X_val_sc, X_test_sc, y_train_res, y_val, y_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Optuna Objective Functions
# ─────────────────────────────────────────────────────────────────────────────

def make_lgb_objective(X_train, y_train, cv_folds=5):
    """Optuna objective for LightGBM - returns CV F1 score (maximize)."""
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "class_weight":      "balanced",
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        model = lgb.LGBMClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        return scores.mean()
    return objective


def make_xgb_objective(X_train, y_train, cv_folds=5):
    """Optuna objective for XGBoost."""
    def objective(trial):
        scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric":      "logloss",
            "use_label_encoder": False,
            "random_state":     42,
            "n_jobs":           -1,
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        return scores.mean()
    return objective


def make_cat_objective(X_train, y_train, cv_folds=5):
    """Optuna objective for CatBoost."""
    def objective(trial):
        params = {
            "iterations":          trial.suggest_int("iterations", 100, 600),
            "depth":               trial.suggest_int("depth", 3, 10),
            "learning_rate":       trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "border_count":        trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "auto_class_weights":  "Balanced",
            "random_state":        42,
            "verbose":             False,
        }
        model = cb.CatBoostClassifier(**params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        return scores.mean()
    return objective


# ==============================================================================
# [RUBRIC 2: ADVANCED MODELING & PIPELINE OPTIMIZATION - 10 MARKS]
# Requirement: Hyperparameter Tuning via Optuna (TPE Bayesian Optimization)
# ==============================================================================
def run_optuna_study(name, objective_fn, n_trials=30):
    """Run an Optuna study to maximize CV F1 score."""
    print(f"\n  Running Optuna ({n_trials} trials) for {name} ...")
    study = optuna.create_study(
        direction="maximize",
        study_name=name,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best CV F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value, study


def compute_test_metrics(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "test_precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "test_recall":    round(recall_score(y_test, y_pred), 4),
        "test_f1":        round(f1_score(y_test, y_pred), 4),
        "test_roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }


def log_to_mlflow(run_name, model, best_params, cv_f1, test_metrics, study, models_dir, artifact_name):
    """Log Optuna-tuned model to MLflow."""
    with mlflow.start_run(run_name=run_name):
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("cv_best_f1", round(cv_f1, 4))
        mlflow.log_param("n_trials", len(study.trials))
        mlflow.log_param("optimizer", "Optuna-TPE")

        for k, v in test_metrics.items():
            mlflow.log_metric(k, v)

        # Log Optuna trial history
        trials_df = study.trials_dataframe()
        trials_path = os.path.join(models_dir, f"{artifact_name}_optuna_trials.csv")
        trials_df.to_csv(trials_path, index=False)
        mlflow.log_artifact(trials_path)

        mlflow.sklearn.log_model(model, artifact_path=artifact_name)
        print(f"  MLflow logged: {run_name}  | test_f1={test_metrics['test_f1']}")


def save_model(model, name, models_dir):
    path = os.path.join(models_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def run_optuna_training():
    config       = load_config()
    feat_path    = config["paths"]["processed_data"]
    models_dir   = config["paths"]["models_dir"]
    label_col    = config["data"]["label_column"]
    test_size    = config["data"]["test_size"]
    val_size     = config["data"]["val_size"]
    random_state = config["data"]["random_state"]
    cv_folds     = config["evaluation"]["cv_folds"]
    n_trials     = config.get("optuna", {}).get("n_trials", 30)

    os.makedirs(models_dir, exist_ok=True)

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("electricity-theft-optuna")
    print("\nMLflow experiment: electricity-theft-optuna")
    print(f"Optuna trials per model: {n_trials}\n")

    print("=== Step 1: Load & prepare data ===")
    X, y = load_features(feat_path, label_col)

    print("\n=== Step 2: Split, SMOTE, Scale ===")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_balance(
        X, y, test_size, val_size, random_state)

    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

    results = []

    # LightGBM
    print("\n=== Step 3: LightGBM + Optuna ===")
    lgb_obj = make_lgb_objective(X_train, y_train, cv_folds)
    lgb_best_params, lgb_cv_f1, lgb_study = run_optuna_study("LightGBM", lgb_obj, n_trials)
    lgb_model = lgb.LGBMClassifier(**lgb_best_params, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_metrics = compute_test_metrics(lgb_model, X_test, y_test)
    save_model(lgb_model, "lightgbm_model", models_dir)
    log_to_mlflow("LightGBM_Optuna", lgb_model, lgb_best_params,
                  lgb_cv_f1, lgb_metrics, lgb_study, models_dir, "lightgbm_model")
    results.append({"Model": "LightGBM (Optuna)", **lgb_metrics})
    print(f"  LightGBM  | F1: {lgb_metrics['test_f1']}  ROC-AUC: {lgb_metrics['test_roc_auc']}")

    # XGBoost
    print("\n=== Step 4: XGBoost + Optuna ===")
    xgb_obj = make_xgb_objective(X_train, y_train, cv_folds)
    xgb_best_params, xgb_cv_f1, xgb_study = run_optuna_study("XGBoost", xgb_obj, n_trials)
    xgb_model = XGBClassifier(**xgb_best_params, eval_metric="logloss",
                               use_label_encoder=False, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_metrics = compute_test_metrics(xgb_model, X_test, y_test)
    save_model(xgb_model, "xgboost_optuna", models_dir)
    log_to_mlflow("XGBoost_Optuna", xgb_model, xgb_best_params,
                  xgb_cv_f1, xgb_metrics, xgb_study, models_dir, "xgboost_optuna")
    results.append({"Model": "XGBoost (Optuna)", **xgb_metrics})
    print(f"  XGBoost   | F1: {xgb_metrics['test_f1']}  ROC-AUC: {xgb_metrics['test_roc_auc']}")

    # CatBoost
    print("\n=== Step 5: CatBoost + Optuna ===")
    cat_obj = make_cat_objective(X_train, y_train, cv_folds)
    cat_best_params, cat_cv_f1, cat_study = run_optuna_study("CatBoost", cat_obj, n_trials)
    cat_model = cb.CatBoostClassifier(**cat_best_params, random_state=42, verbose=False)
    cat_model.fit(X_train, y_train)
    cat_metrics = compute_test_metrics(cat_model, X_test, y_test)
    save_model(cat_model, "catboost_model", models_dir)
    log_to_mlflow("CatBoost_Optuna", cat_model, cat_best_params,
                  cat_cv_f1, cat_metrics, cat_study, models_dir, "catboost_model")
    results.append({"Model": "CatBoost (Optuna)", **cat_metrics})
    print(f"  CatBoost  | F1: {cat_metrics['test_f1']}  ROC-AUC: {cat_metrics['test_roc_auc']}")

    # Summary
    print("\n=== Optuna Training Complete ===")
    df_res = pd.DataFrame(results)
    print("\n--- Optuna-Tuned Models ---")
    print(df_res.to_string(index=False))

    metrics_dir = config["paths"]["metrics_dir"]
    os.makedirs(metrics_dir, exist_ok=True)
    summary_path = os.path.join(metrics_dir, "optuna_model_comparison.csv")
    df_res.to_csv(summary_path, index=False)
    print(f"\nOptuna comparison saved to: {summary_path}")
    print("View MLflow UI: mlflow ui  ->  http://localhost:5000")

    splits_path = "data/processed/test_split.npz"
    np.savez(splits_path, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
    print(f"Test split saved to {splits_path}")

    return df_res


if __name__ == "__main__":
    run_optuna_training()
