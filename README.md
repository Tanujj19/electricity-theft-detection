# ⚡ Explainable Electricity Theft Detection — Advanced ML System

> **B III M.Sc. Artificial Intelligence & Machine Learning**
> Advanced Machine Learning Lab Project

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-orange)](https://mlflow.org)
[![Optuna](https://img.shields.io/badge/Optuna-3.5-purple)](https://optuna.org)

---

## Overview

This project builds an advanced Machine Learning system for detecting electricity theft in smart grid consumer data. It uses the **SGCC (State Grid Corporation of China)** dataset — 42,372 consumers with 1,035 daily readings each — and classifies consumers as **Normal** or **Potential Electricity Theft**.

### Key Features

- ✅ **6 ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM (Optuna), XGBoost (Optuna), CatBoost (Optuna)
- ✅ **Optuna Bayesian HPO** (TPE sampler, 30 trials per model)
- ✅ **GridSearchCV** (baseline hyperparameter tuning)
- ✅ **MLflow Experiment Tracking** (parameters, metrics, artifacts)
- ✅ **SMOTE** class imbalance correction (training set only)
- ✅ **SHAP Explainability** (global + individual consumer level)
- ✅ **16 Visualizations** (EDA, ROC/PR curves, PCA, t-SNE, SHAP plots)
- ✅ **FastAPI** production inference API with Pydantic validation
- ✅ **Streamlit** multi-tab web UI
- ✅ **Gradio** app for Hugging Face Spaces
- ✅ **Docker** + **docker-compose** containerization
- ✅ **Benchmarked** against 3 published papers on the SGCC dataset

---

## Project Structure

```
electricity-theft-detection/
│
├── src/
│   ├── data_preprocessing.py   # Missing values, outlier capping, deduplication
│   ├── feature_engineering.py  # 12 statistical features from raw time-series
│   ├── train.py                # Baseline models (LR + RF + XGBoost + GridSearchCV)
│   ├── train_optuna.py         # Advanced Optuna HPO (LightGBM + XGBoost + CatBoost)
│   ├── evaluate.py             # Metrics, ROC/PR curves, confusion matrix, comparison
│   ├── explain.py              # SHAP global + individual explanations
│   └── predict.py              # Single consumer prediction + risk scoring
│
├── notebooks/
│   └── eda.py                  # 10 EDA plots + PCA + t-SNE
│
├── api/
│   └── main.py                 # FastAPI REST API
│
├── app/
│   └── app.py                  # Streamlit Web UI (3 tabs)
│
├── huggingface_space/
│   ├── app.py                  # Gradio app for HuggingFace Spaces
│   └── ...                     # Models + results bundled
│
├── data/
│   ├── raw/                    # data set.csv (SGCC)
│   └── processed/              # cleaned_data.csv, features.csv, test_split.npz
│
├── models/                     # Saved .pkl model files
├── results/
│   ├── figures/                # All 16 visualization plots
│   └── metrics/                # model_comparison.csv, benchmark_comparison.csv
│
├── config/
│   └── config.yaml             # Centralized configuration (paths, hyperparams, Optuna)
│
├── report/
│   └── IEEE_Report.md          # Complete 10-page IEEE-formatted project report
│
├── run_pipeline.py             # Full end-to-end pipeline orchestrator
├── requirements.txt            # All Python dependencies
├── Dockerfile                  # FastAPI container
├── Dockerfile.ui               # Streamlit container
├── docker-compose.yml          # Multi-service deployment
└── README.md
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | SGCC (Kaggle: `sgcc-electricity-theft-detection`) |
| Consumers | 42,372 |
| Daily readings per consumer | 1,035 (2014–2016) |
| Label | FLAG: 0 = Normal, 1 = Theft |
| Class split | ~91.5% Normal, ~8.5% Theft |
| After preprocessing | 29,888 consumers |

---

## Features Engineered

| Feature | Description |
|---------|-------------|
| `mean_consumption` | Average daily usage (kWh) |
| `std_consumption` | Day-to-day standard deviation |
| `max_consumption` | Peak usage |
| `min_consumption` | Minimum usage |
| `cv_consumption` | Coefficient of variation (σ/μ) — erratic usage flag |
| `peak_offpeak_ratio` | 90th percentile / 10th percentile ratio |
| `weekend_weekday_ratio` | Weekend vs. weekday average |
| `mean_to_peak_ratio` | Mean / Max (load consistency) |
| `load_factor` | Same as mean-to-peak (grid efficiency metric) |
| `sudden_change_count` | Days with >50% day-over-day change |
| `rolling_mean_7d` | 7-day rolling mean (recent trend) |
| `rolling_std_7d` | 7-day rolling std (recent volatility) |

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|-----|---------|--------|
| Logistic Regression | 0.7467 | 0.1820 | 0.6167 | 0.2810 | 0.7666 | 0.2532 |
| Random Forest | 0.8548 | 0.2595 | 0.4361 | 0.3254 | 0.7703 | 0.2846 |
| XGBoost | 0.8272 | 0.2288 | 0.4861 | 0.3111 | 0.7588 | 0.2984 |
| ⚡ LightGBM (Optuna) | **0.8621** | 0.2743 | 0.4512 | **0.3415** | **0.7834** | **0.3102** |
| ⚡ XGBoost (Optuna) | 0.8489 | 0.2601 | 0.4698 | 0.3356 | 0.7791 | 0.3089 |
| ⚡ CatBoost (Optuna) | 0.8576 | 0.2667 | 0.4583 | 0.3381 | 0.7812 | 0.3071 |

### Published Benchmark Comparison

| Model | Source | Accuracy | F1 | ROC-AUC |
|-------|--------|----------|----|---------|
| SVM | Zheng et al. 2018 | 0.82 | 0.394 | 0.75 |
| Random Forest | Nagi et al. 2019 | 0.85 | 0.420 | 0.78 |
| XGBoost | Hu et al. 2020 | 0.87 | 0.450 | 0.81 |
| **Our RF (Baseline)** | This Work | **0.855** | 0.325 | 0.770 |
| **⚡ Our LightGBM (Optuna)** | This Work | **0.862** | **0.342** | **0.783** |

> **Best Model: ⚡ LightGBM (Optuna)** — Highest F1 (0.3415) and ROC-AUC (0.7834) after 30 Bayesian optimization trials
> **Best Baseline: Random Forest** — Highest F1 (0.3254) and ROC-AUC (0.7703) via GridSearchCV

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Full Pipeline (Recommended)

```bash
# Run all stages: preprocess → feature eng → EDA → train → optuna → evaluate → explain
python run_pipeline.py

# Skip Optuna (faster, uses GridSearch only)
python run_pipeline.py --skip-optuna

# Only run evaluation & SHAP (models already trained)
python run_pipeline.py --only-eval
```

### 3. Run Individual Stages

```bash
python src/data_preprocessing.py     # Stage 1: Data cleaning
python src/feature_engineering.py   # Stage 2: Feature extraction
python notebooks/eda.py              # Stage 3: EDA plots
python src/train.py                  # Stage 4: Baseline models (GridSearchCV)
python src/train_optuna.py           # Stage 5: Optuna HPO (LightGBM/XGBoost/CatBoost)
python src/evaluate.py               # Stage 6: Metrics & comparison
python src/explain.py                # Stage 7: SHAP explainability
```

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

### 5. Start the Streamlit UI

```bash
streamlit run app/app.py
```

Visit: [http://localhost:8501](http://localhost:8501)

### 6. View MLflow Dashboard

```bash
mlflow ui
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## API Reference

**POST** `/predict`

**Request:**
```json
{
  "mean_consumption": 25.0,
  "std_consumption": 18.0,
  "max_consumption": 80.0,
  "min_consumption": 0.5,
  "cv_consumption": 0.72,
  "peak_offpeak_ratio": 40.0,
  "weekend_weekday_ratio": 0.85,
  "mean_to_peak_ratio": 0.31,
  "load_factor": 0.31,
  "sudden_change_count": 90.0,
  "rolling_mean_7d": 28.0,
  "rolling_std_7d": 9.0
}
```

**Response:**
```json
{
  "prediction": "Potential Electricity Theft",
  "probability": 0.87,
  "risk_score": 87.0,
  "risk_level": "High",
  "top_features": ["sudden_change_count", "cv_consumption", "peak_offpeak_ratio"]
}
```

---

## Docker Deployment

```bash
# API only
docker build -t theft-detection-api .
docker run -p 8000:8000 theft-detection-api

# Full stack (API + UI)
docker-compose up --build

# Cleanup
docker-compose down
```

Services:
- API: [http://localhost:8000](http://localhost:8000)
- UI:  [http://localhost:8501](http://localhost:8501)

---

## 📸 Sample Output

### API Prediction Response
```json
POST /predict
→ Input: high cv_consumption=1.46, sudden_change_count=115, peak_offpeak_ratio=15

{
  "prediction": "Potential Electricity Theft",
  "probability": 0.87,
  "risk_score": 87,
  "risk_level": "High",
  "top_features": ["sudden_change_count", "cv_consumption", "peak_offpeak_ratio"]
}
```

### Generated Visualizations
| Plot | Description |
|------|-------------|
| `01_class_distribution.png` | Normal vs Theft class split |
| `03_feature_boxplots.png` | Feature separation between classes |
| `09_roc_curves.png` | ROC curves for all 6 models |
| `12_shap_global_bar.png` | SHAP global feature importance |
| `14_shap_waterfall_theft.png` | Individual consumer SHAP explanation |
| `15_pca_projection.png` | 2D PCA projection of all consumers |
| `16_tsne_projection.png` | 2D t-SNE projection (n=3000) |
| `17_error_analysis.png` | FP/FN/FPR/FNR error analysis |

> All 19 plots saved to `results/figures/`

---

## Technology Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Language | Python 3.10 | Core |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost | Models |
| HPO | Optuna (TPE) | Bayesian Optimization |
| Imbalance | imbalanced-learn (SMOTE) | Class balancing |
| Explainability | SHAP | Feature importance |
| Tracking | MLflow | Experiment logging |
| API | FastAPI + Uvicorn | Inference endpoint |
| UI | Streamlit | Local web UI |
| Cloud UI | Gradio | Hugging Face Spaces |
| Container | Docker + docker-compose | Deployment |
| Config | PyYAML | Configuration management |

---

## SHAP Explainability

Three levels of explanation are provided:

1. **Global Bar Plot** — Mean |SHAP| per feature (most important features overall)
2. **Beeswarm Summary** — Feature value vs. SHAP contribution for 500 consumers
3. **Individual Waterfall** — Why a specific consumer was flagged as theft

Top features by SHAP importance:
1. `sudden_change_count` — Abrupt usage changes
2. `cv_consumption` — Erratic usage patterns
3. `peak_offpeak_ratio` — Extreme peak variation

---

## Evaluation Metrics

| Metric | Description | Why it matters |
|--------|-------------|----------------|
| F1-Score | Harmonic mean of Precision & Recall | Primary metric for imbalanced classification |
| ROC-AUC | Area under ROC curve | Model ranking ability |
| PR-AUC | Area under Precision-Recall curve | Better for imbalanced classes |
| Recall | True theft detection rate | Critical for utility investigators |
| Precision | Accuracy of theft flags | Reduces false inspection costs |

---

## Author

B III M.Sc. Artificial Intelligence & Machine Learning
Advanced Machine Learning Lab Project


---

## Project Overview

This project builds a machine learning system that detects abnormal electricity
consumption patterns and classifies consumers as Normal or Potential Electricity Theft.
It uses ensemble learning (Random Forest, XGBoost) with SHAP-based explainability,
a FastAPI backend, and a Streamlit UI.

---

## Dataset

- **Source:** SGCC (State Grid Corporation of China) — Kaggle
- **Size:** 42,372 consumers × 1,035 daily readings
- **Labels:** 0 = Normal, 1 = Theft (~8.5% theft)
- **After cleaning:** 29,888 consumers

---

## Project Structure

```
electricity-theft-detection/
├── data/
│   ├── raw/                  <- original data set.csv
│   └── processed/            <- cleaned_data.csv, features.csv, test_split.npz
├── notebooks/
│   └── eda.py                <- EDA script (8 visualizations)
├── src/
│   ├── data_preprocessing.py <- cleaning, missing values, outliers
│   ├── feature_engineering.py<- 12 features per consumer
│   ├── train.py              <- SMOTE, scaling, GridSearchCV, 3 models
│   ├── evaluate.py           <- metrics, confusion matrix, ROC, PR curves
│   ├── explain.py            <- SHAP global + individual explanations
│   └── predict.py            <- single consumer prediction + risk score
├── api/
│   └── main.py               <- FastAPI REST API
├── app/
│   └── app.py                <- Streamlit UI
├── models/                   <- saved .pkl files
├── results/
│   ├── figures/              <- all 14 plots
│   └── metrics/              <- model_comparison.csv
├── config/
│   └── config.yaml           <- centralized configuration
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Features Engineered

| Feature | Description |
|---------|-------------|
| mean_consumption | Average daily usage |
| std_consumption | Day-to-day variability |
| max_consumption | Peak usage |
| min_consumption | Lowest usage |
| cv_consumption | Coefficient of variation (std/mean) |
| peak_offpeak_ratio | Top 10% vs bottom 10% days |
| weekend_weekday_ratio | Weekend vs weekday usage |
| mean_to_peak_ratio | How close average is to peak |
| load_factor | Mean / Max (efficiency measure) |
| sudden_change_count | Days with >50% change from previous day |
| rolling_mean_7d | 7-day rolling average |
| rolling_std_7d | 7-day rolling standard deviation |

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.747 | 0.182 | 0.617 | 0.281 | 0.767 |
| Random Forest | 0.855 | 0.260 | 0.436 | 0.325 | 0.770 |
| XGBoost | 0.827 | 0.229 | 0.486 | 0.311 | 0.759 |

**Best Model: Random Forest** (highest F1 and ROC-AUC)

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python src/data_preprocessing.py
python src/feature_engineering.py
python notebooks/eda.py
python src/train.py
python src/evaluate.py
python src/explain.py
```

### 3. Start the FastAPI server
```bash
uvicorn api.main:app --reload --port 8000
```
Visit: http://localhost:8000/docs

### 4. Start the Streamlit UI
```bash
streamlit run app/app.py
```
Visit: http://localhost:8501

### 5. Run with Docker
```bash
docker build -t theft-detection .
docker run -p 8000:8000 theft-detection
```

---

## API Usage

**POST** `/predict`

```json
{
  "mean_consumption": 25.0,
  "std_consumption": 18.0,
  "max_consumption": 80.0,
  "min_consumption": 0.5,
  "cv_consumption": 0.72,
  "peak_offpeak_ratio": 40.0,
  "weekend_weekday_ratio": 0.85,
  "mean_to_peak_ratio": 0.31,
  "load_factor": 0.31,
  "sudden_change_count": 90.0,
  "rolling_mean_7d": 28.0,
  "rolling_std_7d": 9.0
}
```

**Response:**
```json
{
  "prediction": "Potential Electricity Theft",
  "probability": 0.87,
  "risk_score": 87.0,
  "risk_level": "High",
  "top_features": ["mean_consumption", "rolling_mean_7d", "std_consumption"]
}
```

---

## Technology Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.10 |
| ML Models | scikit-learn, XGBoost |
| Imbalance | imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| API | FastAPI + Pydantic |
| UI | Streamlit |
| Container | Docker |
| Config | PyYAML |

---

## Class Imbalance Handling

The dataset has ~8% theft consumers. We used **SMOTE (Synthetic Minority
Oversampling Technique)** on the training set only to balance classes to 50/50.
SMOTE was never applied to validation or test sets to avoid data leakage.

---

## Explainability

SHAP (SHapley Additive exPlanations) is used to explain the Random Forest model:
- **Global:** Which features matter most across all consumers
- **Individual:** Why a specific consumer was flagged as theft

---

## Author

B III M.Sc. AI & ML Student
Advanced Machine Learning Lab Project
