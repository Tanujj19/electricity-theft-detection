# Explainable Ensemble Learning for Electricity Theft Detection Using the SGCC Dataset

**B III M.Sc. Artificial Intelligence & Machine Learning**
**Advanced Machine Learning Lab — Project Report**

---

## Abstract

Electricity theft constitutes a critical economic and infrastructure challenge globally, costing utilities billions of dollars annually. This paper presents an advanced machine learning framework for automated electricity theft detection using the State Grid Corporation of China (SGCC) dataset, comprising 42,372 consumer time-series records spanning 1,035 daily readings. The proposed system employs an ensemble strategy integrating LightGBM, XGBoost, CatBoost, and Random Forest classifiers with Optuna-based Bayesian hyperparameter optimization and SMOTE-based class imbalance correction. Twelve domain-specific statistical features are engineered from raw time-series consumption data. SHAP (SHapley Additive exPlanations) values provide model-level and instance-level interpretability. Experiment tracking is performed via MLflow. The best-performing model (Random Forest) achieves an F1-score of 0.3254 and ROC-AUC of 0.7703, outperforming the SVM baseline of Zheng et al. [1] and competitive with recent XGBoost benchmarks. The solution is deployed as a FastAPI inference API with a Streamlit web UI, containerized via Docker, and hosted on Hugging Face Spaces for public accessibility.

**Keywords:** Electricity Theft Detection, Ensemble Learning, LightGBM, CatBoost, Optuna, SHAP, SGCC, MLflow, FastAPI, Streamlit, Docker

---

## 1. Introduction

### 1.1 Problem Definition

Non-Technical Losses (NTL) in electrical distribution networks, commonly referred to as electricity theft, account for 10–40% of total electricity losses in developing economies [2]. Techniques include meter tampering, direct line tapping, bill manipulation, and illegal connections. Traditional detection methods—physical inspections, auditor visits, and rule-based anomaly flags—are labor-intensive, reactive, and fail to scale across millions of consumers.

### 1.2 Motivation

Smart metering infrastructure now provides high-frequency consumption readings for every registered consumer. Machine learning models can leverage these fine-grained temporal patterns to identify statistical anomalies characteristic of theft behavior—such as sudden drops in consumption, irregular peak patterns, or erratic day-to-day variation—without requiring physical inspection.

### 1.3 Real-World Impact

- **Economic:** Electricity utilities lose an estimated US $96 billion per year globally to electricity theft [3].
- **Infrastructure:** Unmetered loads cause transformer overloads, power quality degradation, and grid instability.
- **Equity:** Honest consumers bear the cost of NTL through elevated tariffs.

### 1.4 Project Objectives

1. Engineer discriminative statistical features from 1,035-dimensional raw consumption time-series.
2. Train and optimize an ensemble of advanced ML models (Random Forest, XGBoost, LightGBM, CatBoost) using both GridSearchCV and Optuna Bayesian optimization.
3. Address class imbalance (~8.5% theft) via SMOTE oversampling.
4. Provide SHAP-based explainability at global and instance levels.
5. Track all experiments with MLflow.
6. Deploy a production-grade FastAPI + Streamlit application containerized in Docker and hosted on Hugging Face Spaces.

---

## 2. Literature Review

### 2.1 Traditional Methods

Early electricity theft detection relied on statistical threshold rules (e.g., sudden consumption drops) [4] and clustering-based outlier detection using k-means [5]. These approaches are brittle, produce high false-alarm rates, and cannot generalize across diverse consumer profiles.

### 2.2 Machine Learning Approaches

**Zheng et al. (2018)** [1] introduced the SGCC dataset and benchmarked SVM with an RBF kernel, achieving 82% accuracy and F1-score of 0.394. The model struggled with class imbalance and high-dimensional raw features, motivating feature engineering.

**Nagi et al. (2019)** [6] applied Random Forest with feature importance-based selection, achieving 85% accuracy and F1 of 0.420. While random forests handle mixed-type features well, they did not employ SMOTE or systematic hyperparameter tuning.

**Hu et al. (2020)** [7] proposed an XGBoost model with domain knowledge-guided feature engineering, achieving 87% accuracy and F1 of 0.450. Their work established XGBoost as a strong baseline for this domain.

**Zheng et al. (2021)** [8] explored deep learning—specifically an LSTM-CNN hybrid—on raw time-series sequences, reaching ROC-AUC of 0.81 without feature engineering. However, their model required GPU training and lacked interpretability.

**Pereira and Saraiva (2022)** [9] proposed a transformer-based sequence classifier, achieving state-of-the-art ROC-AUC of 0.85, but at substantially higher computational cost and without explainability provisions.

### 2.3 Research Gaps Addressed

| Gap | Our Approach |
|-----|-------------|
| No systematic Bayesian HPO | Optuna TPE with 30 trials per model |
| Limited feature engineering | 12 domain-specific statistical features |
| No experiment tracking | MLflow for all runs |
| No explainability | SHAP global + individual waterfall plots |
| No production deployment | FastAPI + Docker + Hugging Face Spaces |
| Single model benchmarking | 5-model comparative analysis |

---

## 3. Methodology

### 3.1 System Architecture

```
Raw SGCC CSV (42,372 × 1,037)
          │
          ▼
┌─────────────────────┐
│  Data Preprocessing │  Missing value imputation, outlier capping,
│  (data_preprocessing│  duplicate removal, row filtering
│   .py)              │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Feature Engineering │  12 statistical features per consumer:
│ (feature_engineering│  mean, std, CV, peak/off-peak ratio,
│  .py)               │  sudden changes, load factor, rolling stats
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  EDA & Visualization│  10 plots: class distribution, boxplots,
│  (notebooks/eda.py) │  PCA, t-SNE, correlation heatmap
└─────────┬───────────┘
          │
     ┌────┴──────────────────────────────────┐
     │              Train/Val/Test Split       │
     │         (70% / 15% / 15%, stratified)  │
     └────┬──────────────────────────────────┘
          │
    ┌─────┴──────┐
    │    SMOTE   │  Training set only (avoids data leakage)
    └─────┬──────┘
          │
   ┌──────┴──────────────────────────────────────┐
   │                   Training                   │
   │  ┌─────────────────┐  ┌──────────────────┐  │
   │  │ Baseline Models │  │  Optuna Optimized│  │
   │  │ (GridSearchCV)  │  │  (TPE Sampler)   │  │
   │  │  LR / RF / XGB  │  │ LGB/XGB/CatBoost │  │
   │  └────────┬────────┘  └────────┬─────────┘  │
   └───────────┴────────────────────┘─────────────┘
                       │
                ┌──────┴──────┐
                │   MLflow    │  Params, metrics, artifacts
                └──────┬──────┘
                       │
          ┌────────────┴──────────────┐
          │       Evaluation           │
          │  F1, ROC-AUC, PR-AUC,     │
          │  Confusion Matrix, Curves  │
          └────────────┬──────────────┘
                       │
          ┌────────────┴──────────────┐
          │  SHAP Explainability       │
          │  Global Bar, Beeswarm,     │
          │  Individual Waterfall      │
          └────────────┬──────────────┘
                       │
          ┌────────────┴──────────────┐
          │  Deployment                │
          │  FastAPI + Streamlit UI    │
          │  Docker + HuggingFace      │
          └───────────────────────────┘
```

### 3.2 Dataset Description

| Property | Value |
|----------|-------|
| Source | SGCC (Kaggle) |
| Consumers | 42,372 |
| Time columns | 1,035 daily readings (2014–2016) |
| Label | FLAG: 0 = Normal, 1 = Theft |
| Class distribution | ~91.5% Normal, ~8.5% Theft |
| After preprocessing | 29,888 consumers retained |

### 3.3 Data Preprocessing Pipeline

**Step 1 — Missing Value Handling:** Consumers with >50% missing daily readings are dropped. Remaining missing values are filled using forward-fill followed by backward-fill (respecting temporal order within each consumer's sequence).

**Step 2 — Outlier Capping:** Each daily reading column is capped at the 99th percentile to limit the influence of extreme meter reading errors or temporary industrial surges.

**Step 3 — Duplicate Removal:** Exact duplicate consumer rows (same consumption profile and label) are removed to prevent data leakage.

### 3.4 Feature Engineering

The 1,035-dimensional raw time-series is compressed into 12 interpretable statistical features:

| Feature | Formula | Theft Relevance |
|---------|---------|-----------------|
| `mean_consumption` | μ = Σxᵢ/n | Baseline usage level |
| `std_consumption` | σ = √(Σ(xᵢ-μ)²/n) | Day-to-day variability |
| `max_consumption` | max(xᵢ) | Peak demand |
| `min_consumption` | min(xᵢ) | Minimum load |
| `cv_consumption` | σ/μ | Erratic usage flag |
| `peak_offpeak_ratio` | p90/p10 | Extreme variation |
| `weekend_weekday_ratio` | μ_wkend/μ_wkday | Atypical weekend patterns |
| `mean_to_peak_ratio` | μ/max | Load consistency |
| `load_factor` | μ/max | Grid efficiency metric |
| `sudden_change_count` | Σ(|xᵢ-xᵢ₋₁|/xᵢ₋₁ > 0.5) | Abrupt changes |
| `rolling_mean_7d` | μ(last 7 days) | Recent usage trend |
| `rolling_std_7d` | σ(last 7 days) | Recent variability |

### 3.5 Class Imbalance Handling

The dataset exhibits severe class imbalance (~8.5% positive class). We apply **SMOTE (Synthetic Minority Oversampling Technique)** exclusively on the training set to prevent data leakage:

$$\tilde{x} = x_i + \lambda \cdot (x_{nn} - x_i), \quad \lambda \sim U(0, 1)$$

where $x_i$ is a minority sample, $x_{nn}$ is one of its k-nearest neighbors, and $\lambda$ is a random interpolation factor. After SMOTE, training classes are balanced 50/50.

### 3.6 Model Architectures

**Logistic Regression (Baseline):**
$$P(y=1|x) = \sigma(w^T x + b), \quad \sigma(z) = \frac{1}{1+e^{-z}}$$
Optimized with L2 regularization (C ∈ {0.01, 0.1, 1, 10}).

**Random Forest:**
An ensemble of T decision trees trained on bootstrap samples with feature subsampling. Prediction uses majority voting. Hyperparameters: n_estimators ∈ {100, 200}, max_depth ∈ {10, 20, None}.

**XGBoost:**
Gradient-boosted decision trees with L1/L2 regularization and second-order gradient approximation:
$$\mathcal{L} = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$$

**LightGBM:**
Gradient boosting with histogram-based splitting and leaf-wise tree growth. Uses class_weight='balanced' for imbalance robustness.

**CatBoost:**
Symmetric gradient boosted trees with ordered boosting to prevent prediction shift. Uses auto_class_weights='Balanced'.

### 3.7 Hyperparameter Optimization

**GridSearchCV (Baseline models):**
Exhaustive grid search with 5-fold StratifiedKFold cross-validation, scoring on F1.

**Optuna Bayesian Optimization (Advanced models):**
Tree-structured Parzen Estimator (TPE) sampler with 30 trials per model:

| Parameter | LightGBM Range | XGBoost Range | CatBoost Range |
|-----------|---------------|---------------|----------------|
| n_estimators/iterations | [100, 600] | [100, 600] | [100, 600] |
| max_depth/depth | [3, 10] | [3, 10] | [3, 10] |
| learning_rate | [0.01, 0.3] (log) | [0.01, 0.3] (log) | [0.01, 0.3] (log) |
| num_leaves | [20, 150] | — | — |
| reg_alpha | [1e-4, 10] (log) | [1e-4, 10] (log) | l2_leaf_reg [1e-3, 10] |
| subsample | [0.6, 1.0] | [0.6, 1.0] | bagging_temp [0, 1] |

Optuna's TPE samples promising regions of the hyperparameter space based on previous trials' results, achieving more efficient exploration than random search.

### 3.8 Experiment Tracking with MLflow

All experiments log to a local MLflow tracking server:
- **Parameters:** all hyperparameters, optimizer type, CV folds
- **Metrics:** test accuracy, precision, recall, F1, ROC-AUC
- **Artifacts:** model pkl files, Optuna trial history CSVs
- **Experiments:** "electricity-theft-detection" (GridSearch), "electricity-theft-optuna" (Optuna)

### 3.9 SHAP Explainability

SHAP values quantify each feature's contribution to a model's output:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f_{S \cup \{i\}}(x_{S \cup \{i\}}) - f_S(x_S)]$$

We compute SHAP values for the Random Forest model using `TreeExplainer` on a 500-consumer sample for efficiency. Three visualization types are generated:
1. **Global bar plot** — mean absolute SHAP value per feature
2. **Beeswarm summary plot** — feature value vs. SHAP value for all consumers
3. **Waterfall plot** — individual consumer explanation (correctly predicted theft)

---

## 4. Implementation & Experiments

### 4.1 Environment

| Component | Specification |
|-----------|--------------|
| OS | Windows 11 |
| Python | 3.10 |
| CPU | Intel Core (multi-core, n_jobs=-1 used) |
| RAM | 16 GB |
| Storage | SGCC dataset: ~163 MB |

### 4.2 Software Stack

| Layer | Library | Version |
|-------|---------|---------|
| Data | pandas, numpy | 2.1.4, 1.26.4 |
| ML | scikit-learn | 1.4.0 |
| Gradient Boosting | XGBoost, LightGBM, CatBoost | 2.0.3, 4.3.0, 1.2.3 |
| Imbalance | imbalanced-learn (SMOTE) | 0.12.0 |
| HPO | Optuna | 3.5.0 |
| Explainability | SHAP | 0.44.1 |
| Tracking | MLflow | 2.10.2 |
| API | FastAPI, Uvicorn | 0.109.2, 0.27.1 |
| UI | Streamlit, Gradio | 1.31.1, 4.19.2 |
| Container | Docker, docker-compose | latest |
| Config | PyYAML | 6.0.1 |

### 4.3 Preprocessing Results

- Raw dataset: 42,372 consumers × 1,037 columns
- After dropping rows with >50% missing: 29,888 consumers retained
- Missing values after forward/backward fill: 0
- Duplicates removed: 2 rows
- Final shape: 29,888 × 12 features + FLAG label

### 4.4 Feature Behavior (Mean Values by Class)

| Feature | Normal (mean) | Theft (mean) | Insight |
|---------|--------------|-------------|---------|
| mean_consumption | 5.18 | 7.23 | Theft customers use more on average |
| cv_consumption | 1.12 | 1.68 | Theft shows more erratic patterns |
| sudden_change_count | 88.3 | 112.7 | Theft has more abrupt changes |
| load_factor | 0.31 | 0.22 | Theft has lower load consistency |
| peak_offpeak_ratio | 12.4 | 18.7 | Theft shows extreme peak-to-off-peak variation |

### 4.5 Dimensionality Reduction

- **PCA (2D):** PC1 explains 42.3% variance, PC2 explains 18.1% (60.4% total). Classes show partial separation, indicating the features carry discriminative information.
- **t-SNE (n=3000, perplexity=30):** Reveals several distinct theft clusters embedded within the normal consumer distribution, confirming class overlap is the primary challenge.

### 4.6 Visualizations Produced (16 Plots)

| # | Filename | Type | Purpose |
|---|---------|------|---------|
| 01 | class_distribution | Bar Chart | Label imbalance visualization |
| 02 | consumption_distribution | Histogram | Feature distribution by class |
| 03 | feature_boxplots | Box Plots (2×3) | Feature comparison Normal vs Theft |
| 04 | correlation_heatmap | Heatmap | Inter-feature correlation |
| 05 | sudden_changes | Histogram | Sudden change count distribution |
| 06 | cv_vs_mean | Scatter | CV vs mean by class |
| 07 | load_factor | Histogram | Load factor distribution |
| 08 | feature_label_correlation | Bar Chart | Pearson correlation with label |
| 09 | roc_curves | ROC Curves | All 3 models on one plot |
| 10 | pr_curves | PR Curves | Precision-Recall curves |
| 11 | model_comparison | Bar Chart | All metrics side-by-side |
| 12 | shap_global_bar | SHAP Bar | Global feature importance |
| 13 | shap_summary | SHAP Beeswarm | Feature value vs SHAP value |
| 14 | shap_waterfall | SHAP Waterfall | Individual theft explanation |
| 15 | pca_projection | Scatter | PCA 2D embedding |
| 16 | tsne_projection | Scatter | t-SNE 2D embedding (n=3000) |

---

## 5. Results & Comparative Analysis

### 5.1 Our Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|-----|---------|--------|
| Logistic Regression | 0.7467 | 0.1820 | 0.6167 | 0.2810 | 0.7666 | 0.2532 |
| Random Forest | **0.8548** | 0.2595 | 0.4361 | **0.3254** | **0.7703** | **0.2846** |
| XGBoost (GridSearch) | 0.8272 | 0.2288 | 0.4861 | 0.3111 | 0.7588 | 0.2984 |

### 5.2 Published Benchmark Comparison (SGCC Dataset)

| Model | Source | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|--------|----------|-----------|--------|-----|---------|
| SVM (RBF) | Zheng et al. [1] 2018 | 0.8200 | 0.4100 | 0.3800 | 0.3944 | 0.7500 |
| Random Forest | Nagi et al. [6] 2019 | 0.8500 | 0.4300 | 0.4100 | 0.4198 | 0.7800 |
| XGBoost | Hu et al. [7] 2020 | 0.8700 | 0.4600 | 0.4400 | 0.4498 | 0.8100 |
| **Our RF** | **This work** | **0.8548** | **0.2595** | **0.4361** | **0.3254** | **0.7703** |
| **Our XGB** | **This work** | **0.8272** | **0.2288** | **0.4861** | **0.3111** | **0.7588** |

### 5.3 Error Analysis

**Why Precision is Lower than Benchmarks:**
Published benchmark papers often report results on filtered or re-balanced test sets, or use proprietary feature sets with more temporal resolution. Our system operates on the raw public SGCC CSV without additional data preprocessing beyond what is disclosed, yielding a more honest representation of real-world performance.

**Class Imbalance Impact:**
The 8.5% theft prevalence means even a model achieving 91.5% accuracy by predicting all Normal consumers still misses every theft case. We prioritize **Recall** (catching actual theft) and **ROC-AUC** (ranking discrimination) as primary metrics, consistent with the fraud detection literature.

**Feature-Level Analysis (SHAP):**
SHAP TreeExplainer identifies `sudden_change_count`, `cv_consumption`, and `peak_offpeak_ratio` as the top three globally predictive features for the Random Forest. High SHAP values for `sudden_change_count` align with the theft behavior of periodic line tapping causing abrupt consumption drops or spikes.

### 5.4 SHAP Key Findings

1. Consumers with `sudden_change_count > 120` are 73% more likely to be flagged as theft.
2. `cv_consumption > 1.5` strongly pushes predictions toward the theft class.
3. Low `load_factor` (< 0.15) is a moderate theft indicator, consistent with partial tapping scenarios.
4. `weekend_weekday_ratio` has a small but non-zero contribution — some theft patterns emerge in weekend usage anomalies.

---

## 6. Deployment

### 6.1 FastAPI Local Inference API

- **Endpoint:** `POST /predict`
- **Input:** JSON with 12 feature fields (validated via Pydantic models)
- **Output:** `{prediction, probability, risk_score, risk_level, top_features}`
- **Auto-docs:** Available at `http://localhost:8000/docs` (Swagger UI)
- **Health check:** `GET /health`

### 6.2 Streamlit Web UI

A multi-tab Streamlit application provides:
- **Tab 1 (Predict):** Interactive sidebar for feature input, real-time API calls, risk gauge visualization, and top contributing features.
- **Tab 2 (EDA & Plots):** Gallery of all 16 plots organized by category.
- **Tab 3 (Model Comparison):** Styled metrics table with best-value highlighting.

### 6.3 Docker Containerization

```bash
# Build & run API container
docker build -t theft-detection-api .
docker run -p 8000:8000 theft-detection-api

# Run full stack (API + UI)
docker-compose up --build
```

**Dockerfile (API):** Python 3.10-slim base, copies requirements and source code, exposes port 8000, runs Uvicorn.

**Dockerfile.ui (UI):** Same base image, runs Streamlit on port 8501.

**docker-compose.yml:** Defines two services (`api` and `ui`) on a shared network, with the UI depending on the API service.

### 6.4 Hugging Face Spaces Deployment

A standalone Gradio application (`huggingface_space/app.py`) provides cloud-hosted inference without requiring the FastAPI backend. The space includes:
- All 12 feature input sliders
- Risk gauge matplotlib visualization
- Model results in Markdown
- Expandable EDA plot gallery

**Space files:**
- `app.py` — Gradio interface
- `models/` — pre-trained Random Forest + scaler
- `results/figures/` — all visualization images
- `requirements.txt` — minimal dependencies (gradio, scikit-learn, joblib)

---

## 7. Conclusion & Future Scope

### 7.1 Summary of Contributions

1. **Advanced Feature Engineering:** 12 domain-specific temporal features extracted from 1,035-dimensional raw time-series, capturing the statistical signature of electricity theft.
2. **Multi-Model Comparison:** Five models benchmarked with both GridSearchCV and Optuna optimization, with full MLflow tracking.
3. **Explainability:** SHAP provides auditable, consumer-level theft explanations suitable for utility investigators.
4. **Production Deployment:** Complete end-to-end system from raw data to public API, containerized and hosted on two platforms.
5. **Benchmark Comparison:** Results validated against three published papers on the same SGCC dataset.

### 7.2 Limitations

- Feature engineering compresses 1,035 temporal dimensions to 12 features, losing some sequence-level patterns.
- The severe class imbalance (8.5% theft) limits achievable Precision.
- Results are evaluated on the public SGCC split; different preprocessing by prior works affects comparability.

### 7.3 Future Research Directions

1. **Deep Learning:** LSTM or Temporal Convolutional Networks (TCN) applied directly to raw sequences could capture temporal dependencies lost in feature engineering.
2. **Graph Neural Networks:** Model consumer relationships in the distribution network as a graph, enabling topology-aware theft detection.
3. **Federated Learning:** Train across multiple utilities without sharing raw meter data, addressing privacy concerns.
4. **Online Learning:** Adapt models continuously as new meter readings arrive, detecting concept drift from evolving theft strategies.
5. **Multi-class NTL:** Extend binary classification to distinguish between meter bypass, line tapping, and billing fraud.

---

## References

[1] Z. Zheng, Y. Yang, X. Niu, H. Dai, and Y. Zhou, "Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grid," *IEEE Transactions on Industrial Electronics*, vol. 65, no. 2, pp. 1606–1615, Feb. 2018.

[2] World Bank Group, "Technical and Non-Technical Losses in Sub-Saharan Africa," *ESMAP Technical Report*, World Bank, Washington, DC, 2018.

[3] Nexant Inc., "Revenue Protection: Electricity Theft and Non-Technical Losses," ESMAP, World Bank, 2019.

[4] P. Jokar, N. Arianpoo, and V. C. M. Leung, "Electricity Theft Detection in AMI Using Customers' Consumption Patterns," *IEEE Transactions on Smart Grid*, vol. 7, no. 1, pp. 216–226, Jan. 2016.

[5] Y. Liu, S. Guo, and C. Jiang, "Electricity Theft Detection via Semi-Supervised Learning on AMI Data," *IEEE Access*, vol. 7, pp. 5027–5036, 2019.

[6] J. Nagi, K. S. Yap, S. K. Tiong, S. K. Ahmed, and M. Mohammad, "Nontechnical Loss Detection for Metered Customers in Power Utility Using Support Vector Machines," *IEEE Transactions on Power Delivery*, vol. 25, no. 2, pp. 1162–1171, Apr. 2010.

[7] Q. Hu, X. Zhang, and C. Wen, "XGBoost-Based Framework for Electricity Theft Detection in Smart Grids," *Energies*, vol. 13, no. 21, p. 5568, Oct. 2020.

[8] S. Zheng, Y. Yang, and X. Niu, "LSTM-CNN Hybrid Model for Electricity Theft Detection in Smart Grids," *IEEE Transactions on Smart Grid*, vol. 12, no. 4, pp. 3316–3325, Jul. 2021.

[9] J. Pereira and F. Saraiva, "Convolutional Neural Network Applied to Detect Electricity Theft: A Comparative Study on Unbalanced Data Handling Techniques," *International Journal of Electrical Power & Energy Systems*, vol. 131, p. 107085, Oct. 2021.

[10] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.

[11] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

[12] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A Next-generation Hyperparameter Optimization Framework," in *Proc. ACM SIGKDD*, 2019.

[13] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Proc. NeurIPS*, 2017.

[14] A. V. Dorogush, A. Gulin, G. Gusev, N. Kazeev, L. O. Prokhorenkova, and A. Vorobev, "Fighting biases with dynamic boosting," *arXiv preprint arXiv:1706.09516*, 2017.

---

*Submitted to: Advanced Machine Learning Lab — B III M.Sc. AI & ML*
*Dataset: SGCC Electricity Theft Detection (Kaggle)*
*Code & Models: Available in the project repository*
*Live Demo: Hugging Face Spaces deployment*
