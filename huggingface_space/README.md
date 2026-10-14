---
title: Electricity Theft Detection
emoji: ⚡
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ⚡ Explainable Electricity Theft Detection

Detects abnormal electricity consumption patterns using **Random Forest + SHAP explainability**.

## Features
- 12 engineered statistical features from SGCC time-series data
- Random Forest classifier (best F1: 0.3254, ROC-AUC: 0.7703)
- SHAP-based explainability
- Real-time risk scoring (0–100)

## Dataset
SGCC (State Grid Corporation of China) — 42,372 consumers × 1,035 daily readings

Built for **B III M.Sc. AI & ML — Advanced Machine Learning Lab Project**.
