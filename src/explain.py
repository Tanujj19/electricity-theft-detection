import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import yaml
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_model(models_dir):
    path  = os.path.join(models_dir, "random_forest.pkl")
    model = joblib.load(path)
    print(f"  Loaded model: {path}")
    return model

def load_test_data(splits_path, features_path, label_col):
    data    = np.load(splits_path, allow_pickle=True)
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    # get feature names from features.csv
    df_feat      = pd.read_csv(features_path)
    feature_cols = [c for c in df_feat.columns if c != label_col]

    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    print(f"  Test samples: {X_test_df.shape[0]}  Features: {X_test_df.shape[1]}")
    return X_test_df, y_test

def compute_shap_values(model, X_test_df):
    print("  Computing SHAP values ...")
    # Sample 500 rows for speed
    X_sample  = X_test_df.sample(n=500, random_state=42).reset_index(drop=True)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    # For binary classification RF, shap_values returns list [class0, class1]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]   # use class 1 (theft)

    print(f"  SHAP values shape: {shap_vals.shape}")
    return explainer, shap_vals, X_sample

# ── Plot 1: SHAP Global Bar (mean absolute SHAP) ─────────────────────────────
def plot_shap_bar(shap_vals, X_test_df, save_dir):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    feat_imp = pd.Series(mean_abs, index=X_test_df.columns).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors  = ["#e74c3c" if v == feat_imp.max() else "#3498db" for v in feat_imp.values]
    ax.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor="black")
    ax.set_title("SHAP Global Feature Importance (Mean |SHAP|)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|")
    for i, v in enumerate(feat_imp.values):
        ax.text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "12_shap_global_bar.png"), dpi=150)
    plt.close()
    print("  Saved: 12_shap_global_bar.png")

# ── Plot 2: SHAP Beeswarm Summary Plot ───────────────────────────────────────
def plot_shap_summary(shap_vals, X_test_df, save_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_vals, X_test_df, show=False, plot_size=None)
    plt.title("SHAP Summary Plot — Random Forest (Theft Class)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "13_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 13_shap_summary.png")

# ── Plot 3: Individual SHAP Waterfall for one theft consumer ─────────────────
def plot_shap_waterfall(explainer, shap_vals, X_sample, y_test, save_dir):
    model_path = "models/random_forest.pkl"
    model      = joblib.load(model_path)
    y_pred     = model.predict(X_sample.values)
    X_test_df  = X_sample

    theft_idx = np.where((y_test[:500] == 1) & (y_pred == 1))[0]
    if len(theft_idx) == 0:
        print("  No correctly predicted theft found, using first theft consumer.")
        theft_idx = np.where(y_test == 1)[0]

    idx = theft_idx[0]
    print(f"  Explaining consumer index {idx} (actual=Theft, predicted=Theft)")

    # Waterfall plot using shap.Explanation object
    base_val   = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    explanation = shap.Explanation(
        values    = shap_vals[idx],
        base_values = base_val,
        data      = X_test_df.iloc[idx].values,
        feature_names = X_test_df.columns.tolist()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.title(f"SHAP Waterfall — Consumer #{idx} (Predicted: Theft)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "14_shap_waterfall_theft.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 14_shap_waterfall_theft.png")
    return idx

# ── Print: Individual consumer explanation in plain text ─────────────────────
def print_individual_explanation(shap_vals, X_test_df, y_test, idx):
    feat_names  = X_test_df.columns.tolist()
    shap_row    = shap_vals[idx]
    feat_vals   = X_test_df.iloc[idx].values

    explanation = pd.DataFrame({
        "Feature":    feat_names,
        "Value":      feat_vals,
        "SHAP":       shap_row
    }).sort_values("SHAP", ascending=False)

    print(f"\n  Consumer #{idx} — Individual SHAP Explanation:")
    print(f"  {'Feature':<28} {'Value':>10}  {'SHAP':>10}")
    print("  " + "-" * 52)
    for _, row in explanation.iterrows():
        direction = "pushes THEFT" if row["SHAP"] > 0 else "pushes NORMAL"
        print(f"  {row['Feature']:<28} {row['Value']:>10.3f}  {row['SHAP']:>10.4f}  ({direction})")

def run_explanation():
    config      = load_config()
    models_dir  = config["paths"]["models_dir"]
    figures_dir = config["paths"]["figures_dir"]
    feat_path   = config["paths"]["processed_data"]
    label_col   = config["data"]["label_column"]
    splits_path = "data/processed/test_split.npz"

    os.makedirs(figures_dir, exist_ok=True)

    print("\n=== Step 1: Load model and test data ===")
    model      = load_model(models_dir)
    X_test_df, y_test = load_test_data(splits_path, feat_path, label_col)

    print("\n=== Step 2: Compute SHAP values ===")
    explainer, shap_vals, X_sample = compute_shap_values(model, X_test_df)

    print("\n=== Step 3: Global SHAP bar plot ===")
    plot_shap_bar(shap_vals, X_sample, figures_dir)

    print("\n=== Step 4: SHAP summary (beeswarm) plot ===")
    plot_shap_summary(shap_vals, X_sample, figures_dir)

    print("\n=== Step 5: Individual waterfall explanation ===")
    idx = plot_shap_waterfall(explainer, shap_vals, X_sample, y_test, figures_dir)

    print("\n=== Step 6: Plain-text individual explanation ===")
    print_individual_explanation(shap_vals, X_sample, y_test, idx)

    print("\nExplainability complete. Plots saved to:", figures_dir)

if __name__ == "__main__":
    run_explanation()
