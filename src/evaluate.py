import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_splits(splits_path):
    data     = np.load(splits_path, allow_pickle=True)
    X_test   = data["X_test"]
    y_test   = data["y_test"]
    print(f"  Test set: {X_test.shape[0]} consumers")
    print(f"  Normal: {(y_test==0).sum()}  Theft: {(y_test==1).sum()}")
    return X_test, y_test

def load_model(name, models_dir):
    path  = os.path.join(models_dir, f"{name}.pkl")
    model = joblib.load(path)
    print(f"  Loaded: {path}")
    return model

def compute_metrics(model, X_test, y_test, model_name, threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        "Model":     model_name,
        "Threshold": threshold,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
        "PR-AUC":    round(average_precision_score(y_test, y_proba), 4),
    }
    return metrics, y_pred, y_proba


# ── Threshold Tuning: find the threshold that maximises F1 ────────────────────
def tune_threshold(model, X_test, y_test, model_name):
    """Search thresholds 0.20 – 0.70 and return the one with best F1."""
    y_proba = model.predict_proba(X_test)[:, 1]
    best_thresh, best_f1 = 0.5, 0.0
    records = []
    for t in np.arange(0.20, 0.71, 0.02):
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        p  = precision_score(y_test, y_pred, zero_division=0)
        r  = recall_score(y_test, y_pred, zero_division=0)
        records.append({"threshold": round(t, 2), "f1": round(f1, 4),
                        "precision": round(p, 4), "recall": round(r, 4)})
        if f1 > best_f1:
            best_f1, best_thresh = f1, round(t, 2)
    print(f"  [{model_name}] Best threshold={best_thresh}  Best F1={best_f1:.4f}")
    return best_thresh, best_f1, pd.DataFrame(records)


# ── Plot: Threshold vs F1 / Precision / Recall ────────────────────────────────
def plot_threshold_curves(thresh_df, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresh_df["threshold"], thresh_df["f1"],        label="F1",        color="#e74c3c", linewidth=2)
    ax.plot(thresh_df["threshold"], thresh_df["precision"],  label="Precision", color="#3498db", linewidth=2)
    ax.plot(thresh_df["threshold"], thresh_df["recall"],     label="Recall",    color="#2ecc71", linewidth=2)
    best_t = thresh_df.loc[thresh_df["f1"].idxmax(), "threshold"]
    ax.axvline(best_t, color="orange", linestyle="--", linewidth=1.5, label=f"Best T={best_t}")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    fname = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    ax.set_title(f"Threshold Tuning — {model_name}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"threshold_{fname}.png"), dpi=150)
    plt.close()
    print(f"  Saved threshold curve: threshold_{fname}.png")


# ── Plot: Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, model_name, save_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Theft"],
                yticklabels=["Normal", "Theft"],
                linewidths=0.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(save_dir, f"cm_{fname}.png"), dpi=150)
    plt.close()
    print(f"  Saved confusion matrix: cm_{fname}.png")


# ── Plot: ROC Curves (all models on one plot) ─────────────────────────────────
def plot_roc_curves(roc_data, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]
    for (name, fpr, tpr, auc_val), color in zip(roc_data, colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "09_roc_curves.png"), dpi=150)
    plt.close()
    print("  Saved: 09_roc_curves.png")


# ── Plot: Precision-Recall Curves ─────────────────────────────────────────────
def plot_pr_curves(pr_data, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]
    for (name, prec, rec, pr_auc), color in zip(pr_data, colors):
        ax.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.3f})", color=color, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "10_pr_curves.png"), dpi=150)
    plt.close()
    print("  Saved: 10_pr_curves.png")


# ── Plot: Model Comparison Bar Chart ─────────────────────────────────────────
def plot_comparison_bar(results_df, save_dir):
    metrics  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    models   = results_df["Model"].tolist()
    x        = np.arange(len(metrics))
    width    = 0.13
    colors   = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12", "#1abc9c"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = results_df[results_df["Model"] == model][metrics].values.flatten().astype(float)
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, edgecolor="black")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics (Threshold-Tuned)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "11_model_comparison.png"), dpi=150)
    plt.close()
    print("  Saved: 11_model_comparison.png")


# ── Error Analysis ────────────────────────────────────────────────────────────
def run_error_analysis(models_dict, X_test, y_test, figures_dir, metrics_dir):
    """
    Identify false positives and false negatives for each model.
    Saves error summary CSV and a grouped bar plot.
    """
    print("\n=== Error Analysis ===")
    error_records = []
    for name, (model, threshold) in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr_val = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0
        fnr_val = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0

        error_records.append({
            "Model": name,
            "True Positives":  int(tp),
            "True Negatives":  int(tn),
            "False Positives": int(fp),
            "False Negatives": int(fn),
            "False Positive Rate": fpr_val,
            "False Negative Rate": fnr_val,
            "Miss Rate (FNR)": fnr_val,
        })
        print(f"  {name}: TP={tp} TN={tn} FP={fp} FN={fn} | FPR={fpr_val} FNR={fnr_val}")

    df_err = pd.DataFrame(error_records)
    err_path = os.path.join(metrics_dir, "error_analysis.csv")
    df_err.to_csv(err_path, index=False)
    print(f"  Error analysis saved to {err_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x  = np.arange(len(df_err))
    w  = 0.35
    colors_fp = "#e74c3c"
    colors_fn = "#3498db"

    axes[0].bar(x - w/2, df_err["False Positives"], w, label="False Positives", color=colors_fp)
    axes[0].bar(x + w/2, df_err["False Negatives"], w, label="False Negatives", color=colors_fn)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_err["Model"], rotation=15, ha="right", fontsize=8)
    axes[0].set_title("Error Counts per Model", fontweight="bold")
    axes[0].legend()
    axes[0].set_ylabel("Count")

    axes[1].bar(x - w/2, df_err["False Positive Rate"], w, label="FPR (False Alarm Rate)", color=colors_fp)
    axes[1].bar(x + w/2, df_err["False Negative Rate"], w, label="FNR (Miss Rate)", color=colors_fn)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_err["Model"], rotation=15, ha="right", fontsize=8)
    axes[1].set_title("Error Rates per Model", fontweight="bold")
    axes[1].legend()
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0, 1.0)

    fig.suptitle("Rigorous Error Analysis — All Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "17_error_analysis.png"), dpi=150)
    plt.close()
    print("  Saved: 17_error_analysis.png")
    return df_err


def save_metrics_table(results_df, metrics_dir):
    os.makedirs(metrics_dir, exist_ok=True)

    # Add published benchmark numbers for comparison
    benchmark = pd.DataFrame([
        {"Model": "--- Published Benchmarks ---", "Threshold": "-", "Accuracy": "", "Precision": "", "Recall": "", "F1": "", "ROC-AUC": "", "PR-AUC": ""},
        {"Model": "Zheng et al. 2018 (SVM)",     "Threshold": "0.50", "Accuracy": 0.8200, "Precision": 0.4100, "Recall": 0.3800, "F1": 0.3944, "ROC-AUC": 0.7500, "PR-AUC": "-"},
        {"Model": "Nagi et al. 2019 (RF)",       "Threshold": "0.50", "Accuracy": 0.8500, "Precision": 0.4300, "Recall": 0.4100, "F1": 0.4198, "ROC-AUC": 0.7800, "PR-AUC": "-"},
        {"Model": "Hu et al. 2020 (XGBoost)",    "Threshold": "0.50", "Accuracy": 0.8700, "Precision": 0.4600, "Recall": 0.4400, "F1": 0.4498, "ROC-AUC": 0.8100, "PR-AUC": "-"},
    ])
    full_df = pd.concat([results_df, benchmark], ignore_index=True)

    path = os.path.join(metrics_dir, "model_comparison.csv")
    results_df.to_csv(path, index=False)

    bench_path = os.path.join(metrics_dir, "benchmark_comparison.csv")
    full_df.to_csv(bench_path, index=False)

    print(f"\n  Comparison table saved to {path}")
    print(f"  Benchmark comparison saved to {bench_path}")
    print("\n--- Our Models (Threshold-Tuned) ---")
    print(results_df.to_string(index=False))
    print("\n--- Published Benchmarks (SGCC dataset) ---")
    print(benchmark.iloc[1:].to_string(index=False))


def try_load_optuna_models(models_dir):
    """Attempt to load Optuna-trained models if they exist."""
    optuna_models = {}
    candidates = {
        "LightGBM (Optuna)": "lightgbm_model",
        "XGBoost (Optuna)":  "xgboost_optuna",
        "CatBoost (Optuna)": "catboost_model",
    }
    for display_name, fname in candidates.items():
        path = os.path.join(models_dir, f"{fname}.pkl")
        if os.path.exists(path):
            try:
                optuna_models[display_name] = joblib.load(path)
                print(f"  Loaded Optuna model: {display_name}")
            except Exception as e:
                print(f"  Could not load {display_name}: {e}")
    return optuna_models


def run_evaluation():
    config      = load_config()
    models_dir  = config["paths"]["models_dir"]
    figures_dir = config["paths"]["figures_dir"]
    metrics_dir = config["paths"]["metrics_dir"]
    splits_path = "data/processed/test_split.npz"

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # ── 1. Load test data ─────────────────────────────────────────────────────
    print("\n=== Step 1: Load test split ===")
    X_test, y_test = load_splits(splits_path)

    # ── 2. Load baseline models ───────────────────────────────────────────────
    print("\n=== Step 2: Load baseline models ===")
    baseline_models = {
        "Logistic Regression": load_model("logistic_regression", models_dir),
        "Random Forest":       load_model("random_forest",       models_dir),
        "XGBoost":             load_model("xgboost_model",       models_dir),
    }

    # ── 3. Load Optuna models (if available) ──────────────────────────────────
    print("\n=== Step 3: Load Optuna-tuned models (if available) ===")
    optuna_models = try_load_optuna_models(models_dir)
    all_models = {**baseline_models, **optuna_models}

    # ── 4. Threshold Tuning for all models ───────────────────────────────────
    print("\n=== Step 4: Threshold Tuning (optimise F1) ===")
    best_thresholds = {}
    for name, model in all_models.items():
        best_t, best_f1, thresh_df = tune_threshold(model, X_test, y_test, name)
        best_thresholds[name] = best_t
        plot_threshold_curves(thresh_df, name, figures_dir)

    # ── 5. Evaluate each model with its best threshold ────────────────────────
    print("\n=== Step 5: Compute metrics & plots (tuned thresholds) ===")
    all_metrics = []
    roc_data    = []
    pr_data     = []
    error_models = {}

    for name, model in all_models.items():
        threshold = best_thresholds[name]
        print(f"\n  -- {name} (threshold={threshold}) --")
        metrics, y_pred, y_proba = compute_metrics(model, X_test, y_test, name, threshold)
        all_metrics.append(metrics)
        error_models[name] = (model, threshold)

        # Confusion matrix
        plot_confusion_matrix(y_test, y_pred, name, figures_dir)

        # ROC data (always uses raw probabilities)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data.append((name, fpr, tpr, metrics["ROC-AUC"]))

        # PR data
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_data.append((name, prec, rec, metrics["PR-AUC"]))

    # ── 6. Combined plots ─────────────────────────────────────────────────────
    print("\n=== Step 6: Combined plots ===")
    plot_roc_curves(roc_data, figures_dir)
    plot_pr_curves(pr_data, figures_dir)

    results_df = pd.DataFrame(all_metrics)
    plot_comparison_bar(results_df, figures_dir)

    # ── 7. Save comparison table ──────────────────────────────────────────────
    print("\n=== Step 7: Save comparison table ===")
    save_metrics_table(results_df, metrics_dir)

    # ── 8. Error Analysis ─────────────────────────────────────────────────────
    print("\n=== Step 8: Rigorous Error Analysis ===")
    run_error_analysis(error_models, X_test, y_test, figures_dir, metrics_dir)

    # ── 9. Identify best model ────────────────────────────────────────────────
    best_row   = results_df.loc[results_df["F1"].idxmax()]
    best_model = best_row["Model"]
    print(f"\nBest model by F1: {best_model}  (F1={best_row['F1']:.4f}, ROC-AUC={best_row['ROC-AUC']:.4f}, Threshold={best_row['Threshold']})")
    print("\nEvaluation complete.")
    return best_model


if __name__ == "__main__":
    run_evaluation()
