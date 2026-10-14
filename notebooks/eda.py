import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import yaml
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_features(path):
    df = pd.read_csv(path)
    print(f"Loaded features: {df.shape}")
    return df

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ── Plot 1: Class Distribution ──────────────────────────────────────────────
def plot_class_distribution(df, label_col, save_dir):
    counts = df[label_col].value_counts()
    labels = ["Normal (0)", "Theft (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f"{val}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=11)
    ax.set_title("Class Distribution: Normal vs Theft", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Consumers")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "01_class_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: 01_class_distribution.png")

# ── Plot 2: Mean Consumption Distribution by Class ───────────────────────────
def plot_consumption_distribution(df, label_col, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for flag, label, color in [(0, "Normal", "#2ecc71"), (1, "Theft", "#e74c3c")]:
        subset = df[df[label_col] == flag]["mean_consumption"]
        ax.hist(subset, bins=60, alpha=0.6, label=label, color=color, edgecolor="none")
    ax.set_title("Mean Daily Consumption Distribution: Normal vs Theft", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Daily Consumption (kWh)")
    ax.set_ylabel("Number of Consumers")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "02_consumption_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: 02_consumption_distribution.png")

# ── Plot 3: Feature Boxplots Normal vs Theft ─────────────────────────────────
def plot_feature_boxplots(df, label_col, save_dir):
    features = ["mean_consumption", "std_consumption", "cv_consumption",
                "sudden_change_count", "load_factor", "peak_offpeak_ratio"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        data = [df[df[label_col] == 0][feat].values,
                df[df[label_col] == 1][feat].values]
        bp = axes[i].boxplot(data, patch_artist=True, labels=["Normal", "Theft"],
                             medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("#2ecc71")
        bp["boxes"][1].set_facecolor("#e74c3c")
        axes[i].set_title(feat.replace("_", " ").title(), fontsize=10, fontweight="bold")
        axes[i].set_ylabel("Value")
    fig.suptitle("Feature Comparison: Normal vs Theft Consumers", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_feature_boxplots.png"), dpi=150)
    plt.close()
    print("  Saved: 03_feature_boxplots.png")

# ── Plot 4: Correlation Heatmap ───────────────────────────────────────────────
def plot_correlation_heatmap(df, label_col, save_dir):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "04_correlation_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: 04_correlation_heatmap.png")

# ── Plot 5: Sudden Change Count Distribution ─────────────────────────────────
def plot_sudden_changes(df, label_col, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for flag, label, color in [(0, "Normal", "#2ecc71"), (1, "Theft", "#e74c3c")]:
        subset = df[df[label_col] == flag]["sudden_change_count"]
        ax.hist(subset, bins=50, alpha=0.6, label=label, color=color, edgecolor="none")
    ax.set_title("Sudden Consumption Change Count: Normal vs Theft", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Sudden Changes (>50% day-over-day)")
    ax.set_ylabel("Number of Consumers")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "05_sudden_changes.png"), dpi=150)
    plt.close()
    print("  Saved: 05_sudden_changes.png")

# ── Plot 6: CV Consumption vs Mean Consumption Scatter ───────────────────────
def plot_cv_vs_mean(df, label_col, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for flag, label, color in [(0, "Normal", "#2ecc71"), (1, "Theft", "#e74c3c")]:
        subset = df[df[label_col] == flag]
        ax.scatter(subset["mean_consumption"], subset["cv_consumption"],
                   alpha=0.3, s=8, label=label, color=color)
    ax.set_title("CV of Consumption vs Mean Consumption", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Daily Consumption (kWh)")
    ax.set_ylabel("Coefficient of Variation (CV)")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "06_cv_vs_mean.png"), dpi=150)
    plt.close()
    print("  Saved: 06_cv_vs_mean.png")

# ── Plot 7: Load Factor Distribution ─────────────────────────────────────────
def plot_load_factor(df, label_col, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for flag, label, color in [(0, "Normal", "#2ecc71"), (1, "Theft", "#e74c3c")]:
        subset = df[df[label_col] == flag]["load_factor"]
        ax.hist(subset, bins=50, alpha=0.6, label=label, color=color, edgecolor="none")
    ax.set_title("Load Factor Distribution: Normal vs Theft", fontsize=13, fontweight="bold")
    ax.set_xlabel("Load Factor (Mean / Max Consumption)")
    ax.set_ylabel("Number of Consumers")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "07_load_factor.png"), dpi=150)
    plt.close()
    print("  Saved: 07_load_factor.png")

# ── Plot 8: Feature Importance (Correlation with FLAG) ───────────────────────
def plot_feature_correlation_with_label(df, label_col, save_dir):
    feature_cols = [c for c in df.columns if c != label_col]
    corr_with_label = df[feature_cols].corrwith(df[label_col]).sort_values()
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in corr_with_label.values]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(corr_with_label.index, corr_with_label.values, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Feature Correlation with Theft Label (FLAG)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient")
    for bar, val in zip(bars, corr_with_label.values):
        ax.text(val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "08_feature_label_correlation.png"), dpi=150)
    plt.close()
    print("  Saved: 08_feature_label_correlation.png")

# ── Plot 9: PCA 2D Projection ─────────────────────────────────────────────────
def plot_pca(df, label_col, save_dir):
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values
    y = df[label_col].values
    X_sc  = StandardScaler().fit_transform(X)
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    fig, ax = plt.subplots(figsize=(8, 6))
    for flag, label, color in [(0, "Normal", "#2ecc71"), (1, "Theft", "#e74c3c")]:
        mask = y == flag
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   alpha=0.3, s=8, label=label, color=color)
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title("PCA 2D Projection — Normal vs Theft", fontsize=13, fontweight="bold")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "15_pca_projection.png"), dpi=150)
    plt.close()
    print("  Saved: 15_pca_projection.png")
    print(f"  PCA explained variance: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  Total={sum(var)*100:.1f}%")

# ── Plot 10: t-SNE 2D Projection ──────────────────────────────────────────────
def plot_tsne(df, label_col, save_dir):
    feature_cols = [c for c in df.columns if c != label_col]
    # Sample 3000 for speed
    df_sample = df.sample(n=3000, random_state=42).reset_index(drop=True)
    X = df_sample[feature_cols].values
    y = df_sample[label_col].values
    X_sc   = StandardScaler().fit_transform(X)
    tsne   = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500)
    X_tsne = tsne.fit_transform(X_sc)

    fig, ax = plt.subplots(figsize=(8, 6))
    for flag, label, color in [(0, "Normal", "#2ecc71"), (1, "Theft", "#e74c3c")]:
        mask = y == flag
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   alpha=0.4, s=10, label=label, color=color)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE 2D Projection — Normal vs Theft (n=3000)", fontsize=13, fontweight="bold")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "16_tsne_projection.png"), dpi=150)
    plt.close()
    print("  Saved: 16_tsne_projection.png")

def print_summary_stats(df, label_col):
    print("\n--- Summary Statistics ---")
    print(f"Total consumers : {len(df)}")
    print(f"Normal (0)      : {(df[label_col]==0).sum()} ({(df[label_col]==0).mean()*100:.1f}%)")
    print(f"Theft  (1)      : {(df[label_col]==1).sum()} ({(df[label_col]==1).mean()*100:.1f}%)")
    print("\nFeature statistics (mean per class):")
    feature_cols = [c for c in df.columns if c != label_col]
    print(df.groupby(label_col)[feature_cols].mean().T.rename(
        columns={0: "Normal", 1: "Theft"}).to_string())

def run_eda():
    config    = load_config()
    feat_path = config["paths"]["processed_data"]
    label_col = config["data"]["label_column"]
    save_dir  = config["paths"]["figures_dir"]

    ensure_dir(save_dir)

    print("\n=== Loading features ===")
    df = load_features(feat_path)

    print_summary_stats(df, label_col)

    print("\n=== Generating plots ===")
    plot_class_distribution(df, label_col, save_dir)
    plot_consumption_distribution(df, label_col, save_dir)
    plot_feature_boxplots(df, label_col, save_dir)
    plot_correlation_heatmap(df, label_col, save_dir)
    plot_sudden_changes(df, label_col, save_dir)
    plot_cv_vs_mean(df, label_col, save_dir)
    plot_load_factor(df, label_col, save_dir)
    plot_feature_correlation_with_label(df, label_col, save_dir)

    print("\n=== Dimensionality Reduction Plots ===")
    plot_pca(df, label_col, save_dir)
    print("  Running t-SNE (may take ~30 seconds) ...")
    plot_tsne(df, label_col, save_dir)

    print(f"\nAll 10 plots saved to: {save_dir}")
    print("EDA complete.")

if __name__ == "__main__":
    run_eda()
