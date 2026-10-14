import pandas as pd
import numpy as np
import yaml
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_raw_data(path):
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    print(f"  Raw shape: {df.shape}")
    return df

def separate_features_label(df, label_col):
    """Split daily consumption columns from the FLAG label."""
    y = df[label_col].copy()
    drop_cols = [label_col] + [c for c in ["CONS_NO"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    print(f"  Consumption columns: {X.shape[1]}  |  Label distribution:\n{y.value_counts().to_string()}")
    return X, y

def drop_high_missing_rows(X, y, threshold):
    """Drop consumers where more than `threshold` fraction of readings are missing."""
    missing_frac = X.isnull().mean(axis=1)
    mask = missing_frac <= threshold
    dropped = (~mask).sum()
    print(f"  Rows dropped (>{threshold*100:.0f}% missing): {dropped}")
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

def fill_missing_values(X):
    """Forward-fill then backward-fill remaining missing values."""
    before = X.isnull().sum().sum()
    X = X.ffill(axis=1).bfill(axis=1)
    after = X.isnull().sum().sum()
    print(f"  Missing values: {before} -> {after}")
    return X

def remove_duplicates(X, y):
    df = pd.concat([X, y], axis=1)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  Duplicates removed: {before - len(df)}")
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    return X, y

def cap_outliers(X, percentile):
    """Cap each column at the given percentile to reduce extreme values."""
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    cap_values = X.quantile(percentile / 100.0)
    X = X.clip(upper=cap_values, axis=1)
    print(f"  Outliers capped at {percentile}th percentile")
    return X

def save_cleaned_data(X, y, label_col, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.concat([X, y.rename(label_col)], axis=1)
    df.to_csv(output_path, index=False)
    print(f"  Cleaned data saved to {output_path}  |  Shape: {df.shape}")

def run_preprocessing():
    config = load_config()

    raw_path     = config["paths"]["raw_data"]
    output_path  = "data/processed/cleaned_data.csv"
    label_col    = config["data"]["label_column"]
    miss_thresh  = config["preprocessing"]["missing_threshold"]
    outlier_cap  = config["preprocessing"]["outlier_cap"]

    print("\n=== Step 1: Load ===")
    df = load_raw_data(raw_path)

    print("\n=== Step 2: Separate features and label ===")
    X, y = separate_features_label(df, label_col)

    print("\n=== Step 3: Drop high-missing rows ===")
    X, y = drop_high_missing_rows(X, y, miss_thresh)

    print("\n=== Step 4: Fill remaining missing values ===")
    X = fill_missing_values(X)

    print("\n=== Step 5: Remove duplicates ===")
    X, y = remove_duplicates(X, y)

    print("\n=== Step 6: Cap outliers ===")
    X = cap_outliers(X, outlier_cap)

    print("\n=== Step 7: Save ===")
    save_cleaned_data(X, y, label_col, output_path)

    print("\nPreprocessing complete.")
    return output_path

if __name__ == "__main__":
    run_preprocessing()
