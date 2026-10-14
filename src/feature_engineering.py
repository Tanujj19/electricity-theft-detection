import pandas as pd
import numpy as np
import yaml
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_cleaned_data(path):
    print(f"Loading cleaned data from {path} ...")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    return df

def get_consumption_columns(df, label_col):
    """Return only the daily consumption columns (exclude FLAG)."""
    return [c for c in df.columns if c != label_col]

def parse_dates(consumption_cols):
    """Parse column names as dates to identify weekdays/weekends."""
    dates = pd.to_datetime(consumption_cols, errors="coerce")
    return dates

def extract_features(df, consumption_cols, dates, rolling_window):
    """Extract 12 statistical features per consumer row."""
    X = df[consumption_cols].values.astype(float)

    mean_c   = np.mean(X, axis=1)
    std_c    = np.std(X, axis=1)
    max_c    = np.max(X, axis=1)
    min_c    = np.min(X, axis=1)

    # Coefficient of variation: std/mean — high value = erratic/suspicious usage
    cv = np.where(mean_c > 0, std_c / mean_c, 0)

    # Peak/off-peak ratio: avg of top 10% days / avg of bottom 10% days
    top10    = np.percentile(X, 90, axis=1)
    bot10    = np.percentile(X, 10, axis=1)
    peak_offpeak = np.where(bot10 > 0, top10 / bot10, top10)

    # Weekend vs weekday consumption
    weekend_mask  = np.array([d.weekday() >= 5 if not pd.isnull(d) else False for d in dates])
    weekday_mask  = ~weekend_mask
    weekend_mean  = np.mean(X[:, weekend_mask], axis=1) if weekend_mask.any() else np.zeros(len(X))
    weekday_mean  = np.mean(X[:, weekday_mask], axis=1) if weekday_mask.any() else np.zeros(len(X))
    wk_ratio      = np.where(weekday_mean > 0, weekend_mean / weekday_mean, 1.0)

    # Mean-to-peak ratio: low value means consumption is usually low but spikes high
    mean_to_peak  = np.where(max_c > 0, mean_c / max_c, 0)

    # Load factor: same as mean-to-peak but standard electricity term
    load_factor   = mean_to_peak.copy()

    # Sudden change count: days where consumption changed >50% vs previous day
    diff          = np.abs(np.diff(X, axis=1))
    prev          = X[:, :-1]
    pct_change    = np.where(prev > 0, diff / prev, 0)
    sudden_changes = np.sum(pct_change > 0.5, axis=1)

    # Rolling stats on last `rolling_window` days
    rolling_mean  = np.mean(X[:, -rolling_window:], axis=1)
    rolling_std   = np.std(X[:, -rolling_window:], axis=1)

    features = pd.DataFrame({
        "mean_consumption":    mean_c,
        "std_consumption":     std_c,
        "max_consumption":     max_c,
        "min_consumption":     min_c,
        "cv_consumption":      cv,
        "peak_offpeak_ratio":  peak_offpeak,
        "weekend_weekday_ratio": wk_ratio,
        "mean_to_peak_ratio":  mean_to_peak,
        "load_factor":         load_factor,
        "sudden_change_count": sudden_changes,
        "rolling_mean_7d":     rolling_mean,
        "rolling_std_7d":      rolling_std,
    })

    print(f"  Features extracted: {features.shape[1]} features x {features.shape[0]} consumers")
    return features

def save_features(features, y, label_col, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out = pd.concat([features, y.reset_index(drop=True).rename(label_col)], axis=1)
    df_out.to_csv(output_path, index=False)
    print(f"  Feature file saved to {output_path}  |  Shape: {df_out.shape}")
    print(f"  Label distribution:\n{df_out[label_col].value_counts().to_string()}")

def run_feature_engineering():
    config         = load_config()
    cleaned_path   = "data/processed/cleaned_data.csv"
    output_path    = config["paths"]["processed_data"]
    label_col      = config["data"]["label_column"]
    rolling_window = config["features"]["rolling_window"]

    print("\n=== Step 1: Load cleaned data ===")
    df = load_cleaned_data(cleaned_path)

    print("\n=== Step 2: Identify consumption columns ===")
    consumption_cols = get_consumption_columns(df, label_col)
    dates = parse_dates(consumption_cols)
    print(f"  Daily columns: {len(consumption_cols)}  |  Date range: {consumption_cols[0]} to {consumption_cols[-1]}")

    print("\n=== Step 3: Extract features ===")
    features = extract_features(df, consumption_cols, dates, rolling_window)

    print("\n=== Step 4: Save feature file ===")
    y = df[label_col]
    save_features(features, y, label_col, output_path)

    print("\nFeature engineering complete.")
    return output_path

if __name__ == "__main__":
    run_feature_engineering()
