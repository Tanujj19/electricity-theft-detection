import pandas as pd

for f in ["data set.csv", "datasetsmall.csv"]:
    print(f"\n=== {f} ===")
    df = pd.read_csv(f"data/raw/{f}")
    print("Shape:", df.shape)
    print("FLAG distribution:\n", df["FLAG"].value_counts())
    print("Missing values total:", df.isnull().sum().sum())
