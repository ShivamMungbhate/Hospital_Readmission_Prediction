from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/diabetic_data.csv")
TARGET_COLUMN = "readmitted"


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Place your CSV file in the data folder, or update DATA_PATH.")
        return

    data = pd.read_csv(DATA_PATH)

    print("\n--- Dataset shape ---")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    print("\n--- Column names ---")
    print(data.columns.tolist())

    print("\n--- First five rows ---")
    print(data.head())

    print("\n--- Missing values per column ---")
    print(data.isnull().sum().sort_values(ascending=False).head(15))

    if TARGET_COLUMN in data.columns:
        print(f"\n--- Values in '{TARGET_COLUMN}' ---")
        print(data[TARGET_COLUMN].value_counts(dropna=False))
    else:
        print(f"\nTarget column '{TARGET_COLUMN}' was not found.")
        print("Update TARGET_COLUMN to match your dataset.")


if __name__ == "__main__":
    main()