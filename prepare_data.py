from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/diabetic_data.csv")
OUTPUT_PATH = Path("data/processed_readmission.csv")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}. "
            "Check that diabetic_data.csv is in the data folder."
        )

    data = pd.read_csv(INPUT_PATH)

    data = data.replace("?", pd.NA)

    data["readmitted_within_30_days"] = (
        data["readmitted"] == "<30"
    ).astype(int)

    
    data = data.drop(columns=["readmitted"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Prepared data saved to: {OUTPUT_PATH}")
    print(f"Rows: {len(data)}")
    print("\nTarget counts (0 = not within 30 days, 1 = within 30 days):")
    print(data["readmitted_within_30_days"].value_counts().sort_index())

    print("\nColumns with the most missing values:")
    print(data.isna().sum().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()