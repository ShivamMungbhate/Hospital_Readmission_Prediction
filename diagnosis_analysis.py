from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed_readmission.csv")
TARGET_COLUMN = "readmitted_within_30_days"
DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]
MIN_RECORDS = 100


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {DATA_PATH}. "
            "Run prepare_data.py first."
        )

    data = pd.read_csv(DATA_PATH, low_memory=False)

    for column in DIAGNOSIS_COLUMNS:
        summary = (
            data.dropna(subset=[column])
            .groupby(column)[TARGET_COLUMN]
            .agg(records="count", readmission_rate="mean")
            .reset_index()
        )

        summary = summary[summary["records"] >= MIN_RECORDS].copy()
        summary["readmission_rate_percent"] = (
            summary["readmission_rate"] * 100
        ).round(1)

        print(f"\n--- {column}: highest observed rates ---")
        print(
            summary.sort_values(
                "readmission_rate", ascending=False
            )
            .head(10)[
                [column, "records", "readmission_rate_percent"]
            ]
            .to_string(index=False)
        )

        print(f"\n--- {column}: lowest observed rates ---")
        print(
            summary.sort_values("readmission_rate")
            .head(10)[
                [column, "records", "readmission_rate_percent"]
            ]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()