from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("data/processed_readmission.csv")
INPUT_PATH = Path("data/new_patient.csv")
ACTUAL_PATH = Path("data/demo_actual_outcome.txt")

TARGET_COLUMN = "readmitted_within_30_days"
GROUP_COLUMN = "patient_nbr"

VISIT_COLUMNS = [
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]

ID_COLUMNS = [
    "encounter_id",
    GROUP_COLUMN,
    TARGET_COLUMN,
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {DATA_PATH}. "
            "Run prepare_data.py first."
        )

    data = pd.read_csv(DATA_PATH, low_memory=False)

    groups = data[GROUP_COLUMN]
    y = data[TARGET_COLUMN]
    X = data.drop(columns=ID_COLUMNS, errors="ignore")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )
    _, test_indices = next(splitter.split(X, y, groups=groups))

    
    record = data.iloc[test_indices[0]].copy()
    actual_outcome = int(record[TARGET_COLUMN])

    patient = record.drop(labels=ID_COLUMNS, errors="ignore").to_frame().T
    patient["prior_visits_total"] = (
        pd.to_numeric(patient[VISIT_COLUMNS].iloc[0], errors="coerce")
        .fillna(0)
        .sum()
    )

    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    patient.to_csv(INPUT_PATH, index=False)
    ACTUAL_PATH.write_text(
        f"Actual outcome: {actual_outcome} "
        "(1 = within 30 days, 0 = not within 30 days)\n",
        encoding="utf-8",
    )

    print(f"Demo patient saved to: {INPUT_PATH}")
    print("Now run predict_patient.py.")
    print(f"Afterward, compare with the actual outcome in: {ACTUAL_PATH}")


if __name__ == "__main__":
    main()