from pathlib import Path

import joblib
import pandas as pd


DATA_PATH = Path("data/processed_readmission.csv")
MODEL_PATH = Path("readmission_model_calibrated.joblib")
TEMPLATE_PATH = Path("data/new_patient_template.csv")
INPUT_PATH = Path("data/new_patient.csv")
RESULT_PATH = Path("data/prediction_result.csv")

TARGET_COLUMN = "readmitted_within_30_days"
ID_COLUMNS = ["encounter_id", "patient_nbr", TARGET_COLUMN]

VISIT_COLUMNS = [
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]


def create_template():
    reference = pd.read_csv(DATA_PATH, nrows=1, low_memory=False)
    reference["prior_visits_total"] = 0

    feature_columns = [
        column for column in reference.columns
        if column not in ID_COLUMNS
    ]

    template = pd.DataFrame(
        [{column: "" for column in feature_columns}]
    )
    template.to_csv(TEMPLATE_PATH, index=False)

    print(f"Template created: {TEMPLATE_PATH}")
    print(f"Fill in one record and save a copy as: {INPUT_PATH}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {DATA_PATH}. "
            "Run prepare_data.py first."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Calibrated model not found: {MODEL_PATH}. "
            "Run train_model.py first."
        )

    if not INPUT_PATH.exists():
        create_template()
        return

    reference = pd.read_csv(DATA_PATH, low_memory=False)
    reference["prior_visits_total"] = 0

    feature_columns = [
        column for column in reference.columns
        if column not in ID_COLUMNS
    ]

    numeric_columns = reference[feature_columns].select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column for column in feature_columns
        if column not in numeric_columns
    ]

    patient = pd.read_csv(INPUT_PATH, low_memory=False)

    if len(patient) != 1:
        raise ValueError(
            "new_patient.csv must contain exactly one patient record."
        )

    patient = patient.reindex(columns=feature_columns)

    
    for column in numeric_columns:
        patient[column] = pd.to_numeric(
            patient[column],
            errors="coerce",
        )

    for column in categorical_columns:
        patient[column] = patient[column].map(
            lambda value: str(value) if pd.notna(value) else float("nan")
        )

    
    for column in VISIT_COLUMNS:
        patient[column] = pd.to_numeric(
            patient[column],
            errors="coerce",
        )

    patient["prior_visits_total"] = (
        patient[VISIT_COLUMNS].fillna(0).sum(axis=1)
    )

    model = joblib.load(MODEL_PATH)
    probability = float(model.predict_proba(patient)[0, 1])

    if probability < 0.10:
        risk_category = "Low"
    elif probability < 0.20:
        risk_category = "Moderate"
    else:
        risk_category = "High"

    result = pd.DataFrame(
        [
            {
                "readmission_probability": round(probability, 4),
                "risk_category": risk_category,
            }
        ]
    )

    result.to_csv(RESULT_PATH, index=False)

    print(f"Readmission probability: {probability:.1%}")
    print(f"Risk category: {risk_category}")
    print(f"Result saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()