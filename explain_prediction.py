from pathlib import Path

import joblib
import pandas as pd


DATA_PATH = Path("data/processed_readmission.csv")
MODEL_PATH = Path("readmission_model_calibrated.joblib")
INPUT_PATH = Path("data/new_patient.csv")
OUTPUT_PATH = Path("data/patient_factor_effects.csv")

TARGET_COLUMN = "readmitted_within_30_days"
ID_COLUMNS = ["encounter_id", "patient_nbr", TARGET_COLUMN]

VISIT_COLUMNS = [
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]


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
        raise FileNotFoundError(
            f"Patient input not found: {INPUT_PATH}. "
            "Run make_demo_patient.py first, or create the input CSV."
        )

    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["prior_visits_total"] = data[VISIT_COLUMNS].sum(axis=1)

    feature_columns = [
        column for column in data.columns
        if column not in ID_COLUMNS
    ]

    numeric_columns = data[feature_columns].select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column for column in feature_columns
        if column not in numeric_columns
    ]

    reference_values = {}

    for column in numeric_columns:
        reference_values[column] = data[column].median()

    for column in categorical_columns:
        modes = data[column].mode(dropna=True)
        reference_values[column] = (
            modes.iloc[0] if not modes.empty else ""
        )

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
    base_probability = float(model.predict_proba(patient)[0, 1])

    effects = []

    for column in feature_columns:
        if column == "prior_visits_total":
            continue

        changed_patient = patient.copy()
        original_value = changed_patient.iloc[0][column]
        reference_value = reference_values[column]

        if column in categorical_columns:
            changed_patient[column] = pd.Series(
                [reference_value],
                index=changed_patient.index,
                dtype="object",
            )
        else:
            changed_patient.at[
                changed_patient.index[0],
                column,
            ] = reference_value

        if column in VISIT_COLUMNS:
            for visit_column in VISIT_COLUMNS:
                changed_patient[visit_column] = pd.to_numeric(
                    changed_patient[visit_column],
                    errors="coerce",
                )

            changed_patient["prior_visits_total"] = (
                changed_patient[VISIT_COLUMNS].fillna(0).sum(axis=1)
            )

        changed_probability = float(
            model.predict_proba(changed_patient)[0, 1]
        )

        effects.append(
            {
                "factor": column,
                "patient_value": original_value,
                "comparison_value": reference_value,
                "score_shift_percentage_points": round(
                    (base_probability - changed_probability) * 100,
                    2,
                ),
            }
        )

    results = pd.DataFrame(effects)
    results["absolute_shift"] = results[
        "score_shift_percentage_points"
    ].abs()

    results = (
        results.sort_values("absolute_shift", ascending=False)
        .drop(columns=["absolute_shift"])
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"Patient score: {base_probability:.1%}")
    print("\nLargest one-at-a-time factor shifts:")
    print(results.head(10).to_string(index=False))
    print(f"\nAll factor shifts saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()