from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


DATA_PATH = Path("data/processed_readmission.csv")
MODEL_PATH = Path("readmission_model_calibrated.joblib")

TARGET_COLUMN = "readmitted_within_30_days"
GROUP_COLUMN = "patient_nbr"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {DATA_PATH}. "
            "Run prepare_data.py first."
        )

    data = pd.read_csv(DATA_PATH, low_memory=False)

    visit_columns = [
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
    ]
    data["prior_visits_total"] = data[visit_columns].sum(axis=1)

    groups = data[GROUP_COLUMN]
    y = data[TARGET_COLUMN]
    X = data.drop(
        columns=[TARGET_COLUMN, GROUP_COLUMN, "encounter_id"],
        errors="ignore",
    )

    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )
    train_indices, test_indices = next(
        outer_split.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    groups_train = groups.iloc[train_indices]

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()
    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    base_model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    grouped_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    calibration_splits = list(
        grouped_cv.split(X_train, y_train, groups=groups_train)
    )

    model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=calibration_splits,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"Actual test readmission rate: {y_test.mean():.1%}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_test, probabilities):.3f}")
    print(
        f"Brier score (lower is better): "
        f"{brier_score_loss(y_test, probabilities):.4f}"
    )
    print(
        f"Accuracy at a 0.50 threshold: "
        f"{(predictions == y_test.to_numpy()).mean():.1%}"
    )

    print("\nConfusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report at 0.50 threshold:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Not within 30 days", "Within 30 days"],
            zero_division=0,
        )
    )
    results = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted_probability": probabilities,
        }
    )
    results["risk_category"] = pd.cut(
        results["predicted_probability"],
        bins=[0.0, 0.10, 0.20, 1.000001],
        labels=["Low", "Moderate", "High"],
        right=False,
        include_lowest=True,
    )

    results["risk_category"] = results["risk_category"].cat.add_categories(
        ["No records"]
    ).fillna("No records")

    band_summary = (
        results.groupby("risk_category", observed=False)
        .agg(
            records=("actual", "size"),
            average_predicted_probability=("predicted_probability", "mean"),
            observed_readmission_rate=("actual", "mean"),
        )
    )

    print("\nDemonstration risk-band check:")
    print(band_summary.round(3))

    joblib.dump(model, MODEL_PATH)
    print(f"\nCalibrated model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()