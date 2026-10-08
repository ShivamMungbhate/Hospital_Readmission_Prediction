from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path("readmission_model.joblib")
OUTPUT_PATH = Path("model_factors.csv")


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run train_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    preprocessing = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessing.get_feature_names_out()
    coefficients = classifier.coef_[0]

    factors = pd.DataFrame(
        {
            "factor": feature_names,
            "coefficient": coefficients,
        }
    )

    factors = factors.sort_values("coefficient", ascending=False)

    print("\nFactors associated with higher model scores:")
    print(factors.head(15).to_string(index=False))

    print("\nFactors associated with lower model scores:")
    print(factors.tail(15).sort_values("coefficient").to_string(index=False))

    factors.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAll factors saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()