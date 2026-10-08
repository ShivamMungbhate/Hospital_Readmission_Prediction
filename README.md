# Hospital Patient Readmission Prediction System

An educational machine learning project that predicts whether a patient record from the Diabetes 130-US Hospitals dataset is associated with readmission within 30 days.

The project demonstrates classification, feature engineering, patient-grouped validation, probability calibration, model evaluation, and prediction explanations.

## Project output

For one patient record, the system reports:

- A predicted probability of readmission within 30 days
- A Low, Moderate, or High demonstration risk category
- Factors that shift the model score compared with typical dataset values

The risk bands are demonstration thresholds, not clinical cutoffs. Factor shifts show model associations, not causes.

## Dataset

This project uses the [Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us+hospitals+for+years+1999-2008) from the UCI Machine Learning Repository. It contains hospital records from 1999–2008 and uses three original readmission labels: `NO`, `<30`, and `>30`.

Download and extract the dataset. Place `diabetic_data.csv` here:

```text
data/diabetic_data.csv