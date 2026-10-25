# Hospital Patient Readmission Prediction System

A college AIML project that uses historical hospital encounter data to estimate the probability of readmission within 30 days. It includes a machine-learning workflow and a Streamlit dashboard for exploring encounter patterns and predictions.

## Live Dashboard

[Open the Hospital Readmission Dashboard](https://hospitalreadmissionprediction-auqt7wbvkufkzwpbdbcd7g.streamlit.app/)

**Demo login**

- Username: `Demo`
- Password: `1234`

These are demonstration credentials for the public project app.

## Project Objectives

- Prepare historical hospital encounter data for machine learning.
- Engineer features from existing encounter information.
- Train and evaluate a classification model.
- Show a readmission probability and a Low, Moderate, or High risk category.
- Explore encounter information, diagnosis codes, and model score factors in a dashboard.

## Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset. It contains 101,766 encounters and 71,518 unique patient IDs.

The target column, `readmitted_within_30_days`, is defined as:

- `1`: The original readmission label was `<30`.
- `0`: The original readmission label was `NO` or `>30`.

The dataset contains 11,357 positive examples, approximately 11.2% of all encounters.

## How It Works

1. The data preparation script handles missing-value markers and creates the binary readmission target.
2. Feature engineering adds `prior_visits_total`, the sum of prior outpatient, emergency, and inpatient visits.
3. A calibrated classification model is trained and evaluated using a patient-grouped split. Records for the same patient are kept together in either the training set or the test set.
4. The Streamlit app loads the trained model, prepares encounter features, and displays a predicted probability and risk category.

Risk categories shown by the dashboard:

- **Low:** below 10%
- **Moderate:** 10% to below 20%
- **High:** 20% or higher

## Dashboard Features

- **Overview:** Encounter counts, unique patient counts, readmission rate, and charts.
- **Patient search:** Search by patient ID or encounter ID.
- **Diagnosis search:** Search encounters by diagnosis code.
- **Full dataset:** Browse and filter encounter records.
- **Unseen data test:** Select an encounter from the held-out test set, view the prediction, then reveal its actual outcome.
- **Prediction factors:** View one-feature-at-a-time shifts in model score compared with typical dataset values.

Prediction-factor shifts describe model behavior; they do not establish cause and effect.

## Model Results

Reported results on the patient-grouped test set:

| Metric | Result |
|---|---:|
| Accuracy at a 0.50 threshold | 89.3% |
| ROC-AUC | 0.630 |
| PR-AUC | 0.182 |
| Brier score | 0.0929 |
| Readmission rate in the test set | 10.7% |



