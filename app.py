from pathlib import Path
import hmac
import math

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed_readmission.csv"
MODEL_PATH = ROOT / "readmission_model_calibrated.joblib"

TARGET_COLUMN = "readmitted_within_30_days"
GROUP_COLUMN = "patient_nbr"
ID_COLUMNS = ["encounter_id", GROUP_COLUMN, TARGET_COLUMN]

DIAGNOSIS_COLUMNS = ["diag_1", "diag_2", "diag_3"]
VISIT_COLUMNS = [
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]


st.set_page_config(
    page_title="Hospital Readmission Dashboard",
    page_icon="🏥",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    section.main {
        background: #080b10 !important;
        color: #e6edf3 !important;
    }

    [data-testid="stHeader"] {
        background: #080b10 !important;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        background: #0c1420 !important;
        border-right: 1px solid #263345;
    }

    [data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    [data-testid="stMetric"] {
        background: #141d29 !important;
        border: 1px solid #263345 !important;
        border-radius: 14px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #aab8ca !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #f4f7fb !important;
    }

    div[data-testid="stForm"],
    [data-testid="stExpander"] {
        background: #111923 !important;
        border: 1px solid #263345 !important;
        border-radius: 14px;
    }

    div[data-testid="stForm"] {
        padding: 22px;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: #111923 !important;
        color: #e6edf3 !important;
        border: 1px solid #334155 !important;
    }

    [data-testid="stSelectbox"] div,
    [data-testid="stMultiSelect"] div {
        color: #e6edf3;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #263345;
        border-radius: 10px;
    }

    .main-title,
    h1,
    h2,
    h3 {
        color: #f4f7fb !important;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        color: #aab8ca !important;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 9px;
        font-weight: 600;
        border: 1px solid #34445a;
    }

    .login-hero {
        min-height: 440px;
        padding: 42px;
        border: 1px solid #263345;
        border-radius: 22px;
        background:
            radial-gradient(circle at 85% 15%, #183754 0, transparent 38%),
            linear-gradient(145deg, #111d2b, #0b111a 72%);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .login-label {
        color: #82c8ff;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .login-heading {
        color: #f4f7fb;
        font-size: 2.5rem;
        line-height: 1.15;
        font-weight: 750;
        margin-top: 20px;
    }

    .login-copy {
        color: #aab8ca;
        font-size: 1.05rem;
        line-height: 1.7;
        margin-top: 16px;
    }

    .login-foot {
        color: #8292a7;
        font-size: 0.9rem;
        border-top: 1px solid #263345;
        padding-top: 18px;
        margin-top: 35px;
    }

    [data-testid="stProgressBar"] > div > div {
        background-color: #58a6ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def require_demo_login():
    if st.session_state.get("logged_in", False):
        return

    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.markdown(
            """
            <div class="login-hero">
                <div>
                    <div class="login-label">Readmission Insights</div>
                    <div class="login-heading">
                        Hospital encounter<br>dashboard
                    </div>
                    <div class="login-copy">
                        Explore encounter patterns, look up patient records,
                        and review readmission risk estimates from the
                        project dataset.
                    </div>
                </div>
                <div class="login-foot">
                     Dashboard access
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("## Welcome back")
        st.write("Sign in to open the dashboard.")

        try:
            expected_username = st.secrets["demo_login"]["username"]
            expected_password = st.secrets["demo_login"]["password"]
        except Exception:
            st.error(
                "Demo login is not configured. Create "
                ".streamlit/secrets.toml with a username and password."
            )
            st.stop()

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Sign in",
                use_container_width=True,
            )

        if submitted:
            username_matches = hmac.compare_digest(
                username,
                str(expected_username),
            )
            password_matches = hmac.compare_digest(
                password,
                str(expected_password),
            )

            if username_matches and password_matches:
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Username or password is incorrect.")

    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_project_data():
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["prior_visits_total"] = data[VISIT_COLUMNS].sum(axis=1)

    feature_data = data.drop(columns=ID_COLUMNS, errors="ignore")
    feature_columns = feature_data.columns.tolist()

    numeric_columns = feature_data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in feature_columns
        if column not in numeric_columns
    ]

    reference_values = {}

    for column in numeric_columns:
        reference_values[column] = feature_data[column].median()

    for column in categorical_columns:
        modes = feature_data[column].mode(dropna=True)
        reference_values[column] = (
            modes.iloc[0] if not modes.empty else ""
        )

    
    groups = data[GROUP_COLUMN]
    y = data[TARGET_COLUMN]
    X = data.drop(columns=ID_COLUMNS, errors="ignore")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )
    _, test_indices = next(splitter.split(X, y, groups=groups))

    return (
        data,
        feature_columns,
        numeric_columns,
        categorical_columns,
        reference_values,
        set(test_indices.tolist()),
    )


def prepare_model_input(
    source,
    feature_columns,
    numeric_columns,
    categorical_columns,
):
    patient = source.reindex(columns=feature_columns).copy()

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

    return patient


def explain_record(
    model,
    patient,
    feature_columns,
    categorical_columns,
    reference_values,
    base_probability,
):
    effects = []

    for column in feature_columns:
        if column == "prior_visits_total":
            continue

        changed_patient = patient.copy()
        patient_value = changed_patient.iloc[0][column]
        typical_value = reference_values[column]

        if column in categorical_columns:
            changed_patient[column] = pd.Series(
                [typical_value],
                index=changed_patient.index,
                dtype="object",
            )
        else:
            changed_patient.at[
                changed_patient.index[0],
                column,
            ] = typical_value

        if column in VISIT_COLUMNS:
            changed_patient["prior_visits_total"] = (
                changed_patient[VISIT_COLUMNS].fillna(0).sum(axis=1)
            )

        changed_probability = float(
            model.predict_proba(changed_patient)[0, 1]
        )

        effects.append(
            {
                "Factor": column,
                "Patient value": patient_value,
                "Typical value": typical_value,
                "Score shift (percentage points)": round(
                    (base_probability - changed_probability) * 100,
                    2,
                ),
            }
        )

    result = pd.DataFrame(effects)
    result["Absolute shift"] = result[
        "Score shift (percentage points)"
    ].abs()

    return (
        result.sort_values("Absolute shift", ascending=False)
        .drop(columns=["Absolute shift"])
        .reset_index(drop=True)
    )


def show_patient(
    row_index,
    data,
    model,
    feature_columns,
    numeric_columns,
    categorical_columns,
    reference_values,
    test_indices,
):
    record = data.loc[row_index]

    source = (
        record.drop(labels=[TARGET_COLUMN], errors="ignore")
        .to_frame()
        .T
    )

    model_input = prepare_model_input(
        source,
        feature_columns,
        numeric_columns,
        categorical_columns,
    )

    probability = float(model.predict_proba(model_input)[0, 1])

    if probability < 0.10:
        risk_category = "Low"
    elif probability < 0.20:
        risk_category = "Moderate"
    else:
        risk_category = "High"

    st.markdown("### Encounter summary")

    metrics = st.columns(4)
    metrics[0].metric("Encounter ID", str(record["encounter_id"]))
    metrics[1].metric("Patient ID", str(record[GROUP_COLUMN]))
    metrics[2].metric(
        "Historical outcome",
        "Within 30 days" if int(record[TARGET_COLUMN]) == 1 else "No",
    )
    metrics[3].metric("Predicted risk band", risk_category)

    st.progress(
        min(max(probability, 0.0), 1.0),
        text=f"Estimated probability: {probability:.1%}",
    )

    if row_index in test_indices:
        st.success(
            "This patient's records were held out during model training."
        )
    else:
        st.info(
            "This patient's records were part of model training, so this "
            "prediction is an in-sample example."
        )

    with st.expander("View all fields for this encounter"):
        details = (
            record.drop(labels=[TARGET_COLUMN], errors="ignore")
            .rename("Value")
            .rename_axis("Field")
            .reset_index()
        )
        st.dataframe(
            details,
            hide_index=True,
            use_container_width=True,
        )

    factors = explain_record(
        model,
        model_input,
        feature_columns,
        categorical_columns,
        reference_values,
        probability,
    )

    st.markdown("### Factors shifting the prediction")
    st.caption(
        "A positive shift means the recorded value raises the model score "
        "compared with a typical dataset value."
    )
    st.dataframe(
        factors.head(10),
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download factor details",
        data=factors.to_csv(index=False),
        file_name=f"encounter_{record['encounter_id']}_factors.csv",
        mime="text/csv",
    )

    with st.expander("How this prediction was generated"):
        st.write(
            "This estimate was generated by the project's trained, calibrated "
            "machine-learning model loaded from "
            "`readmission_model_calibrated.joblib`."
        )
        st.write(
            "The app sends this encounter's prepared features to the model and "
            "uses its predicted probability to assign the risk category."
        )

        st.markdown("**Evaluation on the patient-grouped test set**")
        metric_columns = st.columns(3)
        metric_columns[0].metric("ROC-AUC", "0.630")
        metric_columns[1].metric("PR-AUC", "0.182")
        metric_columns[2].metric("Brier score", "0.0929")

        st.caption(
            "These are test-set metrics, not a guarantee for an individual "
            "prediction. Accuracy alone is not shown because most encounters "
            "were not readmitted within 30 days."
        )


def render_patient_search(
    data,
    model,
    feature_columns,
    numeric_columns,
    categorical_columns,
    reference_values,
    test_indices,
):
    st.subheader("Patient search")
    search_type = st.radio(
        "Search using",
        ["Patient ID", "Encounter ID"],
        horizontal=True,
    )

    search_column = (
        GROUP_COLUMN if search_type == "Patient ID" else "encounter_id"
    )

    query = st.text_input(
        search_type,
        placeholder="Enter all or part of an ID",
    )

    if not query:
        st.info("Enter a patient or encounter ID to search.")
        return

    matches_mask = (
        data[search_column]
        .astype("string")
        .str.contains(
            query.strip(),
            case=False,
            regex=False,
            na=False,
        )
    )
    matches = data.loc[matches_mask]

    st.write(f"Found **{len(matches):,}** matching encounters.")

    if matches.empty:
        st.warning("No matching records.")
        return

    filters = st.columns(2)
    age_options = sorted(matches["age"].dropna().astype(str).unique())
    gender_options = sorted(matches["gender"].dropna().astype(str).unique())

    chosen_ages = filters[0].multiselect("Filter by age group", age_options)
    chosen_genders = filters[1].multiselect(
        "Filter by gender",
        gender_options,
    )

    if chosen_ages:
        matches = matches[matches["age"].astype(str).isin(chosen_ages)]
    if chosen_genders:
        matches = matches[
            matches["gender"].astype(str).isin(chosen_genders)
        ]

    if matches.empty:
        st.warning("No records match those filters.")
        return

    shown = matches.head(200)
    preview_columns = [
        "encounter_id",
        GROUP_COLUMN,
        "age",
        "gender",
        "diag_1",
        TARGET_COLUMN,
    ]

    st.dataframe(
        shown[preview_columns],
        hide_index=True,
        use_container_width=True,
    )

    selected_index = st.selectbox(
        "Choose an encounter",
        shown.index.tolist(),
        format_func=lambda index: (
            f"Encounter {data.at[index, 'encounter_id']} "
            f"· Patient {data.at[index, GROUP_COLUMN]}"
        ),
    )

    show_patient(
        selected_index,
        data,
        model,
        feature_columns,
        numeric_columns,
        categorical_columns,
        reference_values,
        test_indices,
    )


def render_diagnosis_search(
    data,
    model,
    feature_columns,
    numeric_columns,
    categorical_columns,
    reference_values,
    test_indices,
):
    st.subheader("Diagnosis search")
    st.caption(
        "Search by diagnosis code. The dataset stores codes rather than "
        "disease names."
    )

    query = st.text_input(
        "Diagnosis code",
        placeholder="For example: 250.7 or 428",
    )

    if not query:
        st.info("Enter a diagnosis code to find matching encounters.")
        return

    code_matches = (
        data[DIAGNOSIS_COLUMNS]
        .astype("string")
        .apply(
            lambda column: column.str.contains(
                query.strip(),
                case=False,
                regex=False,
                na=False,
            )
        )
        .any(axis=1)
    )

    matches = data.loc[code_matches]
    st.write(f"Found **{len(matches):,}** matching encounters.")

    if matches.empty:
        st.warning("No encounters matched that code.")
        return

    summary_columns = st.columns(2)
    summary_columns[0].metric(
        "Historical readmission rate",
        f"{matches[TARGET_COLUMN].mean():.1%}",
    )
    summary_columns[1].metric(
        "Patient IDs",
        f"{matches[GROUP_COLUMN].nunique():,}",
    )

    shown = matches.head(200)
    preview_columns = [
        "encounter_id",
        GROUP_COLUMN,
        "age",
        "gender",
        *DIAGNOSIS_COLUMNS,
        TARGET_COLUMN,
    ]

    st.dataframe(
        shown[preview_columns],
        hide_index=True,
        use_container_width=True,
    )

    selected_index = st.selectbox(
        "Choose an encounter to view",
        shown.index.tolist(),
        format_func=lambda index: (
            f"Encounter {data.at[index, 'encounter_id']} "
            f"· Patient {data.at[index, GROUP_COLUMN]}"
        ),
        key="diagnosis_selected_encounter",
    )

    show_patient(
        selected_index,
        data,
        model,
        feature_columns,
        numeric_columns,
        categorical_columns,
        reference_values,
        test_indices,
    )


def render_full_dataset(data):
    st.subheader("Full patient encounter dataset")
    st.write(
        f"Browse all **{len(data):,}** encounters. Use filters to narrow "
        "the table, then move through pages."
    )

    filter_columns = st.columns(4)

    id_query = filter_columns[0].text_input(
        "Patient or encounter ID",
        placeholder="Optional",
    )
    diagnosis_query = filter_columns[1].text_input(
        "Diagnosis code",
        placeholder="Optional",
    )
    age_options = sorted(data["age"].dropna().astype(str).unique())
    age_filter = filter_columns[2].multiselect(
        "Age groups",
        age_options,
    )
    outcome_filter = filter_columns[3].selectbox(
        "Historical outcome",
        [
            "All",
            "Within 30 days",
            "Not within 30 days",
        ],
    )

    filtered = data

    if id_query.strip():
        patient_match = (
            filtered[GROUP_COLUMN]
            .astype("string")
            .str.contains(
                id_query.strip(),
                case=False,
                regex=False,
                na=False,
            )
        )
        encounter_match = (
            filtered["encounter_id"]
            .astype("string")
            .str.contains(
                id_query.strip(),
                case=False,
                regex=False,
                na=False,
            )
        )
        filtered = filtered.loc[patient_match | encounter_match]

    if diagnosis_query.strip():
        code_match = (
            filtered[DIAGNOSIS_COLUMNS]
            .astype("string")
            .apply(
                lambda column: column.str.contains(
                    diagnosis_query.strip(),
                    case=False,
                    regex=False,
                    na=False,
                )
            )
            .any(axis=1)
        )
        filtered = filtered.loc[code_match]

    if age_filter:
        filtered = filtered[
            filtered["age"].astype(str).isin(age_filter)
        ]

    if outcome_filter == "Within 30 days":
        filtered = filtered[filtered[TARGET_COLUMN] == 1]
    elif outcome_filter == "Not within 30 days":
        filtered = filtered[filtered[TARGET_COLUMN] == 0]

    st.write(f"Showing **{len(filtered):,}** matching encounters.")

    page_size = st.selectbox(
        "Rows per page",
        [50, 100, 250, 500],
        index=1,
    )

    page_key = "full_dataset_page"
    total_pages = max(1, math.ceil(len(filtered) / page_size))

    if st.session_state.get(page_key, 1) > total_pages:
        st.session_state[page_key] = 1
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    previous_column, page_column, next_column = st.columns([1, 2, 1])

    with previous_column:
        if st.button(
            "← Previous",
            disabled=st.session_state[page_key] <= 1,
        ):
            st.session_state[page_key] -= 1

    with next_column:
        if st.button(
            "Next →",
            disabled=st.session_state[page_key] >= total_pages,
        ):
            st.session_state[page_key] += 1

    current_page = st.session_state[page_key]
    page_column.markdown(
        f"<div style='text-align:center;padding-top:8px;'>"
        f"Page {current_page} of {total_pages}</div>",
        unsafe_allow_html=True,
    )

    start = (current_page - 1) * page_size
    end = start + page_size

    st.dataframe(
        filtered.iloc[start:end],
        hide_index=True,
        height=520,
        use_container_width=True,
    )

    st.download_button(
        "Download filtered records",
        data=filtered.to_csv(index=False),
        file_name="filtered_patient_encounters.csv",
        mime="text/csv",
    )


def render_unseen_data_test(
    data,
    model,
    feature_columns,
    numeric_columns,
    categorical_columns,
    test_indices,
):
    st.subheader("Test the model on unseen encounters")
    st.write(
        "These encounters belong to the patient-grouped test set held out "
        "during training. The actual outcome stays hidden until you choose "
        "to reveal it."
    )

    if not test_indices:
        st.error("No held-out test encounters were found.")
        return

    if st.button("Select a random held-out encounter"):
        selected_index = int(
            pd.Series(sorted(test_indices)).sample(n=1).iloc[0]
        )
        st.session_state["unseen_test_index"] = selected_index

    selected_index = st.session_state.get("unseen_test_index")

    if selected_index is None or selected_index not in test_indices:
        st.info("Click the button to select an encounter for testing.")
        return

    record = data.loc[selected_index]

    # Exclude the actual outcome from model input.
    source = (
        record.drop(labels=[TARGET_COLUMN], errors="ignore")
        .to_frame()
        .T
    )
    model_input = prepare_model_input(
        source,
        feature_columns,
        numeric_columns,
        categorical_columns,
    )

    probability = float(model.predict_proba(model_input)[0, 1])

    if probability < 0.10:
        risk_category = "Low"
    elif probability < 0.20:
        risk_category = "Moderate"
    else:
        risk_category = "High"

    predicted_outcome = int(probability >= 0.50)

    st.markdown("### Model prediction")
    result_columns = st.columns(3)
    result_columns[0].metric(
        "Encounter ID",
        str(record["encounter_id"]),
    )
    result_columns[1].metric(
        "Readmission probability",
        f"{probability:.1%}",
    )
    result_columns[2].metric("Risk category", risk_category)

    st.progress(
        min(max(probability, 0.0), 1.0),
        text=f"Estimated probability: {probability:.1%}",
    )

    st.caption(
        "The binary prediction uses a 0.50 probability threshold. "
        "Risk categories use the project's Low, Moderate, and High bands."
    )

    reveal_key = f"reveal_test_outcome_{selected_index}"

    if st.button("Reveal actual outcome", key=reveal_key):
        actual_outcome = int(record[TARGET_COLUMN])

        st.markdown("### Actual outcome")
        actual_label = (
            "Readmitted within 30 days"
            if actual_outcome == 1
            else "Not readmitted within 30 days"
        )
        predicted_label = (
            "Readmitted within 30 days"
            if predicted_outcome == 1
            else "Not readmitted within 30 days"
        )

        st.write(f"**Actual:** {actual_label}")
        st.write(f"**Model's binary prediction:** {predicted_label}")

        if predicted_outcome == actual_outcome:
            st.success("The binary prediction matched this encounter's outcome.")
        else:
            st.warning(
                "The binary prediction did not match this encounter's outcome."
            )

        st.caption(
            "One encounter is only a demonstration. Use the test-set metrics "
            "to assess overall model performance."
        )


def render_overview(data):
    st.markdown(
        "<div class='main-title'>Hospital Readmission Dashboard</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='main-subtitle'>"
        "Patient encounter overview and readmission patterns"
        "</div>",
        unsafe_allow_html=True,
    )

    total_encounters = len(data)
    unique_patients = data[GROUP_COLUMN].nunique()
    readmission_rate = data[TARGET_COLUMN].mean()

    metrics = st.columns(3)
    metrics[0].metric("Total encounters", f"{total_encounters:,}")
    metrics[1].metric("Unique patient IDs", f"{unique_patients:,}")
    metrics[2].metric(
        "Readmitted within 30 days",
        f"{readmission_rate:.1%}",
    )

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Encounter outcomes")
        outcome_counts = (
            data[TARGET_COLUMN]
            .map({0: "Not within 30 days", 1: "Within 30 days"})
            .value_counts()
            .rename_axis("Outcome")
            .to_frame("Encounters")
        )
        st.bar_chart(outcome_counts)

    with right:
        st.markdown("### Readmission rate by age group")
        age_summary = (
            data.groupby("age", dropna=False)[TARGET_COLUMN]
            .mean()
            .mul(100)
            .round(1)
            .to_frame("Readmission rate (%)")
        )
        st.bar_chart(age_summary)

    st.markdown("### Primary diagnosis codes with highest observed rates")
    diagnosis_summary = (
        data.dropna(subset=["diag_1"])
        .groupby("diag_1")[TARGET_COLUMN]
        .agg(Encounters="count", Readmission_rate="mean")
        .reset_index()
    )
    diagnosis_summary = diagnosis_summary[
        diagnosis_summary["Encounters"] >= 100
    ].copy()
    diagnosis_summary["Readmission rate (%)"] = (
        diagnosis_summary.pop("Readmission_rate") * 100
    ).round(1)

    st.dataframe(
        diagnosis_summary.sort_values(
            "Readmission rate (%)",
            ascending=False,
        ).head(10),
        hide_index=True,
        use_container_width=True,
    )


def main():
    require_demo_login()

    if not DATA_PATH.exists():
        st.error(f"Processed dataset not found: {DATA_PATH}")
        st.stop()

    if not MODEL_PATH.exists():
        st.error(f"Trained model not found: {MODEL_PATH}")
        st.stop()

    try:
        (
            data,
            feature_columns,
            numeric_columns,
            categorical_columns,
            reference_values,
            test_indices,
        ) = load_project_data()
        model = load_model()
    except Exception as error:
        st.error(f"Could not load project files: {error}")
        st.stop()

    st.sidebar.markdown("## 🏥 Readmission Insights")
    st.sidebar.caption("Patient encounter dashboard")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Patient search",
            "Diagnosis search",
            "Full dataset",
            "Unseen data test",
        ],
    )

    st.sidebar.divider()
    st.sidebar.caption(f"{len(data):,} encounters available")

    if st.sidebar.button("Log out", use_container_width=True):
        st.session_state["logged_in"] = False
        st.rerun()

    if page == "Overview":
        render_overview(data)

    elif page == "Patient search":
        st.markdown("## Patient search")
        render_patient_search(
            data,
            model,
            feature_columns,
            numeric_columns,
            categorical_columns,
            reference_values,
            test_indices,
        )

    elif page == "Diagnosis search":
        st.markdown("## Diagnosis search")
        render_diagnosis_search(
            data,
            model,
            feature_columns,
            numeric_columns,
            categorical_columns,
            reference_values,
            test_indices,
        )

    elif page == "Unseen data test":
        render_unseen_data_test(
            data,
            model,
            feature_columns,
            numeric_columns,
            categorical_columns,
            test_indices,
        )

    else:
        st.markdown("## Full dataset")
        render_full_dataset(data)


if __name__ == "__main__":
    main()