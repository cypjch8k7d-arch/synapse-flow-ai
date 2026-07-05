
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Synapse Flow AI",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Demo login only
# -----------------------------
DEMO_USERS = {
    "doctor1": "synapse123",
    "neurology": "brain123",
    "admin": "demo123",
}

def show_login():
    st.title("🧠 Synapse Flow AI")
    st.subheader("Doctor Login")

    st.warning(
        "Demo login only. Do not use real hospital usernames, passwords, or patient data."
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

    with st.expander("Demo credentials"):
        st.write("doctor1 / synapse123")
        st.write("neurology / brain123")
        st.write("admin / demo123")

    if login_button:
        if username in DEMO_USERS and DEMO_USERS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Incorrect demo username or password.")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    show_login()
    st.stop()

# -----------------------------
# Main app
# -----------------------------
st.title("🧠 Synapse Flow AI")
st.subheader("Cognitive Risk Review-Priority Assistant")

st.success(f"Logged in as: {st.session_state.get('username', 'doctor')}")

if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.rerun()

st.warning(
    "Clinical decision-support concept only. "
    "This tool does NOT diagnose dementia and does NOT replace clinical judgment."
)

# -----------------------------
# Load model, features, and demo patient database
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("synapse_flow_model.pkl")
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    return model, feature_columns

@st.cache_data
def load_demo_database():
    try:
        return pd.read_csv("patient_database.csv")
    except FileNotFoundError:
        return pd.DataFrame()

model, feature_columns = load_model()
demo_database = load_demo_database()

# -----------------------------
# Feature groups
# -----------------------------
MODEL_CATEGORICAL_COLS = {
    "Gender",
    "MARRIAGE_STATUS",
    "APOE_GENOTYPE",
    "APOE4_STATUS",
    "Dizziness",
    "Low_Energy",
    "Headache",
    "Insomia",
    "Dep_Mood",
    "Wandering",
    "Fall",
}

SUMMARY_FEATURES = [
    "Age",
    "Gender",
    "EDUCATION_YEARS",
    "MARRIAGE_STATUS",
    "APOE_GENOTYPE",
    "APOE4_STATUS",
    "APOE4_COUNT",
]

COGNITIVE_FEATURES = [
    "MMSE_TOTAL",
    "MOCA_TOTAL",
    "CDR_GLOBAL",
    "CDR_SUM_OF_BOXES",
    "FAQ_TOTAL",
    "ADAS_TOTSCORE",
    "ADAS_TOTAL13",
    "NPIQ_TOTAL",
    "LOGICAL_MEMORY_TOTAL",
    "AVLT_TOTAL",
    "AVLT_DELAY_30MIN",
    "AVLT_DELAY_TOTAL",
    "TRAILS_A_SCORE",
    "BNT_TOTAL",
    "CLOCK_SCORE",
    "COPY_SCORE",
]

SYMPTOM_FEATURES = {
    "Dizziness": "Dizziness",
    "Low_Energy": "Low energy",
    "Headache": "Headache",
    "Insomia": "Insomnia",
    "Dep_Mood": "Depressed mood",
    "Wandering": "Wandering",
    "Fall": "Falls",
}

IMAGING_FEATURES = [
    "TOTAL_HIPP_VOL",
    "HIPP_VOL_ICV",
    "WMH_TOTAL",
    "FS_TOTAL_GM",
    "FS_L_ENTORHINAL_THICKNESS",
    "FS_R_ENTORHINAL_THICKNESS",
    "FOXLAB_BRAINVOL",
    "FOXLAB_VENTVOL",
    "FDG_METAROI_MEAN",
]

BIOMARKER_FEATURES = [
    "AMY_CENTILOIDS",
    "AMY_SUMMARY_SUVR",
    "TAU_META_TEMPORAL_SUVR",
    "PLASMA_PTAU217_F",
    "PLASMA_AB42_F",
    "PLASMA_AB40_F",
    "CSF_ABETA42",
    "CSF_TTAU",
    "CSF_PTAU181",
]

DOCTOR_ENTERED_FEATURES = COGNITIVE_FEATURES + list(SYMPTOM_FEATURES.keys())

MAIN_SHOWN_FEATURES = set(
    SUMMARY_FEATURES
    + COGNITIVE_FEATURES
    + list(SYMPTOM_FEATURES.keys())
    + IMAGING_FEATURES
    + BIOMARKER_FEATURES
)

# -----------------------------
# Helper functions
# -----------------------------
def is_missing_value(value):
    if value is None:
        return True
    try:
        return pd.isna(value)
    except Exception:
        return False

def get_source_value(source_data, column):
    if source_data is not None and len(source_data) > 0 and column in source_data.columns:
        value = source_data.iloc[0][column]
        if is_missing_value(value):
            return np.nan
        return value
    return np.nan

def optional_number(label, default_value=np.nan):
    if is_missing_value(default_value):
        text_default = ""
    else:
        text_default = str(default_value)

    value = st.text_input(label, value=text_default)

    if value.strip() == "":
        return np.nan

    try:
        return float(value)
    except ValueError:
        st.error(f"Please enter a valid number for {label}")
        return np.nan

def optional_advanced_value(column, default_value=np.nan):
    if is_missing_value(default_value):
        text_default = ""
    else:
        text_default = str(default_value)

    value = st.text_input(column, value=text_default, key=f"advanced_{column}")

    if value.strip() == "":
        return np.nan

    if column in MODEL_CATEGORICAL_COLS:
        return value

    try:
        return float(value)
    except ValueError:
        st.error(f"{column} must be a number.")
        return np.nan

def clean_selectbox_value(value):
    if value in ["Missing", "", None]:
        return np.nan
    return value

def default_index(options, value):
    if is_missing_value(value):
        return 0
    if value in options:
        return options.index(value)
    return 0

def priority_from_probs(cn_prob, mci_prob, ad_prob):
    if ad_prob >= 0.50:
        return "RED", "High review priority"
    elif mci_prob >= 0.40 or ad_prob >= 0.20:
        return "YELLOW", "Closer follow-up recommended"
    else:
        return "GREEN", "Routine review"

def priority_color(priority):
    if priority == "RED":
        return "#ff4b4b"
    if priority == "YELLOW":
        return "#f4c542"
    return "#2ecc71"

def count_source_loaded_features(source_data):
    count = 0
    for col in feature_columns:
        if col not in DOCTOR_ENTERED_FEATURES and not is_missing_value(get_source_value(source_data, col)):
            count += 1
    return count

# -----------------------------
# Sidebar workflow and patient source
# -----------------------------
st.sidebar.header("Workflow")
st.sidebar.write("1. Doctor logs in")
st.sidebar.write("2. Patient record auto-loads demographics/labs/imaging")
st.sidebar.write("3. Doctor fills cognitive tests + symptoms")
st.sidebar.write("4. AI generates Green / Yellow / Red review priority")

st.sidebar.info(
    "Missing values are allowed. Do not enter 0 unless the real value is actually 0."
)

st.sidebar.header("Patient Source")

source_data = None

if len(demo_database) > 0:
    display_names = demo_database["PATIENT_NAME"].tolist()
    selected_patient = st.sidebar.selectbox("Select patient from database", display_names)
    source_data = demo_database[demo_database["PATIENT_NAME"] == selected_patient].head(1)
    st.sidebar.success("Patient loaded from demo database.")
else:
    st.sidebar.error("patient_database.csv was not found.")

uploaded_file = st.sidebar.file_uploader(
    "Optional: upload one-patient CSV instead",
    type=["csv"]
)

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)
    if len(uploaded_data) > 1:
        st.sidebar.warning("CSV has more than one row. Only the first row will be used.")
    source_data = uploaded_data.head(1)
    st.sidebar.success("Uploaded CSV is being used instead of demo database.")

if source_data is not None and len(source_data) > 0:
    with st.sidebar.expander("Preview loaded patient data"):
        st.dataframe(source_data)

# -----------------------------
# Create patient row
# -----------------------------
patient_row = {col: np.nan for col in feature_columns}

if source_data is not None:
    for col in feature_columns:
        if col in source_data.columns:
            patient_row[col] = source_data.iloc[0][col]

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Patient Summary",
    "Cognitive Tests",
    "Symptoms",
    "Auto-filled Records",
    "Prediction"
])

with tab1:
    st.header("Patient Summary")

    patient_name = get_source_value(source_data, "PATIENT_NAME")
    patient_id = get_source_value(source_data, "RID")

    st.write("**Selected patient:**", patient_name if not is_missing_value(patient_name) else "Demo patient")
    st.write("**Patient ID:**", patient_id if not is_missing_value(patient_id) else "N/A")

    st.caption(
        "These fields come from the patient database/EHR. "
        "The doctor can edit them if needed."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        patient_row["Age"] = optional_number("Age", get_source_value(source_data, "Age"))

        gender_options = ["Missing", "Female", "Male"]
        gender_default = get_source_value(source_data, "Gender")
        patient_row["Gender"] = clean_selectbox_value(
            st.selectbox("Gender", gender_options, index=default_index(gender_options, gender_default))
        )

    with col2:
        patient_row["EDUCATION_YEARS"] = optional_number(
            "Education years",
            get_source_value(source_data, "EDUCATION_YEARS")
        )

        marriage_options = ["Missing", "Married", "Widowed", "Divorced", "Never married", "Unknown"]
        marriage_default = get_source_value(source_data, "MARRIAGE_STATUS")
        patient_row["MARRIAGE_STATUS"] = clean_selectbox_value(
            st.selectbox("Marriage status", marriage_options, index=default_index(marriage_options, marriage_default))
        )

    with col3:
        apoe_options = ["Missing", "2/2", "2/3", "2/4", "3/3", "3/4", "4/4"]
        apoe_default = get_source_value(source_data, "APOE_GENOTYPE")
        patient_row["APOE_GENOTYPE"] = clean_selectbox_value(
            st.selectbox("APOE genotype", apoe_options, index=default_index(apoe_options, apoe_default))
        )

        apoe4_options = ["Missing", "No E4", "One E4 copy", "Two E4 copies"]
        apoe4_default = get_source_value(source_data, "APOE4_STATUS")
        patient_row["APOE4_STATUS"] = clean_selectbox_value(
            st.selectbox("APOE4 status", apoe4_options, index=default_index(apoe4_options, apoe4_default))
        )

        patient_row["APOE4_COUNT"] = optional_number(
            "APOE4 count",
            get_source_value(source_data, "APOE4_COUNT")
        )

with tab2:
    st.header("Doctor-entered Cognitive Screening")
    st.caption("These are intentionally left blank unless already available. The doctor fills them during review.")

    cols = st.columns(3)
    for i, col in enumerate(COGNITIVE_FEATURES):
        with cols[i % 3]:
            patient_row[col] = optional_number(col, get_source_value(source_data, col))

with tab3:
    st.header("Doctor-entered Symptoms and Functional Concerns")
    st.caption("These are intentionally left blank unless already available. The doctor fills them during review.")

    symptom_options = ["Missing", "Absent", "Present"]
    cols = st.columns(3)

    for i, (col, label) in enumerate(SYMPTOM_FEATURES.items()):
        default = get_source_value(source_data, col)
        with cols[i % 3]:
            patient_row[col] = clean_selectbox_value(
                st.selectbox(label, symptom_options, index=default_index(symptom_options, default))
            )

    clinical_notes_default = get_source_value(source_data, "CLINICAL_NOTES")
    clinical_notes = st.text_area(
        "Doctor notes",
        value="" if is_missing_value(clinical_notes_default) else str(clinical_notes_default),
        placeholder="Optional clinical observations, functional concerns, medication changes..."
    )

with tab4:
    st.header("Auto-filled Records")
    st.caption(
        "These values normally come from labs, radiology, EHR, or image processing. "
        "They are separate from the doctor-entered cognitive/symptom fields."
    )

    st.subheader("MRI / Imaging")
    cols = st.columns(3)
    for i, col in enumerate(IMAGING_FEATURES):
        with cols[i % 3]:
            patient_row[col] = optional_number(col, get_source_value(source_data, col))

    st.subheader("PET / Plasma / CSF Biomarkers")
    cols = st.columns(3)
    for i, col in enumerate(BIOMARKER_FEATURES):
        with cols[i % 3]:
            patient_row[col] = optional_number(col, get_source_value(source_data, col))

    advanced_features = [col for col in feature_columns if col not in MAIN_SHOWN_FEATURES]

    with st.expander("Advanced Optional Model Fields"):
        st.caption(
            "These are the remaining model features. They can stay blank. "
            "Fill them only if the hospital/lab/radiology system provides them."
        )

        cols = st.columns(3)
        for i, col in enumerate(advanced_features):
            with cols[i % 3]:
                patient_row[col] = optional_advanced_value(col, get_source_value(source_data, col))

with tab5:
    st.header("AI Review Priority")

    patient_df = pd.DataFrame([patient_row])[feature_columns]

    total_count = len(feature_columns)
    filled_count = patient_df.notna().sum(axis=1).iloc[0]

    doctor_valid_cols = [col for col in DOCTOR_ENTERED_FEATURES if col in feature_columns]
    doctor_total = len(doctor_valid_cols)
    doctor_filled = patient_df[doctor_valid_cols].notna().sum(axis=1).iloc[0]

    auto_total = total_count - doctor_total
    auto_loaded = count_source_loaded_features(source_data)

    overall_completeness = filled_count / total_count
    doctor_completeness = doctor_filled / doctor_total if doctor_total else 0
    auto_completeness = auto_loaded / auto_total if auto_total else 0

    st.subheader("Data completeness breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Auto-filled record data", f"{auto_completeness:.1%}")
        st.caption(f"{auto_loaded} of {auto_total} non-doctor model fields loaded.")

    with col2:
        st.metric("Doctor-entered clinical data", f"{doctor_completeness:.1%}")
        st.caption(f"{doctor_filled} of {doctor_total} cognitive/symptom fields filled.")

    with col3:
        st.metric("Overall model data available", f"{overall_completeness:.1%}")
        st.caption(f"{filled_count} of {total_count} total model features available.")

    st.info(
        "The app may start with auto-filled data already available from the patient database. "
        "The doctor-entered section can still be empty until the doctor fills cognitive tests and symptoms."
    )

    missing_doctor_cols = [col for col in doctor_valid_cols if patient_df[col].isna().iloc[0]]
    missing_all_cols = patient_df.columns[patient_df.isna().iloc[0]].tolist()

    with st.expander("Missing doctor-entered fields"):
        st.write(missing_doctor_cols)

    with st.expander("Missing all model fields"):
        st.write(missing_all_cols[:60])
        if len(missing_all_cols) > 60:
            st.write(f"... and {len(missing_all_cols) - 60} more missing fields")

    if st.button("Generate Review Priority", type="primary"):
        prediction = model.predict(patient_df)[0]
        probabilities = model.predict_proba(patient_df)[0]
        classes = model.classes_

        prob_dict = dict(zip(classes, probabilities))
        cn_prob = prob_dict.get("CN", 0)
        mci_prob = prob_dict.get("MCI", 0)
        ad_prob = prob_dict.get("AD", 0)

        priority, priority_text = priority_from_probs(cn_prob, mci_prob, ad_prob)
        color = priority_color(priority)

        st.markdown(
            f"""
            <div style="
                padding: 25px;
                border-radius: 14px;
                background-color: {color};
                color: black;
                text-align: center;
                font-size: 30px;
                font-weight: bold;">
                {priority} — {priority_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Model probability support")

        prob_chart = pd.DataFrame({
            "Category": ["CN support", "MCI support", "AD support"],
            "Probability": [cn_prob, mci_prob, ad_prob]
        })

        st.bar_chart(prob_chart.set_index("Category"))

        st.write("CN:", round(cn_prob * 100, 2), "%")
        st.write("MCI:", round(mci_prob * 100, 2), "%")
        st.write("AD:", round(ad_prob * 100, 2), "%")

        st.subheader("Explanation summary")

        reasons = []

        if doctor_filled == 0:
            reasons.append("No cognitive test or symptom fields were entered by the doctor yet.")
        else:
            reasons.append("Doctor-entered cognitive/symptom fields were included in the review.")

        if auto_loaded > 0:
            reasons.append("Auto-filled patient record, lab, or imaging data was included.")

        if overall_completeness < 0.30:
            reasons.append("Many model fields are missing, so confidence may be limited.")
        else:
            reasons.append("The model used the available fields and handled missing values automatically.")

        for reason in reasons:
            st.write("- " + reason)

        st.info(
            "Suggested use: review the patient clinically, repeat cognitive screening if needed, "
            "and consider referral or additional tests only if clinically appropriate."
        )

        st.warning("This output is a review-priority support signal, not a diagnosis.")
