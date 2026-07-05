import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Synapse Flow AI", page_icon="🧠", layout="wide")
st.title("🧠 Synapse Flow AI")
st.subheader("Cognitive Risk Review-Priority Assistant")
st.warning("Clinical decision-support concept only. This tool does NOT diagnose dementia and does NOT replace clinical judgment.")

@st.cache_resource
def load_model():
    model = joblib.load("synapse_flow_model.pkl")
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    return model, feature_columns

model, feature_columns = load_model()

def missing(v):
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:
        return False

def get_val(uploaded_data, col):
    if uploaded_data is not None and col in uploaded_data.columns:
        v = uploaded_data.iloc[0][col]
        return np.nan if missing(v) else v
    return np.nan

def optional_number(label, default_value=np.nan):
    text_default = "" if missing(default_value) else str(default_value)
    value = st.text_input(label, value=text_default)
    if value.strip() == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        st.error(f"Please enter a valid number for {label}")
        return np.nan

def clean_choice(value):
    return np.nan if value in ["Missing", "", None] else value

def idx(options, value):
    if missing(value):
        return 0
    return options.index(value) if value in options else 0

def priority_from_probs(cn, mci, ad):
    if ad >= 0.50:
        return "RED", "High review priority"
    elif mci >= 0.40 or ad >= 0.20:
        return "YELLOW", "Closer follow-up recommended"
    return "GREEN", "Routine review"

def priority_color(priority):
    return {"RED": "#ff4b4b", "YELLOW": "#f4c542", "GREEN": "#2ecc71"}.get(priority, "#ddd")

st.sidebar.header("Workflow")
st.sidebar.write("1. Upload patient CSV if available")
st.sidebar.write("2. Doctor adds cognitive tests")
st.sidebar.write("3. Doctor adds symptoms")
st.sidebar.write("4. Generate Green / Yellow / Red priority")
st.sidebar.info("Missing values are allowed. Do not enter 0 unless the real value is actually 0.")

st.sidebar.header("Upload Patient CSV")
uploaded_file = st.sidebar.file_uploader("Upload one-patient CSV row", type=["csv"])
uploaded_data = None

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)
    if len(uploaded_data) > 1:
        st.sidebar.warning("CSV has more than one row. Only the first row will be used.")
    st.sidebar.success("Patient CSV uploaded.")
    with st.sidebar.expander("Preview uploaded CSV"):
        st.dataframe(uploaded_data.head())

patient_row = {col: np.nan for col in feature_columns}
if uploaded_data is not None:
    for col in feature_columns:
        if col in uploaded_data.columns:
            patient_row[col] = uploaded_data.iloc[0][col]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Patient Summary", "Cognitive Tests", "Symptoms", "Auto-filled Records", "Prediction"])

with tab1:
    st.header("Patient Summary")
    st.caption("These fields can come from the patient record. The doctor can edit them if needed.")
    c1, c2, c3 = st.columns(3)
    with c1:
        rid = get_val(uploaded_data, "RID")
        st.text_input("Patient ID / RID", value=str(rid) if not missing(rid) else "Demo patient")
        patient_row["Age"] = optional_number("Age", get_val(uploaded_data, "Age"))
        gender_options = ["Missing", "Female", "Male"]
        patient_row["Gender"] = clean_choice(st.selectbox("Gender", gender_options, index=idx(gender_options, get_val(uploaded_data, "Gender"))))
    with c2:
        patient_row["EDUCATION_YEARS"] = optional_number("Education years", get_val(uploaded_data, "EDUCATION_YEARS"))
        marriage_options = ["Missing", "Married", "Widowed", "Divorced", "Never married", "Unknown"]
        patient_row["MARRIAGE_STATUS"] = clean_choice(st.selectbox("Marriage status", marriage_options, index=idx(marriage_options, get_val(uploaded_data, "MARRIAGE_STATUS"))))
    with c3:
        apoe_options = ["Missing", "2/2", "2/3", "2/4", "3/3", "3/4", "4/4"]
        patient_row["APOE_GENOTYPE"] = clean_choice(st.selectbox("APOE genotype", apoe_options, index=idx(apoe_options, get_val(uploaded_data, "APOE_GENOTYPE"))))
        apoe4_options = ["Missing", "No E4", "One E4 copy", "Two E4 copies"]
        patient_row["APOE4_STATUS"] = clean_choice(st.selectbox("APOE4 status", apoe4_options, index=idx(apoe4_options, get_val(uploaded_data, "APOE4_STATUS"))))
        patient_row["APOE4_COUNT"] = optional_number("APOE4 count", get_val(uploaded_data, "APOE4_COUNT"))

with tab2:
    st.header("Doctor-entered Cognitive Screening")
    st.caption("Enter scores if performed today. Leave blank if not performed.")
    cognitive_cols = ["MMSE_TOTAL", "MOCA_TOTAL", "CDR_GLOBAL", "CDR_SUM_OF_BOXES", "FAQ_TOTAL", "ADAS_TOTSCORE", "ADAS_TOTAL13", "NPIQ_TOTAL", "LOGICAL_MEMORY_TOTAL", "AVLT_TOTAL", "AVLT_DELAY_30MIN", "AVLT_DELAY_TOTAL", "TRAILS_A_SCORE", "BNT_TOTAL", "CLOCK_SCORE", "COPY_SCORE"]
    cols = st.columns(3)
    for i, col in enumerate(cognitive_cols):
        with cols[i % 3]:
            patient_row[col] = optional_number(col, get_val(uploaded_data, col))

with tab3:
    st.header("Symptoms and Functional Concerns")
    st.caption("These are practical doctor-entered observations.")
    symptom_cols = {"Dizziness": "Dizziness", "Low_Energy": "Low energy", "Headache": "Headache", "Insomia": "Insomnia", "Dep_Mood": "Depressed mood", "Wandering": "Wandering", "Fall": "Falls"}
    symptom_options = ["Missing", "Absent", "Present"]
    cols = st.columns(3)
    for i, (col, label) in enumerate(symptom_cols.items()):
        with cols[i % 3]:
            patient_row[col] = clean_choice(st.selectbox(label, symptom_options, index=idx(symptom_options, get_val(uploaded_data, col))))
    st.text_area("Doctor notes", placeholder="Optional clinical observations, functional concerns, medication changes...")

with tab4:
    st.header("Auto-filled Records")
    st.caption("These values normally come from labs, radiology, EHR, or image processing. For the MVP, they can be uploaded by CSV or left blank.")
    st.subheader("MRI / Imaging")
    imaging_cols = ["TOTAL_HIPP_VOL", "HIPP_VOL_ICV", "WMH_TOTAL", "FS_TOTAL_GM", "FS_L_ENTORHINAL_THICKNESS", "FS_R_ENTORHINAL_THICKNESS", "FOXLAB_BRAINVOL", "FOXLAB_VENTVOL", "FDG_METAROI_MEAN"]
    cols = st.columns(3)
    for i, col in enumerate(imaging_cols):
        with cols[i % 3]:
            patient_row[col] = optional_number(col, get_val(uploaded_data, col))
    st.subheader("PET / Plasma / CSF Biomarkers")
    biomarker_cols = ["AMY_CENTILOIDS", "AMY_SUMMARY_SUVR", "TAU_META_TEMPORAL_SUVR", "PLASMA_PTAU217_F", "PLASMA_AB42_F", "PLASMA_AB40_F", "CSF_ABETA42", "CSF_TTAU", "CSF_PTAU181"]
    cols = st.columns(3)
    for i, col in enumerate(biomarker_cols):
        with cols[i % 3]:
            patient_row[col] = optional_number(col, get_val(uploaded_data, col))

with tab5:
    st.header("AI Review Priority")
    patient_df = pd.DataFrame([patient_row])[feature_columns]
    filled_count = patient_df.notna().sum(axis=1).iloc[0]
    total_count = len(feature_columns)
    completeness = filled_count / total_count
    st.metric("Data completeness", f"{completeness:.1%}")
    st.caption(f"{filled_count} of {total_count} model features available.")
    missing_cols = patient_df.columns[patient_df.isna().iloc[0]].tolist()
    with st.expander("Missing data preview"):
        st.write(missing_cols[:40])
        if len(missing_cols) > 40:
            st.write(f"... and {len(missing_cols) - 40} more missing fields")
    if st.button("Generate Review Priority", type="primary"):
        prediction = model.predict(patient_df)[0]
        probabilities = model.predict_proba(patient_df)[0]
        prob_dict = dict(zip(model.classes_, probabilities))
        cn_prob = prob_dict.get("CN", 0)
        mci_prob = prob_dict.get("MCI", 0)
        ad_prob = prob_dict.get("AD", 0)
        priority, priority_text = priority_from_probs(cn_prob, mci_prob, ad_prob)
        color = priority_color(priority)
        st.markdown(f"""
        <div style="padding:25px;border-radius:14px;background-color:{color};color:black;text-align:center;font-size:30px;font-weight:bold;">
        {priority} — {priority_text}
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Model probability support")
        prob_chart = pd.DataFrame({"Category": ["CN support", "MCI support", "AD support"], "Probability": [cn_prob, mci_prob, ad_prob]})
        st.bar_chart(prob_chart.set_index("Category"))
        st.write("CN:", round(cn_prob * 100, 2), "%")
        st.write("MCI:", round(mci_prob * 100, 2), "%")
        st.write("AD:", round(ad_prob * 100, 2), "%")
        st.subheader("Explanation summary")
        reasons = []
        if not missing(patient_row.get("MOCA_TOTAL")):
            reasons.append("MOCA score was included in the review.")
        if not missing(patient_row.get("MMSE_TOTAL")):
            reasons.append("MMSE score was included in the review.")
        if not missing(patient_row.get("CDR_SUM_OF_BOXES")):
            reasons.append("CDR Sum of Boxes was included in the review.")
        if not missing(patient_row.get("APOE4_STATUS")):
            reasons.append("APOE4 status was included if available.")
        if completeness < 0.30:
            reasons.append("Many advanced biomarkers or imaging values are missing, so confidence may be limited.")
        else:
            reasons.append("Multiple clinical or biomarker fields were available for review.")
        for reason in reasons:
            st.write("- " + reason)
        st.info("Suggested use: review the patient clinically, repeat cognitive screening if needed, and consider referral or additional tests only if clinically appropriate.")
        st.warning("This output is a review-priority support signal, not a diagnosis.")
