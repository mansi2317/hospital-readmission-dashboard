import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hospital Readmission Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("diabetic_data.csv")
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
    return df

df = load_data()
sample_df = df.sample(n=min(5000, len(df)), random_state=42)

st.title("Hospital Readmission Prediction Dashboard")

st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Overview", "EDA", "Prediction Tool", "Model Performance"]
)

if section == "Overview":
    st.header("Dataset Overview")

    total_patients = len(df)
    readmitted_count = int(df["readmitted"].sum())
    not_readmitted_count = int(total_patients - readmitted_count)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", total_patients)
    col2.metric("Readmitted (<30 days)", readmitted_count)
    col3.metric("Not Readmitted", not_readmitted_count)

    st.subheader("Readmission Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sample_df["readmitted"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Readmission Distribution")
    ax.set_xlabel("Readmitted")
    ax.set_ylabel("Count")
    st.pyplot(fig)

elif section == "EDA":
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Time in Hospital")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(sample_df["time_in_hospital"].dropna(), bins=20)
        ax1.set_title("Time in Hospital Distribution")
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)

    with col2:
        st.subheader("Number of Medications")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(sample_df["num_medications"].dropna(), bins=20)
        ax2.set_title("Number of Medications Distribution")
        ax2.set_xlabel("Medications")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    st.subheader("Summary Statistics")
    st.dataframe(
        df[
            [
                "time_in_hospital",
                "num_lab_procedures",
                "num_medications",
                "number_outpatient",
                "number_emergency",
                "number_inpatient",
                "number_diagnoses",
            ]
        ].describe()
    )

elif section == "Prediction Tool":
    st.header("Readmission Risk Prediction Tool")
    st.write("Enter patient characteristics to estimate readmission risk.")

    time_in_hospital = st.slider("Time in Hospital", 1, 14, 5)
    num_medications = st.slider("Number of Medications", 1, 50, 10)
    number_inpatient = st.slider("Previous Inpatient Visits", 0, 10, 0)
    number_emergency = st.slider("Emergency Visits", 0, 10, 0)
    number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)

    if st.button("Predict Risk"):
        score = (
            0.10 * time_in_hospital
            + 0.10 * num_medications
            + 0.30 * number_inpatient
            + 0.20 * number_emergency
            + 0.10 * number_diagnoses
        )

        prob = 1 / (1 + pow(2.71828, -score / 10))

        if prob >= 0.60:
            risk = "High Risk of Readmission"
        elif prob >= 0.40:
            risk = "Moderate Risk of Readmission"
        else:
            risk = "Low Risk of Readmission"

        st.success(f"Risk Probability: {prob:.3f}")
        st.info(f"Prediction: {risk}")

elif section == "Model Performance":
    st.header("Model Performance Comparison")

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Naive Bayes"],
        "AUC Score": [0.60, 0.59, 0.62, 0.58]
    })

    st.dataframe(results)

    st.subheader("Best Model")
    st.write("XGBoost achieved the best AUC in the project, with an AUC of approximately 0.62.")
