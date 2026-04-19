import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Hospital Readmission Dashboard",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetic_data.csv")
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
    return df

df = load_data()

# smaller sample for faster charts
sample_df = df.sample(n=min(8000, len(df)), random_state=42)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Dashboard Controls")

section = st.sidebar.radio(
    "Navigate",
    ["Overview", "EDA", "Prediction Tool", "Model Performance"]
)

# optional filters
age_filter = None
if "age" in sample_df.columns:
    age_options = sorted(sample_df["age"].astype(str).unique().tolist())
    age_filter = st.sidebar.multiselect(
        "Filter by Age Group",
        age_options,
        default=age_options
    )
    sample_df = sample_df[sample_df["age"].astype(str).isin(age_filter)]

# -----------------------------
# TITLE
# -----------------------------
st.title("🏥 Hospital Readmission Prediction Dashboard")
st.markdown(
    """
    This dashboard presents the final capstone analysis for predicting 30-day hospital readmissions
    among diabetic patients. It summarizes the dataset, exploratory analysis, model performance,
    and a simple readmission risk prediction tool.
    """
)

# -----------------------------
# OVERVIEW
# -----------------------------
if section == "Overview":
    total_patients = len(df)
    readmitted_count = int(df["readmitted"].sum())
    not_readmitted_count = int(total_patients - readmitted_count)
    readmit_rate = round((readmitted_count / total_patients) * 100, 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", f"{total_patients:,}")
    col2.metric("Readmitted (<30 days)", f"{readmitted_count:,}")
    col3.metric("Not Readmitted", f"{not_readmitted_count:,}")
    col4.metric("Readmission Rate", f"{readmit_rate}%")

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        readmit_counts = (
            sample_df["readmitted"]
            .value_counts()
            .sort_index()
            .rename(index={0: "Not Readmitted", 1: "Readmitted"})
            .reset_index()
        )
        readmit_counts.columns = ["Status", "Count"]

        fig = px.bar(
            readmit_counts,
            x="Status",
            y="Count",
            title="Readmission Distribution",
            text="Count"
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Project Summary")
        st.info(
            """
            The goal of this project is to identify diabetic patients who are at higher risk of
            being readmitted within 30 days. The dashboard helps summarize dataset patterns,
            compare predictive models, and provide a simple risk estimation tool.
            """
        )

        st.subheader("Key Insight")
        st.write(
            "The dataset is imbalanced, with many more patients not readmitted than readmitted. "
            "This is why model evaluation focuses on metrics such as AUC, Recall, and F1-score."
        )

# -----------------------------
# EDA
# -----------------------------
elif section == "EDA":
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(
            sample_df,
            x="time_in_hospital",
            nbins=20,
            title="Distribution of Time in Hospital"
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            sample_df,
            x="num_medications",
            nbins=20,
            title="Distribution of Number of Medications"
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.box(
            sample_df,
            x="readmitted",
            y="time_in_hospital",
            title="Time in Hospital vs Readmission"
        )
        fig3.update_xaxes(
            tickvals=[0, 1],
            ticktext=["Not Readmitted", "Readmitted"]
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.box(
            sample_df,
            x="readmitted",
            y="num_medications",
            title="Number of Medications vs Readmission"
        )
        fig4.update_xaxes(
            tickvals=[0, 1],
            ticktext=["Not Readmitted", "Readmitted"]
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Summary Statistics")
    numeric_cols = [
        c for c in [
            "time_in_hospital",
            "num_lab_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses"
        ] if c in df.columns
    ]
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

# -----------------------------
# PREDICTION TOOL
# -----------------------------
elif section == "Prediction Tool":
    st.subheader("Readmission Risk Prediction Tool")
    st.write(
        "Use the sliders below to simulate patient conditions and estimate readmission risk."
    )

    col1, col2 = st.columns(2)

    with col1:
        time_in_hospital = st.slider("Time in Hospital", 1, 14, 5)
        num_medications = st.slider("Number of Medications", 1, 50, 10)
        number_inpatient = st.slider("Previous Inpatient Visits", 0, 10, 0)

    with col2:
        number_emergency = st.slider("Emergency Visits", 0, 10, 0)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)

    if st.button("Predict Readmission Risk"):
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
            color = "red"
        elif prob >= 0.40:
            risk = "Moderate Risk of Readmission"
            color = "orange"
        else:
            risk = "Low Risk of Readmission"
            color = "green"

        st.markdown("### Prediction Result")
        st.metric("Risk Probability", f"{prob:.3f}")
        st.markdown(f"**Risk Category:** :{color}[{risk}]")

        st.progress(min(int(prob * 100), 100))

        st.caption(
            "Note: This is a simplified dashboard prediction tool for demonstration. "
            "Your full capstone modeling was done separately using machine learning models."
        )

# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
elif section == "Model Performance":
    st.subheader("Model Performance Comparison")

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Naive Bayes"],
        "AUC Score": [0.60, 0.59, 0.62, 0.58]
    })

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.dataframe(results, use_container_width=True)

    with col2:
        fig = px.bar(
            results,
            x="Model",
            y="AUC Score",
            color="Model",
            title="AUC Comparison Across Models",
            text="AUC Score"
        )
        fig.update_layout(showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Interpretation")
    st.write(
        "Among the tested models, XGBoost achieved the highest AUC and therefore showed the best "
        "predictive performance for identifying patients at risk of readmission. Random Forest and "
        "Logistic Regression served as important benchmark models, while Naive Bayes provided an "
        "additional probabilistic baseline."
    )
