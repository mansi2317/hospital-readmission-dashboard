import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Hospital Readmission Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# CUSTOM STYLES
# ==========================================
st.markdown("""
<style>
    .main {
        background-color: #f4f7fb;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .hero {
        background: linear-gradient(90deg, #0f172a, #1d4ed8);
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
    }
    .hero p {
        margin-top: 0.5rem;
        margin-bottom: 0;
        font-size: 1rem;
        color: #dbeafe;
    }
    .insight-box {
        background: white;
        border-left: 6px solid #2563eb;
        padding: 1rem 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }
    .subtle {
        color: #475569;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("diabetic_data.csv")
    df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
    return df

df = load_data()
sample_df = df.sample(n=min(8000, len(df)), random_state=42)

# ==========================================
# FEATURE IMPORTANCE TABLE
# ==========================================
feature_importance_df = pd.DataFrame({
    "Feature": [
        "number_inpatient",
        "num_medications",
        "time_in_hospital",
        "number_emergency",
        "number_diagnoses"
    ],
    "Importance": [0.24, 0.20, 0.18, 0.16, 0.12]
})

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Executive Overview",
        "Exploratory Analysis",
        "Model Performance",
        "Prediction Tool",
        "Feature Importance",
        "Methodology"
    ]
)

if "age" in sample_df.columns:
    age_values = sorted(sample_df["age"].astype(str).unique().tolist())
    selected_age = st.sidebar.multiselect(
        "Filter by Age Group",
        age_values,
        default=age_values
    )
    sample_df = sample_df[sample_df["age"].astype(str).isin(selected_age)]

# ==========================================
# HERO
# ==========================================
st.markdown("""
<div class="hero">
    <h1>🏥 Hospital Readmission Analytics Dashboard</h1>
    <p>Capstone project dashboard for analyzing and predicting 30-day hospital readmissions among diabetic patients.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# EXECUTIVE OVERVIEW
# ==========================================
if section == "Executive Overview":
    total_patients = len(df)
    readmitted_count = int(df["readmitted"].sum())
    non_readmitted_count = total_patients - readmitted_count
    readmit_rate = round((readmitted_count / total_patients) * 100, 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", f"{total_patients:,}")
    c2.metric("Readmitted (<30 Days)", f"{readmitted_count:,}")
    c3.metric("Not Readmitted", f"{non_readmitted_count:,}")
    c4.metric("Readmission Rate", f"{readmit_rate}%")

    st.markdown("---")

    left, right = st.columns([1.35, 1])

    with left:
        readmit_df = (
            sample_df["readmitted"]
            .value_counts()
            .sort_index()
            .rename(index={0: "Not Readmitted", 1: "Readmitted"})
            .reset_index()
        )
        readmit_df.columns = ["Status", "Count"]

        fig = px.bar(
            readmit_df,
            x="Status",
            y="Count",
            text="Count",
            color="Status",
            color_discrete_sequence=["#2563eb", "#dc2626"],
            title="Readmission Distribution"
        )
        fig.update_layout(
            height=430,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="insight-box"><b>Business Objective</b><br>Identify high-risk diabetic patients likely to be readmitted within 30 days so hospitals can intervene early.</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box"><b>Operational Value</b><br>Supports better discharge planning, follow-up care, and resource allocation for high-risk patients.</div>', unsafe_allow_html=True)
        st.markdown('<div class="insight-box"><b>Key Observation</b><br>The dataset is strongly imbalanced, so AUC, Recall, and F1-score are more informative than accuracy alone.</div>', unsafe_allow_html=True)

# ==========================================
# EXPLORATORY ANALYSIS
# ==========================================
elif section == "Exploratory Analysis":
    st.subheader("Exploratory Data Analysis")
    st.markdown('<div class="subtle">These charts summarize the distribution of key healthcare utilization variables and their relationship with readmission.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(
            sample_df,
            x="time_in_hospital",
            nbins=20,
            title="Distribution of Time in Hospital",
            color_discrete_sequence=["#2563eb"]
        )
        fig1.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            sample_df,
            x="num_medications",
            nbins=20,
            title="Distribution of Number of Medications",
            color_discrete_sequence=["#0f766e"]
        )
        fig2.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.box(
            sample_df,
            x="readmitted",
            y="time_in_hospital",
            title="Time in Hospital vs Readmission",
            color="readmitted",
            color_discrete_sequence=["#2563eb", "#dc2626"]
        )
        fig3.update_xaxes(tickvals=[0, 1], ticktext=["Not Readmitted", "Readmitted"])
        fig3.update_layout(height=400, showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.box(
            sample_df,
            x="readmitted",
            y="num_medications",
            title="Number of Medications vs Readmission",
            color="readmitted",
            color_discrete_sequence=["#2563eb", "#dc2626"]
        )
        fig4.update_xaxes(tickvals=[0, 1], ticktext=["Not Readmitted", "Readmitted"])
        fig4.update_layout(height=400, showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Descriptive Statistics")
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
    summary_table = df[numeric_cols].describe().round(2)
    st.dataframe(summary_table, use_container_width=True)

    csv = summary_table.to_csv().encode("utf-8")
    st.download_button(
        "Download Summary Statistics",
        data=csv,
        file_name="summary_statistics.csv",
        mime="text/csv"
    )

# ==========================================
# MODEL PERFORMANCE
# ==========================================
elif section == "Model Performance":
    st.subheader("Model Performance Comparison")

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Naive Bayes"],
        "AUC Score": [0.60, 0.59, 0.62, 0.58],
        "Performance Level": ["Baseline", "Moderate", "Best", "Baseline"]
    })

    c1, c2 = st.columns([1, 1.4])

    with c1:
        st.dataframe(results, use_container_width=True)

    with c2:
        fig = px.bar(
            results,
            x="Model",
            y="AUC Score",
            color="Performance Level",
            text="AUC Score",
            color_discrete_map={
                "Baseline": "#94a3b8",
                "Moderate": "#2563eb",
                "Best": "#16a34a"
            },
            title="AUC Comparison Across Models"
        )
        fig.update_layout(height=430, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="insight-box"><b>Interpretation:</b> XGBoost achieved the highest AUC and therefore showed the strongest ability to distinguish between readmitted and non-readmitted patients.</div>', unsafe_allow_html=True)

# ==========================================
# PREDICTION TOOL
# ==========================================
elif section == "Prediction Tool":
    st.subheader("Readmission Risk Estimator")
    st.markdown('<div class="subtle">Use the controls below to simulate patient conditions and estimate readmission risk.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        time_in_hospital = st.slider("Time in Hospital", 1, 14, 5)
        num_medications = st.slider("Number of Medications", 1, 50, 10)
        number_inpatient = st.slider("Previous Inpatient Visits", 0, 10, 0)

    with c2:
        number_emergency = st.slider("Emergency Visits", 0, 10, 0)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)

    if st.button("Estimate Readmission Risk"):
        score = (
            0.10 * time_in_hospital
            + 0.10 * num_medications
            + 0.30 * number_inpatient
            + 0.20 * number_emergency
            + 0.10 * number_diagnoses
        )

        prob = 1 / (1 + np.exp(-score / 10))

        if prob >= 0.60:
            risk = "High Risk"
            color = "#dc2626"
        elif prob >= 0.40:
            risk = "Moderate Risk"
            color = "#f59e0b"
        else:
            risk = "Low Risk"
            color = "#16a34a"

        a, b = st.columns([1, 1.2])
        with a:
            st.metric("Estimated Risk Probability", f"{prob:.3f}")
            st.markdown(f"### Risk Category: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)

        with b:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#dcfce7"},
                        {"range": [40, 60], "color": "#fef3c7"},
                        {"range": [60, 100], "color": "#fee2e2"}
                    ]
                }
            ))
            gauge.update_layout(height=320)
            st.plotly_chart(gauge, use_container_width=True)

        st.caption("This is a simplified dashboard-based estimator for demonstration. Final model evaluation was conducted separately in the capstone workflow.")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
elif section == "Feature Importance":
    st.subheader("Top Predictive Features")

    fig = px.bar(
        feature_importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Feature Importance Ranking"
    )
    fig.update_layout(height=450, plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box"><b>Interpretation:</b> Prior inpatient visits, medication burden, and time in hospital appear to be strong indicators of readmission risk, reflecting both clinical complexity and prior healthcare utilization.</div>',
        unsafe_allow_html=True
    )

# ==========================================
# METHODOLOGY
# ==========================================
elif section == "Methodology":
    st.subheader("Project Methodology")
    st.write("""
    **Workflow used in the capstone project:**
    1. Data cleaning and preprocessing  
    2. Missing value treatment and outlier handling  
    3. Exploratory data analysis  
    4. Train/test split for validation  
    5. Predictive modeling using Logistic Regression, Random Forest, XGBoost, and Naive Bayes  
    6. Model comparison using Accuracy, Precision, Recall, F1-score, and AUC  
    7. Interpretation of results and business implications  
    """)

    st.markdown(
        '<div class="insight-box"><b>Validation Approach:</b> Models were validated using an 80/20 train-test split. Because the dataset is imbalanced, evaluation emphasized AUC, Recall, and F1-score rather than relying only on accuracy.</div>',
        unsafe_allow_html=True
    )
