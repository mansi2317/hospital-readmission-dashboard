import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Hospital Readmission Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 1350px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        padding: 24px 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #dbeafe;
        margin: 0;
    }

    .card {
        background: white;
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.07);
        border: 1px solid #e5e7eb;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.4rem;
    }

    .section-sub {
        color: #6b7280;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 14px;
        border-radius: 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 800;
    }

    .stButton > button {
        background-color: #0f172a;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
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
# STATIC DATA
# ==========================================
model_results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Naive Bayes"],
    "AUC Score": [0.60, 0.59, 0.62, 0.58]
})

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
# NAVIGATION
# ==========================================
section = st.radio(
    "Navigation",
    ["Overview", "EDA", "Model Performance", "Prediction Tool", "Feature Importance", "Methodology"],
    horizontal=True
)

# ==========================================
# HERO
# ==========================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🏥 Hospital Readmission Dashboard</div>
    <p class="hero-subtitle">
        Interactive capstone dashboard for analyzing and predicting 30-day hospital readmissions among diabetic patients.
    </p>
</div>
""", unsafe_allow_html=True)

total_patients = len(df)
readmitted_count = int(df["readmitted"].sum())
not_readmitted_count = total_patients - readmitted_count
readmit_rate = round((readmitted_count / total_patients) * 100, 2)

# ==========================================
# OVERVIEW
# ==========================================
if section == "Overview":
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">High-level summary of dataset composition and readmission patterns.</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Patients", f"{total_patients:,}")
    k2.metric("Readmitted (<30 Days)", f"{readmitted_count:,}")
    k3.metric("Not Readmitted", f"{not_readmitted_count:,}")
    k4.metric("Readmission Rate", f"{readmit_rate}%")

    st.markdown("")

    left, right = st.columns([1.5, 1])

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
            color_discrete_map={
                "Not Readmitted": "#2563eb",
                "Readmitted": "#ef4444"
            }
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            title="Readmission Distribution",
            height=460,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            font=dict(color="#111827", size=14),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="card"><b style="color:#1d4ed8;">Business Objective</b><br><br>Identify high-risk diabetic patients likely to be readmitted within 30 days so hospitals can intervene earlier.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card"><b style="color:#1d4ed8;">Operational Value</b><br><br>Supports discharge planning, follow-up care, and more efficient allocation of healthcare resources.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card"><b style="color:#1d4ed8;">Analytical Note</b><br><br>The dataset is imbalanced, so AUC, Recall, and F1-score are more informative than accuracy alone.</div>', unsafe_allow_html=True)

# ==========================================
# EDA
# ==========================================
elif section == "EDA":
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Distribution analysis of key variables associated with readmission risk.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.histogram(
            sample_df,
            x="time_in_hospital",
            nbins=20,
            title="Distribution of Time in Hospital",
            color_discrete_sequence=["#2563eb"]
        )
        fig1.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#111827"))
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.histogram(
            sample_df,
            x="num_medications",
            nbins=20,
            title="Distribution of Number of Medications",
            color_discrete_sequence=["#0f766e"]
        )
        fig2.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#111827"))
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig3 = px.box(
            sample_df,
            x="readmitted",
            y="time_in_hospital",
            color="readmitted",
            title="Time in Hospital by Readmission Status",
            color_discrete_sequence=["#94a3b8", "#2563eb"]
        )
        fig3.update_xaxes(tickvals=[0, 1], ticktext=["Not Readmitted", "Readmitted"])
        fig3.update_layout(height=400, showlegend=False, plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#111827"))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = px.box(
            sample_df,
            x="readmitted",
            y="num_medications",
            color="readmitted",
            title="Number of Medications by Readmission Status",
            color_discrete_sequence=["#94a3b8", "#2563eb"]
        )
        fig4.update_xaxes(tickvals=[0, 1], ticktext=["Not Readmitted", "Readmitted"])
        fig4.update_layout(height=400, showlegend=False, plot_bgcolor="white", paper_bgcolor="white", font=dict(color="#111827"))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Descriptive Statistics")
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
    st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

# ==========================================
# MODEL PERFORMANCE
# ==========================================
elif section == "Model Performance":
    st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Comparison of model performance using AUC.</div>', unsafe_allow_html=True)

    a, b = st.columns([1, 1.4])

    with a:
        st.dataframe(model_results_df, use_container_width=True)

    with b:
        fig = px.bar(
            model_results_df,
            x="Model",
            y="AUC Score",
            text="AUC Score",
            color="Model",
            color_discrete_sequence=["#94a3b8", "#2563eb", "#16a34a", "#cbd5e1"]
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            title="AUC Comparison Across Models",
            height=430,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            font=dict(color="#111827")
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="card">Among the evaluated models, <b>XGBoost</b> achieved the highest AUC and showed the strongest predictive performance.</div>', unsafe_allow_html=True)

# ==========================================
# PREDICTION TOOL
# ==========================================
elif section == "Prediction Tool":
    st.markdown('<div class="section-title">Readmission Risk Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Interactive estimation tool based on patient utilization and diagnosis indicators.</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)

    with p1:
        time_in_hospital = st.slider("Time in Hospital", 1, 14, 5)
        num_medications = st.slider("Number of Medications", 1, 50, 10)
        number_inpatient = st.slider("Previous Inpatient Visits", 0, 10, 0)

    with p2:
        number_emergency = st.slider("Emergency Visits", 0, 10, 0)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)

    if st.button("Estimate Readmission Risk", use_container_width=True):
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
            color = "#ef4444"
        elif prob >= 0.40:
            risk = "Moderate Risk"
            color = "#f59e0b"
        else:
            risk = "Low Risk"
            color = "#16a34a"

        r1, r2 = st.columns([0.9, 1.1])

        with r1:
            st.metric("Risk Probability", f"{prob:.3f}")
            st.markdown(f"### Risk Category: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
            st.caption("Simplified dashboard estimator for demonstration purposes.")

        with r2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Estimated Risk Score", "font": {"color": "#111827", "size": 22}},
                number={"font": {"color": "#111827", "size": 56}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"color": "#111827"}},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#dcfce7"},
                        {"range": [40, 60], "color": "#fef3c7"},
                        {"range": [60, 100], "color": "#fee2e2"}
                    ]
                }
            ))
            gauge.update_layout(height=320, paper_bgcolor="white", font=dict(color="#111827"))
            st.plotly_chart(gauge, use_container_width=True)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
elif section == "Feature Importance":
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Top variables contributing to readmission risk.</div>', unsafe_allow_html=True)

    fig = px.bar(
        feature_importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Top Predictive Features"
    )
    fig.update_layout(height=430, plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False, font=dict(color="#111827"))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# METHODOLOGY
# ==========================================
elif section == "Methodology":
    st.markdown('<div class="section-title">Project Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Summary of the analytics workflow used in the capstone project.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <b>Workflow Used</b><br><br>
        1. Data cleaning and preprocessing<br>
        2. Missing value treatment and outlier handling<br>
        3. Exploratory data analysis<br>
        4. Train/test split for validation<br>
        5. Predictive modeling using Logistic Regression, Random Forest, XGBoost, and Naive Bayes<br>
        6. Model evaluation using Accuracy, Precision, Recall, F1-score, and AUC<br>
        7. Interpretation of business implications
    </div>
    """, unsafe_allow_html=True)
