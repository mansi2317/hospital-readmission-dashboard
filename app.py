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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 50%, #0ea5e9 100%);
        border-radius: 20px;
        padding: 26px 30px;
        color: white;
        margin-bottom: 18px;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #e5eefc;
        margin-bottom: 0;
        line-height: 1.6;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.35rem;
    }

    .section-desc {
        color: #4b5563;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    .insight-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
        margin-bottom: 14px;
    }

    .insight-title {
        font-weight: 700;
        color: #1d4ed8;
        margin-bottom: 0.35rem;
        font-size: 1rem;
    }

    .insight-text {
        color: #374151;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 16px 14px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 800;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    /* Buttons */
    .stButton > button {
        background-color: #0f172a;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        padding: 0.6rem 1rem;
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
# STATIC TABLES
# ==========================================
model_results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost", "Naive Bayes"],
    "AUC Score": [0.60, 0.59, 0.62, 0.58],
    "Category": ["Baseline", "Moderate", "Best", "Baseline"]
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
# SIDEBAR
# ==========================================
st.sidebar.title("Dashboard Navigation")
section = st.sidebar.radio(
    "Select a View",
    [
        "Executive Overview",
        "Exploratory Analysis",
        "Model Performance",
        "Risk Prediction Tool",
        "Feature Importance",
        "Project Methodology"
    ]
)

if "age" in sample_df.columns:
    age_values = sorted(sample_df["age"].astype(str).unique().tolist())
    selected_age = st.sidebar.multiselect(
        "Filter by Age Group",
        options=age_values,
        default=age_values
    )
    sample_df = sample_df[sample_df["age"].astype(str).isin(selected_age)]

# ==========================================
# HERO
# ==========================================
st.markdown("""
<div class="hero">
    <div class="hero-title">🏥 Hospital Readmission Analytics Dashboard</div>
    <p class="hero-subtitle">
        Capstone project dashboard for analyzing and predicting 30-day hospital readmissions among diabetic patients.
    </p>
</div>
""", unsafe_allow_html=True)

total_patients = len(df)
readmitted_count = int(df["readmitted"].sum())
not_readmitted_count = total_patients - readmitted_count
readmit_rate = round((readmitted_count / total_patients) * 100, 2)

# ==========================================
# EXECUTIVE OVERVIEW
# ==========================================
if section == "Executive Overview":
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">High-level summary of the dataset and readmission patterns.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", f"{total_patients:,}")
    c2.metric("Readmitted (<30 Days)", f"{readmitted_count:,}")
    c3.metric("Not Readmitted", f"{not_readmitted_count:,}")
    c4.metric("Readmission Rate", f"{readmit_rate}%")

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
            color_discrete_map={
                "Not Readmitted": "#2563eb",
                "Readmitted": "#ef4444"
            }
        )
        fig.update_traces(
            textposition="outside",
            textfont=dict(color="#111827", size=14)
        )
        fig.update_layout(
            title="Readmission Distribution",
            title_font=dict(color="#111827", size=20),
            height=450,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827", size=14),
            xaxis=dict(title_font=dict(color="#111827"), tickfont=dict(color="#111827")),
            yaxis=dict(title_font=dict(color="#111827"), tickfont=dict(color="#111827")),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">Business Objective</div>
            <div class="insight-text">
                Identify high-risk diabetic patients likely to be readmitted within 30 days so hospitals can intervene earlier.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">Operational Value</div>
            <div class="insight-text">
                Supports discharge planning, follow-up care, and more efficient healthcare resource allocation.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">Analytical Note</div>
            <div class="insight-text">
                Because the dataset is imbalanced, AUC, Recall, and F1-score are more informative than accuracy alone.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# EDA
# ==========================================
elif section == "Exploratory Analysis":
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Visual analysis of hospital stay duration, medication usage, and readmission-related patterns.</div>', unsafe_allow_html=True)

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        fig1 = px.histogram(
            sample_df,
            x="time_in_hospital",
            nbins=20,
            title="Distribution of Time in Hospital",
            color_discrete_sequence=["#2563eb"]
        )
        fig1.update_layout(
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
            title_font=dict(color="#111827"),
            xaxis=dict(title_font=dict(color="#111827"), tickfont=dict(color="#111827")),
            yaxis=dict(title_font=dict(color="#111827"), tickfont=dict(color="#111827"))
        )
        st.plotly_chart(fig1, use_container_width=True)

    with row1_col2:
        fig2 = px.histogram(
            sample_df,
            x="num_medications",
            nbins=20,
            title="Distribution of Number of Medications",
            color_discrete_sequence=["#0f766e"]
        )
        fig2.update_layout(
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
            title_font=dict(color="#111827"),
            xaxis=dict(title_font=dict(color="#111827"), tickfont=dict(color="#111827")),
            yaxis=dict(title_font=dict(color="#111827"), tickfont=dict(color="#111827"))
        )
        st.plotly_chart(fig2, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        fig3 = px.box(
            sample_df,
            x="readmitted",
            y="time_in_hospital",
            color="readmitted",
            title="Time in Hospital by Readmission Status",
            color_discrete_sequence=["#2563eb", "#ef4444"]
        )
        fig3.update_xaxes(
            tickvals=[0, 1],
            ticktext=["Not Readmitted", "Readmitted"]
        )
        fig3.update_layout(
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            font=dict(color="#111827"),
            title_font=dict(color="#111827")
        )
        st.plotly_chart(fig3, use_container_width=True)

    with row2_col2:
        fig4 = px.box(
            sample_df,
            x="readmitted",
            y="num_medications",
            color="readmitted",
            title="Number of Medications by Readmission Status",
            color_discrete_sequence=["#2563eb", "#ef4444"]
        )
        fig4.update_xaxes(
            tickvals=[0, 1],
            ticktext=["Not Readmitted", "Readmitted"]
        )
        fig4.update_layout(
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            font=dict(color="#111827"),
            title_font=dict(color="#111827")
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Summary Statistics")
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
    summary_stats = df[numeric_cols].describe().round(2)
    st.dataframe(summary_stats, use_container_width=True)

# ==========================================
# MODEL PERFORMANCE
# ==========================================
elif section == "Model Performance":
    st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Comparison of predictive models used in the capstone project.</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.45])

    with left:
        st.dataframe(model_results_df, use_container_width=True)

    with right:
        fig = px.bar(
            model_results_df,
            x="Model",
            y="AUC Score",
            color="Category",
            text="AUC Score",
            color_discrete_map={
                "Baseline": "#94a3b8",
                "Moderate": "#3b82f6",
                "Best": "#16a34a"
            },
            title="AUC Comparison Across Models"
        )
        fig.update_traces(textposition="outside", textfont=dict(color="#111827", size=13))
        fig.update_layout(
            height=450,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827"),
            title_font=dict(color="#111827"),
            xaxis=dict(title_font=dict(color="#111827"), tickfont=dict(color="#111827")),
            yaxis=dict(title_font=dict(color="#111827"), tickfont=dict(color="#111827"))
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">Interpretation</div>
        <div class="insight-text">
            XGBoost achieved the highest AUC and therefore showed the strongest predictive ability among the tested models.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# RISK PREDICTION TOOL
# ==========================================
elif section == "Risk Prediction Tool":
    st.markdown('<div class="section-title">Readmission Risk Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Interactive estimator based on patient utilization and diagnosis indicators.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        time_in_hospital = st.slider("Time in Hospital", 1, 14, 5)
        num_medications = st.slider("Number of Medications", 1, 50, 10)
        number_inpatient = st.slider("Previous Inpatient Visits", 0, 10, 0)

    with c2:
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

        left, right = st.columns([0.9, 1.2])

        with left:
            st.metric("Risk Probability", f"{prob:.3f}")
            st.markdown(f"### Risk Category: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
            st.caption("This is a simplified estimator for dashboard demonstration purposes.")

        with right:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Estimated Risk Score", "font": {"color": "#111827", "size": 22}},
                number={"font": {"color": "#111827", "size": 64}},
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickcolor": "#111827",
                        "tickfont": {"color": "#111827"}
                    },
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#dcfce7"},
                        {"range": [40, 60], "color": "#fef3c7"},
                        {"range": [60, 100], "color": "#fee2e2"}
                    ]
                }
            ))
            gauge.update_layout(height=340, paper_bgcolor="white", font=dict(color="#111827"))
            st.plotly_chart(gauge, use_container_width=True)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================
elif section == "Feature Importance":
    st.markdown('<div class="section-title">Feature Importance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Top variables contributing to readmission risk.</div>', unsafe_allow_html=True)

    fig = px.bar(
        feature_importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        title="Top Predictive Features"
    )
    fig.update_layout(
        height=450,
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_showscale=False,
        font=dict(color="#111827"),
        title_font=dict(color="#111827")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">Feature Insight</div>
        <div class="insight-text">
            Prior inpatient visits, medication burden, hospital stay duration, and emergency utilization are strong indicators of readmission risk.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# METHODOLOGY
# ==========================================
elif section == "Project Methodology":
    st.markdown('<div class="section-title">Project Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Summary of the analytics workflow used in the capstone project.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">Workflow Used</div>
        <div class="insight-text">
            1. Data cleaning and preprocessing<br>
            2. Missing value treatment and outlier handling<br>
            3. Exploratory data analysis<br>
            4. Train/test split for validation<br>
            5. Predictive modeling using Logistic Regression, Random Forest, XGBoost, and Naive Bayes<br>
            6. Model evaluation using Accuracy, Precision, Recall, F1-score, and AUC<br>
            7. Interpretation of business implications
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
        <div class="insight-title">Validation Approach</div>
        <div class="insight-text">
            Models were validated using an 80/20 train-test split. Because the dataset is imbalanced, AUC, Recall, and F1-score were emphasized over accuracy alone.
        </div>
    </div>
    """, unsafe_allow_html=True)
