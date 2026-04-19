import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Hospital Readmission Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LIGHT, SAFE STYLING ONLY
# =========================================================
st.markdown("""
<style>
    .main .block-container {
        max-width: 1350px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        color: white;
        padding: 22px 26px;
        border-radius: 18px;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #dbeafe;
        margin: 0;
    }

    .note-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .note-title {
        font-weight: 700;
        color: #1d4ed8;
        margin-bottom: 0.4rem;
    }

    .note-text {
        color: #374151;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_data():
    data = pd.read_csv("diabetic_data.csv")
    data["readmitted"] = data["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
    return data

df = load_data()
sample_df = df.sample(n=min(8000, len(df)), random_state=42)

# =========================================================
# CONSTANT TABLES
# =========================================================
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

# =========================================================
# SHARED HELPERS
# =========================================================
def style_plot(fig, title=None, x_title=None, y_title=None, height=420):
    if title is not None:
        fig.update_layout(title=title)

    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827", size=14),
        title_font=dict(color="#111827", size=20),
        margin=dict(l=20, r=20, t=60, b=20),
        legend_font=dict(color="#111827")
    )

    fig.update_xaxes(
        title_text=x_title,
        title_font=dict(color="#111827", size=14),
        tickfont=dict(color="#111827", size=13),
        showgrid=False,
        zeroline=False
    )

    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(color="#111827", size=14),
        tickfont=dict(color="#111827", size=13),
        gridcolor="#d1d5db",
        zeroline=False
    )

    return fig

def show_plot(fig):
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False}
    )

# =========================================================
# GLOBAL METRICS
# =========================================================
total_patients = len(df)
readmitted_count = int(df["readmitted"].sum())
not_readmitted_count = total_patients - readmitted_count
readmit_rate = round((readmitted_count / total_patients) * 100, 2)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("Navigation")
section = st.sidebar.selectbox(
    "Select Section",
    [
        "Overview",
        "EDA",
        "Model Performance",
        "Prediction Tool",
        "Feature Importance"
    ]
)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-title">🏥 Hospital Readmission Dashboard</div>
    <p class="hero-subtitle">
        Capstone dashboard for analyzing and predicting 30-day hospital readmissions among diabetic patients.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# OVERVIEW
# =========================================================
if section == "Overview":
    st.subheader("Executive Overview")
    st.caption("High-level summary of dataset composition and readmission patterns.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Patients", f"{total_patients:,}")
    m2.metric("Readmitted (<30 Days)", f"{readmitted_count:,}")
    m3.metric("Not Readmitted", f"{not_readmitted_count:,}")
    m4.metric("Readmission Rate", f"{readmit_rate}%")

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
        fig.update_traces(textposition="outside", textfont=dict(color="#111827", size=14))
        fig = style_plot(
            fig,
            title="Readmission Distribution",
            x_title="Status",
            y_title="Count",
            height=450
        )
        show_plot(fig)

    with right:
        st.markdown("""
        <div class="note-box">
            <div class="note-title">Business Objective</div>
            <div class="note-text">
                Identify high-risk diabetic patients likely to be readmitted within 30 days
                so hospitals can intervene earlier.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="note-box">
            <div class="note-title">Operational Value</div>
            <div class="note-text">
                Supports discharge planning, follow-up care, and more efficient resource allocation.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="note-box">
            <div class="note-title">Analytical Note</div>
            <div class="note-text">
                Because the dataset is imbalanced, AUC, Recall, and F1-score are more informative than accuracy alone.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# EDA
# =========================================================
elif section == "EDA":
    st.subheader("Exploratory Data Analysis")
    st.caption("Distribution analysis of key variables associated with readmission risk.")

    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.histogram(
            sample_df,
            x="time_in_hospital",
            nbins=20,
            color_discrete_sequence=["#2563eb"]
        )
        fig1 = style_plot(
            fig1,
            title="Distribution of Time in Hospital",
            x_title="Time in Hospital",
            y_title="Count",
            height=400
        )
        show_plot(fig1)

    with c2:
        fig2 = px.histogram(
            sample_df,
            x="num_medications",
            nbins=20,
            color_discrete_sequence=["#0f766e"]
        )
        fig2 = style_plot(
            fig2,
            title="Distribution of Number of Medications",
            x_title="Number of Medications",
            y_title="Count",
            height=400
        )
        show_plot(fig2)

    c3, c4 = st.columns(2)

    with c3:
        fig3 = px.box(
            sample_df,
            x="readmitted",
            y="time_in_hospital",
            color="readmitted",
            color_discrete_sequence=["#94a3b8", "#2563eb"]
        )
        fig3.update_xaxes(tickvals=[0, 1], ticktext=["Not Readmitted", "Readmitted"])
        fig3 = style_plot(
            fig3,
            title="Time in Hospital by Readmission Status",
            x_title="Readmission Status",
            y_title="Time in Hospital",
            height=400
        )
        fig3.update_layout(showlegend=False)
        show_plot(fig3)

    with c4:
        fig4 = px.box(
            sample_df,
            x="readmitted",
            y="num_medications",
            color="readmitted",
            color_discrete_sequence=["#94a3b8", "#2563eb"]
        )
        fig4.update_xaxes(tickvals=[0, 1], ticktext=["Not Readmitted", "Readmitted"])
        fig4 = style_plot(
            fig4,
            title="Number of Medications by Readmission Status",
            x_title="Readmission Status",
            y_title="Number of Medications",
            height=400
        )
        fig4.update_layout(showlegend=False)
        show_plot(fig4)

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

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif section == "Model Performance":
    st.subheader("Model Performance Comparison")
    st.caption("Comparison of predictive models using AUC.")

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
        fig.update_traces(textposition="outside", textfont=dict(color="#111827", size=14))
        fig = style_plot(
            fig,
            title="AUC Comparison Across Models",
            x_title="Model",
            y_title="AUC Score",
            height=430
        )
        fig.update_layout(showlegend=False)
        show_plot(fig)

    st.info("Among the evaluated models, XGBoost achieved the highest AUC and showed the strongest predictive performance.")

# =========================================================
# PREDICTION TOOL
# =========================================================
elif section == "Prediction Tool":
    st.subheader("Readmission Risk Estimator")
    st.caption("Interactive estimation tool based on patient utilization and diagnosis indicators.")

    p1, p2 = st.columns(2)

    with p1:
        time_in_hospital = st.slider("Time in Hospital", 1, 14, 5)
        num_medications = st.slider("Number of Medications", 1, 50, 10)
        number_inpatient = st.slider("Previous Inpatient Visits", 0, 10, 0)

    with p2:
        number_emergency = st.slider("Emergency Visits", 0, 10, 0)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)

# Estimate risk
if st.button("Estimate Readmission Risk", use_container_width=True):

    score = (
        -3
        + 0.20 * time_in_hospital
        + 0.05 * num_medications
        + 0.60 * number_inpatient
        + 0.40 * number_emergency
        + 0.10 * number_diagnoses
    )

    prob = 1 / (1 + np.exp(-score))

    # Determine risk level
    if prob >= 0.60:
        risk = "High Risk"
        color = "#ef4444"

    elif prob >= 0.40:
        risk = "Moderate Risk"
        color = "#f59e0b"

    else:
        risk = "Low Risk"
        color = "#16a34a"

    with r1:
        st.metric("Risk Probability", f"{prob:.3f}")
        st.markdown(
            f"### Risk Category: <span style='color:{color}'>{risk}</span>",
            unsafe_allow_html=True
        )
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
                    "axis": {
                        "range": [0, 100],
                        "tickfont": {"color": "#111827", "size": 13}
                    },
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#dcfce7"},
                        {"range": [40, 60], "color": "#fef3c7"},
                        {"range": [60, 100], "color": "#fee2e2"}
                    ]
                }
            ))
            gauge.update_layout(
                height=320,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="#111827")
            )
            show_plot(gauge)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================
elif section == "Feature Importance":
    st.subheader("Feature Importance")
    st.caption("Top variables contributing to readmission risk.")

    fig = px.bar(
        feature_importance_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues"
    )
    fig = style_plot(
        fig,
        title="Top Predictive Features",
        x_title="Importance",
        y_title="Feature",
        height=430
    )
    fig.update_layout(coloraxis_showscale=False)
    show_plot(fig)

