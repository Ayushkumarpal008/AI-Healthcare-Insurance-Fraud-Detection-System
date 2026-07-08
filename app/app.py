"""Streamlit dashboard for healthcare insurance fraud detection."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Streamlit is not installed. Install requirements.txt to run the dashboard."
    ) from exc

try:
    import plotly.express as px
except ImportError:  # pragma: no cover
    px = None

from src.explainability import ExplanationEngine
from src.predict import FraudPredictor
from src.train import MODEL_PATH
from src.utils import PROCESSED_DATA_DIR, REPORTS_DIR, load_json


st.set_page_config(
    page_title="Healthcare Fraud Detection AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f051d 0%, #1a0b2e 45%, #2d1559 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: #120620;
    }

    .hero-card {
        background: linear-gradient(
            135deg,
            rgba(91,33,182,0.95),
            rgba(124,58,237,0.95),
            rgba(147,51,234,0.95)
        );

        backdrop-filter: blur(18px);

        border: 1px solid rgba(255,255,255,0.10);

        padding: 35px;

        border-radius: 28px;

        margin-bottom: 30px;

        box-shadow:
            0 18px 40px rgba(124,58,237,0.35),
            inset 0 1px 0 rgba(255,255,255,0.08);

        transition: all 0.35s ease;
    }

    .hero-card:hover{
        transform: translateY(-4px);
        box-shadow:
            0 25px 55px rgba(124,58,237,0.45),
            inset 0 1px 0 rgba(255,255,255,0.10);
    }

    .section-card{
        background:rgba(255,255,255,0.06);
        border:1px solid rgba(255,255,255,0.10);
        border-radius:22px;
        padding:25px;
        margin-top:25px;
        margin-bottom:25px;
        box-shadow:0 10px 30px rgba(0,0,0,0.25);
        transition:all 0.3s ease;
    }

    .section-card:hover{
        transform:translateY(-4px);
        border:1px solid rgba(196,181,253,0.35);
        box-shadow:0 18px 42px rgba(124,58,237,0.30);
    }

    .section-title{
        font-size:30px;
        font-weight:700;
        color:white;
        margin-bottom:15px;
    }

    .section-subtitle{
        color:#d8d4fe;
        font-size:15px;
        margin-bottom:25px;
    }

    .hero-card h1 {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        color: white;
    }

    .hero-card p {
        font-size: 18px;
        color: #ede9fe;
        margin: 0;
    }

    div[data-testid="stMetric"]{

        background:linear-gradient(
            145deg,
            rgba(46,24,74,.92),
            rgba(63,34,104,.90)
        );

        border:1px solid rgba(255,255,255,.08);

        border-radius:22px;

        padding:22px;

        box-shadow:
            0 12px 35px rgba(0,0,0,.30),
            inset 0 1px 0 rgba(255,255,255,.06);

        transition:.35s;
    }

    div[data-testid="stMetric"]:hover{

        transform:
            translateY(-8px)
            scale(1.02);

        border:1px solid #8b5cf6;

        box-shadow:
            0 25px 60px rgba(124,58,237,.40);

    }

    div[data-testid="stMetric"] label{

        color:#DDD6FE !important;

        font-size:15px;

        letter-spacing:.5px;

        font-weight:700;

    }
    

    /* Metric Label */
    div[data-testid="stMetricLabel"] p{
        color:#DDD6FE !important;
        font-size:15px !important;
        font-weight:700 !important;
        white-space:normal !important;
        overflow:visible !important;
        text-overflow:unset !important;
        line-height:1.3 !important;
    }

    /* Metric Value */
    div[data-testid="stMetricValue"]{
        color:white !important;
        font-size:36px !important;
        font-weight:800 !important;
    }

    .risk-high {
        color: #fecaca;
        background: #7f1d1d;
        padding: 10px 14px;
        border-radius: 12px;
        font-weight: 800;
    }

    .risk-medium {
        color: #fde68a;
        background: #713f12;
        padding: 10px 14px;
        border-radius: 12px;
        font-weight: 800;
    }

    .risk-low {
        color: #bbf7d0;
        background: #14532d;
        padding: 10px 14px;
        border-radius: 12px;
        font-weight: 800;
    }
    /* Sidebar Navigation Cards */

    section[data-testid="stSidebar"] label {
        font-size: 15px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 6px 12px;
        margin-bottom: 6px;
        border: 1px solid rgba(255,255,255,0.08);
        transition: 0.3s;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
        background:#6d28d9;
        transform:translateX(5px);
    }

    section[data-testid="stSidebar"] hr{
        border-color:#5b21b6;
    }

    /* Selected navigation card */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
        background: rgba(192,132,252,0.10) !important;
        border: 1.5px solid #C084FC !important;
        box-shadow: 0 0 12px rgba(192,132,252,0.25);
    }

    /* Selected radio outer circle */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) 
    div[class*="st-b0"]{
        background: #C084FC !important;
        border-color: #C084FC !important;
    }

    /* Selected radio inner dot */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) 
    div[class*="st-b0"]::after{
        background: #C084FC !important;
    }

    .prediction-low {
        background: linear-gradient(135deg, #064e3b, #14532d);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #22c55e;
        box-shadow: 0 10px 30px rgba(34,197,94,0.25);
        margin-top: 20px;
    }

    .prediction-medium {
        background: linear-gradient(135deg, #78350f, #92400e);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #f59e0b;
        box-shadow: 0 10px 30px rgba(245,158,11,0.25);
        margin-top: 20px;
    }

    .prediction-high {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        padding: 24px;
        border-radius: 20px;
        border: 1px solid #ef4444;
        box-shadow: 0 10px 30px rgba(239,68,68,0.25);
        margin-top: 20px;
    }

    .prediction-low h1,
    .prediction-medium h1,
    .prediction-high h1 {
        color: white;
        font-size: 42px;
        margin: 8px 0;
    }

    .prediction-low h2,
    .prediction-medium h2,
    .prediction-high h2 {
        color: white;
        margin-bottom: 8px;
    }

    .prediction-low p,
    .prediction-medium p,
    .prediction-high p {
        color: #f8fafc;
        font-size: 16px;
    }

    .info-card {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(196, 181, 253, 0.30);
        padding: 16px 18px;
        border-radius: 16px;
        color: #ede9fe;
        margin-bottom: 18px;
    }   

    .ai-response-card{
        background:rgba(255,255,255,0.07);
        border:1px solid rgba(192,132,252,0.30);
        border-radius:22px;
        padding:24px;
        margin-top:20px;
        box-shadow:0 14px 35px rgba(0,0,0,0.28);
    }

    .ai-response-card h3{
        color:white;
        margin-bottom:12px;
    }

    .ai-response-card p{
        color:#ddd6fe;
        font-size:16px;
        line-height:1.6;
    }

    /* Premium DataFrame Styling */

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }

    /* Download Buttons */

    .stDownloadButton > button {
        width: 100%;
        border-radius: 14px;
        background: linear-gradient(135deg,#7c3aed,#9333ea);
        color: white;
        font-weight: 700;
        border: none;
        transition: all .3s ease;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(124,58,237,.35);
    }

    /* Primary Buttons */

    .stButton > button {
        border-radius: 14px;
        font-weight: 700;
        transition: all .3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
    }

    /* Page Spacing */

    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    hr {
        border: none;
        height: 1px;
        background: rgba(255,255,255,0.12);
        margin: 28px 0;
    }

    /* Progress Bar */

    div[data-testid="stProgressBar"] > div > div {
        background: linear-gradient(
            90deg,
            #7c3aed,
            #8b5cf6,
            #a855f7
        ) !important;
        border-radius: 12px;
    }

    div[data-testid="stProgressBar"] {
        border-radius: 12px;
    }

    /* Alert / Message Boxes */

    div[data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 8px 24px rgba(0,0,0,0.22);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> FraudPredictor:
    """Load the saved model bundle once."""
    return FraudPredictor(MODEL_PATH)


def plot_dataframe_bar(dataframe: pd.DataFrame, x: str, y: str, title: str) -> None:
    """Render a bar chart with Plotly when available, otherwise Streamlit chart."""
    if px is not None:
        fig = px.bar(dataframe, x=x, y=y, title=title, color=y, color_continuous_scale="Purples")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            height=460,
            margin=dict(l=20, r=20, t=60, b=40),
            title_font=dict(size=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(dataframe.set_index(x)[y])


def show_metrics() -> None:
    """Show saved model metrics."""
    summary_path = REPORTS_DIR / "training_summary.json"
    if not summary_path.exists():
        st.info("Train the model to populate metrics.")
        return
    summary = load_json(summary_path)
    metrics = summary.get("best_metrics", {})
    cols = st.columns(5)
    cols[0].metric("Model", summary.get("best_model", "Unknown"))
    cols[1].metric("Recall", f"{metrics.get('recall', 0):.3f}")
    cols[2].metric("Precision", f"{metrics.get('precision', 0):.3f}")
    cols[3].metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
    cols[4].metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")


def show_feature_importance(predictor: FraudPredictor) -> None:
    """Show global feature importance."""
    importance = ExplanationEngine(predictor.bundle).global_feature_importance(20)
    if importance.empty:
        st.info("Feature importance is unavailable for this model.")
        return
    plot_dataframe_bar(importance.sort_values("importance"), "importance", "feature", "Global Feature Importance")


def show_suspicious_providers() -> None:
    """Show provider intelligence analytics."""
    st.subheader("Provider Intelligence")
    st.caption("Identify high-volume and suspicious healthcare providers.")

    provider_path = REPORTS_DIR / "top_provider_patterns.csv"

    if not provider_path.exists():
        st.info("Provider analytics will appear after training.")
        return

    providers = pd.read_csv(provider_path)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Providers", providers["Provider"].nunique())
    col2.metric("Highest Claim Volume", int(providers["claims"].max()))
    col3.metric("Average Claims", f"{providers['claims'].mean():.1f}")

    st.markdown("### Provider Risk Table")
    st.dataframe(providers, use_container_width=True)

    st.markdown("### Top Providers by Claim Volume")
    plot_dataframe_bar(
        providers.head(15),
        "claims",
        "Provider",
        "Top Providers by Claim Volume",
    )


def single_prediction(predictor: FraudPredictor) -> None:
    """Single prediction panel."""
    st.subheader("Claim Prediction Center")
    st.caption("Enter claim details and get AI-based fraud probability, risk level, and explanation.")

    example = {
        "BeneID": "BENE_SAMPLE",
        "ClaimID": "CLM_SAMPLE",
        "Provider": "PRV_SAMPLE",
        "ClaimType": "Outpatient",
        "ClaimStartDt": "2009-06-01",
        "ClaimEndDt": "2009-06-01",
        "InscClaimAmtReimbursed": 1200,
        "DeductibleAmtPaid": 0,
        "Gender": 1,
        "Race": 1,
        "RenalDiseaseIndicator": 0,
        "State": 39,
        "County": 230,
        "DOB": "1943-01-01",
        "IPAnnualReimbursementAmt": 0,
        "IPAnnualDeductibleAmt": 0,
        "OPAnnualReimbursementAmt": 1200,
        "OPAnnualDeductibleAmt": 0,
    }

    st.markdown("### Claim Input")
    text = st.text_area(
        "Claim Data (JSON)",
        value=str(example).replace("'", '"'),
        height=320,
        help="Paste a valid healthcare insurance claim in JSON format.",
    )

    analyze = st.button(
        "Analyze Claim",
        type="primary",
        use_container_width=True,
    )

    if analyze:
        try:
            import json

            claim = json.loads(text)
            result = predictor.predict_single(claim)

            probability_text = result["fraud_probability"]
            probability_value = float(probability_text.replace("%", "")) / 100
            risk = result["risk_level"]
            prediction = result["fraud_prediction"]

            if risk.lower() == "high":
                card_class = "prediction-high"
            elif risk.lower() == "medium":
                card_class = "prediction-medium"
            else:
                card_class = "prediction-low"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <h2>{prediction}</h2>
                    <p>Fraud Probability</p>
                    <h1>{probability_text}</h1>
                    <p>Risk Level: <b>{risk}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(probability_value)

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Fraud Probability",
                    probability_text,
                )

            with col2:
                st.metric(
                    "Risk Level",
                    risk,
                )

            st.markdown("---")

            st.markdown("### AI Explanation")

            for reason in result["explanation"]:
                st.success(reason)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Check that the JSON is valid and contains claim fields like BeneID, ClaimID, Provider, ClaimStartDt, ClaimEndDt, and InscClaimAmtReimbursed."
        )
    


def batch_prediction(predictor: FraudPredictor) -> None:
    """Batch prediction panel."""
    st.subheader("Batch Claim Analysis Center")
    st.caption("Upload a claim dataset and generate fraud predictions, risk levels, and downloadable reports.")

    st.markdown(
        """
        <div class="info-card">
            Upload a merged claim CSV file to analyze multiple insurance claims together.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload Merged Claim CSV File",
        type=["csv"],
    )

    if uploaded is None:
        st.info("Waiting for CSV upload.")
        return

    data = pd.read_csv(uploaded)

    st.markdown("### Uploaded Data Preview")
    st.dataframe(data.head(20), use_container_width=True)

    st.caption(f"Uploaded file contains {len(data):,} rows and {len(data.columns):,} columns.")

    try:
        with st.spinner("Analyzing claims..."):
            predictions = predictor.predict_dataframe(data)

    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")
        st.info("Check that your uploaded CSV has valid claim columns and no unsupported values.")
        return

    # Prediction history saving disabled for cloud deployment
    # This prevents memory crash on Render while still allowing full CSV prediction.
    #history_path = PROCESSED_DATA_DIR / "prediction_history.csv"

    #if history_path.exists():
    #    history = pd.concat(
    #        [pd.read_csv(history_path), predictions],
    #        ignore_index=True,
    #    )
    #else:
    #    history = predictions

    #history.to_csv(history_path, index=False)

    st.success("Batch analysis completed successfully.")

    if "fraud_probability" in predictions.columns:
        avg_risk = predictions["fraud_probability"].mean()
        total_claims = len(predictions)
        fraud_cases = (predictions["fraud_prediction"] == "Fraud").sum()
        safe_cases = total_claims - fraud_cases

        st.markdown("## Analysis Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(" Total Claims", f"{total_claims:,}")

        with col2:
            st.metric(" Fraud Cases", f"{fraud_cases:,}")

        with col3:
            st.metric(" Safe Claims", f"{safe_cases:,}")

        with col4:
            st.metric(" Average Fraud Risk", f"{avg_risk:.1%}")

        risk_counts = predictions["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]

        st.markdown("### Risk Level Distribution")

        plot_dataframe_bar(
            risk_counts,
            "risk_level",
            "count",
            "Risk Level Distribution",
        )

    if "risk_level" in predictions.columns:
        st.markdown("### Risk Level Summary")

        high = (predictions["risk_level"] == "High").sum()
        medium = (predictions["risk_level"] == "Medium").sum()
        low = (predictions["risk_level"] == "Low").sum()

        c1, c2, c3 = st.columns(3)

        c1.success(f"High Risk Claims: {high}")
        c2.warning(f"Medium Risk Claims: {medium}")
        c3.info(f"Low Risk Claims: {low}")

    st.markdown("### Prediction Results Preview")
    st.caption("Showing first 100 prediction records.")

    st.dataframe(
        predictions.head(100),
        use_container_width=True,
    )

    st.download_button(
        "Download Fraud Prediction Report",
        predictions.to_csv(index=False),
        file_name="fraud_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    

def show_history() -> None:
    """Show saved prediction history."""
    st.subheader("Prediction History")
    st.caption("Review previous batch prediction results.")
    st.caption("The history not available due  to deployement issues ")
    history_path = PROCESSED_DATA_DIR / "prediction_history.csv"

    if not history_path.exists():
        st.info("No prediction history yet.")
        return

    history = pd.read_csv(history_path)

    col1, col2, col3 = st.columns(3)

    col1.metric("Saved Predictions", len(history))

    if "fraud_prediction" in history.columns:
        fraud_cases = (history["fraud_prediction"] == "Fraud").sum()
        col2.metric("Fraud Cases", fraud_cases)
    else:
        col2.metric("Fraud Cases", "N/A")

    if "risk_level" in history.columns:
        high_risk = (history["risk_level"] == "High").sum()
        col3.metric("High Risk Claims", high_risk)
    else:
        col3.metric("High Risk Claims", "N/A")

    st.markdown("### Recent Prediction Records")
    st.dataframe(history.tail(200), use_container_width=True)

    st.download_button(
        "Download Full History",
        history.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

def show_reports() -> None:
    """Show project reports and allow downloads."""
    st.subheader("Reports Center")
    st.caption("Download model summaries, prediction history, and generated project reports.")

    report_path = REPORTS_DIR / "project_report.md"
    summary_path = REPORTS_DIR / "training_summary.json"
    history_path = PROCESSED_DATA_DIR / "prediction_history.csv"

    available_reports = sum([
        report_path.exists(),
        summary_path.exists(),
        history_path.exists(),
    ])

    col1, col2, col3 = st.columns(3)

    col1.metric("Available Reports", available_reports)
    col2.metric("Report Formats", "MD / JSON / CSV")
    col3.metric("Export Status", "Ready")

    st.markdown("---")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("### Project Report")
        if report_path.exists():
            st.success("Available")
            st.download_button(
                "Download Project Report",
                report_path.read_text(encoding="utf-8"),
                file_name="project_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("Project report not found.")

    with col5:
        st.markdown("### Training Summary")
        if summary_path.exists():
            st.success("Available")
            st.download_button(
                "Download Training Summary",
                summary_path.read_text(encoding="utf-8"),
                file_name="training_summary.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("Training summary not found.")

    with col6:
        st.markdown("### Prediction History")
        if history_path.exists():
            history = pd.read_csv(history_path)
            st.success("Available")
            st.download_button(
                "Download Prediction History",
                history.to_csv(index=False),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Prediction history not found.")

def show_dashboard_overview() -> None:
    """Executive dashboard."""

    st.subheader("Executive Dashboard")
    st.caption("Healthcare Fraud Detection System Overview")

    history_path = PROCESSED_DATA_DIR / "prediction_history.csv"

    total_claims = 0
    fraud_cases = 0
    fraud_rate = 0

    if history_path.exists():
        history = pd.read_csv(history_path)

        total_claims = len(history)

        if "fraud_prediction" in history.columns:
            fraud_cases = (
                history["fraud_prediction"] == "Fraud"
            ).sum()

        if total_claims > 0:
            fraud_rate = fraud_cases / total_claims

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="section-card">
                <div style="color:#c4b5fd; font-size:15px;">Total Claims</div>
                <div style="color:white; font-size:34px; font-weight:800;">{total_claims:,}</div>
                <div style="color:#ddd6fe;">Claims analyzed by the system</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="section-card">
                <div style="color:#fecaca; font-size:15px;">Fraud Cases</div>
                <div style="color:white; font-size:34px; font-weight:800;">{fraud_cases:,}</div>
                <div style="color:#ddd6fe;">Claims predicted as suspicious</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="section-card">
                <div style="color:#fde68a; font-size:15px;">Fraud Rate</div>
                <div style="color:white; font-size:34px; font-weight:800;">{fraud_rate:.1%}</div>
                <div style="color:#ddd6fe;">Percentage of risky claims</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    summary_path = REPORTS_DIR / "training_summary.json"

    model_name = "Unknown"
    recall = 0
    precision = 0
    roc_auc = 0

    if summary_path.exists():
        summary = load_json(summary_path)
        metrics = summary.get("best_metrics", {})
        model_name = summary.get("best_model", "Unknown")
        recall = metrics.get("recall", 0)
        precision = metrics.get("precision", 0)
        roc_auc = metrics.get("roc_auc", 0)

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Best Model", model_name)

    with col5:
        st.metric("Recall", f"{recall:.3f}")

    with col6:
        st.metric("ROC-AUC", f"{roc_auc:.3f}")

    st.info(
        "Welcome to the Healthcare Fraud Detection Intelligence Platform."
    )

    if history_path.exists() and "fraud_prediction" in history.columns:
        st.markdown("### Fraud Prediction Summary")

        fraud_summary = history["fraud_prediction"].value_counts().reset_index()
        fraud_summary.columns = ["Prediction", "Count"]

        plot_dataframe_bar(
            fraud_summary,
            "Prediction",
            "Count",
            "Fraud vs Non-Fraud Claims",
        )

def show_ai_assistant() -> None:
    """Enterprise AI assistant panel."""

    st.subheader("AI Assistant")
    st.caption("Ask questions about fraud risk, model performance, provider behavior, and dashboard results.")

    st.markdown(
        """
        <div class="info-card">
            This assistant helps explain fraud analytics, model metrics, provider intelligence, and prediction results in simple language.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Quick Questions")

    q1, q2, q3 = st.columns(3)

    with q1:
        if st.button("Explain fraud risk", use_container_width=True):
            st.session_state["assistant_question"] = "Explain the main fraud risk signals in this dashboard."

    with q2:
        if st.button("Provider warning signs", use_container_width=True):
            st.session_state["assistant_question"] = "What provider behavior should be investigated first?"

    with q3:
        if st.button("Audit recommendations", use_container_width=True):
            st.session_state["assistant_question"] = "Give audit recommendations based on this fraud dashboard."

    question = st.text_area(
        "Ask the AI Assistant",
        value=st.session_state.get("assistant_question", ""),
        placeholder="Example: Why is fraud rate high? Which providers look suspicious? How should I explain this dashboard?",
        height=130,
    )

    if st.button("Generate Insight", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
            return

        st.markdown("### AI Response")

        if "provider" in question.lower():
            insight_type = "Provider Intelligence"
            ai_text = """
            Provider warning signs include unusually high claim volume, repeated claims with similar amounts,
            frequent high-risk predictions, abnormal reimbursement patterns, and providers with many suspicious claims.
            These providers should be reviewed first in the Provider Intelligence section.
            """

        elif "audit" in question.lower():
            insight_type = "Audit Recommendation"
            ai_text = """
            Start the audit with high-risk claims, then review providers with large claim volumes and repeated suspicious patterns.
            Export prediction reports, compare fraud probability with explanations, and document claim-level evidence before escalation.
            """

        elif "risk" in question.lower() or "fraud" in question.lower():
            insight_type = "Fraud Risk Analysis"
            ai_text = """
            Fraud risk is mainly driven by abnormal claim amounts, suspicious provider behavior, repeated claim patterns,
            unusual beneficiary or diagnosis combinations, and strong model probability signals.
            High-risk claims should be prioritized for manual review.
            """

        else:
            insight_type = "General Dashboard Insight"
            ai_text = """
            This dashboard helps identify suspicious insurance claims using machine learning predictions,
            provider intelligence, model metrics, and explainable AI insights.
            Use the prediction probability, risk level, and explanation together before making audit decisions.
            """

        st.html(
            f"""
            <div class="ai-response-card">
                <div style="color:#C4B5FD; font-size:14px; font-weight:700; margin-bottom:8px;">
                    Insight Type: {insight_type}
                </div>

                <h3>AI Insight</h3>

                <p>{ai_text}</p>

                <div style="margin-top:18px; color:#86EFAC; font-weight:700;">
                    Confidence: High
                </div>
            </div>
            """
        )

        st.markdown("### Recommended Actions")

        if insight_type == "Provider Intelligence":
            st.info(
                "1. Open Provider Intelligence.\n\n"
                "2. Sort providers by claim volume.\n\n"
                "3. Review providers linked with repeated suspicious claims.\n\n"
                "4. Export provider data for audit review."
            )

        elif insight_type == "Audit Recommendation":
            st.info(
                "1. Start with high-risk claims.\n\n"
                "2. Compare prediction probability with explanations.\n\n"
                "3. Validate claim details manually.\n\n"
                "4. Export the final report."
            )

        elif insight_type == "Fraud Risk Analysis":
            st.info(
                "1. Prioritize high-risk claims.\n\n"
                "2. Check claim amount and provider pattern.\n\n"
                "3. Review model explanation.\n\n"
                "4. Mark suspicious cases for investigation."
        )

        else:
            st.info(
                "1. Review dashboard KPIs.\n\n"
                "2. Check AI Insights.\n\n"
                "3. Analyze provider behavior.\n\n"
                "4. Export reports if needed."
            )

        if "assistant_history" not in st.session_state:
            st.session_state["assistant_history"] = []

        st.session_state["assistant_history"].append(
            {
                "question": question,
                "insight_type": insight_type,
                "confidence": "High",
            }
        )

        st.markdown("### Recent AI Questions")

        for item in st.session_state["assistant_history"][-5:][::-1]:
            st.markdown(
                f"""
                <div class="info-card">
                    <b>Question:</b> {item["question"]}<br>
                    <b>Insight Type:</b> {item["insight_type"]}<br>
                    <b>Confidence:</b> {item["confidence"]}
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Healthcare Fraud Detection Intelligence Platform</h1>
            <p>AI-powered insurance claim analysis, fraud prediction, provider intelligence, explainable AI, and enterprise reporting.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error("Model artifact not found. Run `python -m src.train` first.")
        return

    predictor = load_predictor()
    
    with st.sidebar:
        st.html(
        """
        <div style="text-align:center; padding:10px 8px;">
            <div style="
                width:52px;
                height:52px;
                margin:0 auto 8px auto;
                border-radius:14px;
                background:linear-gradient(135deg,#7c3aed,#9333ea);
                display:flex;
                align-items:center;
                justify-content:center;
                color:white;
                font-size:24px;
                font-weight:900;
                box-shadow:0 8px 20px rgba(124,58,237,0.38);
            ">
                AI
            </div>

            <div style="color:white; font-size:18px; font-weight:800;">
                Healthcare AI
            </div>

            <div style="color:#c4b5fd; font-size:11px; margin-top:2px;">
                Fraud Intelligence Platform
            </div>
        </div>
        """,
    )

    st.sidebar.markdown("---")
    

    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard Overview",
            "Claim Prediction",
            "Batch Analysis",
            "Analytics Dashboard",
            "AI Insights",
            "AI Assistant",
            "Provider Intelligence",
            "Reports",
            "Prediction History",
        ],
    )

    st.sidebar.markdown("---")

    with st.sidebar:
        st.html(
        """
        <div style="
            background:rgba(255,255,255,0.06);
            border-radius:18px;
            padding:18px;
            border:1px solid rgba(255,255,255,0.08);
            text-align:center;
        ">

            <div style="
                color:#4ADE80;
                font-size:16px;
                font-weight:800;
                margin-bottom:10px;
            ">
                ● Model Online
            </div>

            <div style="color:white;font-size:15px;">
                Version <b>1.0</b>
            </div>

            <div style="
                color:#c4b5fd;
                margin-top:10px;
                font-size:13px;
            ">
                Healthcare Fraud Detection
            </div>

            <div style="
                color:#c4b5fd;
                font-size:12px;
                margin-top:6px;
            ">
                © 2026 Ayush Kumar Pal
            </div>

        </div>
        """,
    )

    show_metrics()

    if page == "Dashboard Overview":
        show_dashboard_overview()

    elif page == "Claim Prediction":
        single_prediction(predictor)
    elif page == "Batch Analysis":
        batch_prediction(predictor)

    elif page == "Analytics Dashboard":
        st.subheader("Fraud Analytics Dashboard")
        st.caption("Visual analysis of claim patterns, model performance, and fraud trends.")

        analytics_images = [
            ("class_distribution.png", "Fraud Class Distribution"),
            ("claim_amount_histogram.png", "Claim Amount Distribution"),
            ("claim_amount_by_fraud_boxplot.png", "Claim Amount by Fraud Status"),
            ("age_distribution.png", "Beneficiary Age Distribution"),
            ("monthly_fraud_trends.png", "Monthly Fraud Trends"),
            ("confusion_matrix.png", "Confusion Matrix"),
            ("precision_recall_curve.png", "Precision Recall Curve"),
            ("roc_curve.png", "ROC Curve"),
            ("correlation_heatmap.png", "Correlation Heatmap"),
        ]

        col1, col2 = st.columns(2)

        for index, (image_name, title) in enumerate(analytics_images):
            image_path = REPORTS_DIR / image_name

            if image_path.exists():
                with col1 if index % 2 == 0 else col2:
                    st.markdown(f"#### {title}")
                    st.image(str(image_path), use_container_width=True)
            else:
                st.info(f"{title} chart not found. Run training/evaluation to generate it.")


    elif page == "AI Insights":
        st.subheader("AI Insights")
        st.caption("Understand how the trained AI model makes fraud predictions.")

        summary_path = REPORTS_DIR / "training_summary.json"

        if summary_path.exists():
            summary = load_json(summary_path)
            metrics = summary.get("best_metrics", {})

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Best Model", summary.get("best_model", "Unknown"))
            c2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
            c3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            c4.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

        st.markdown("---")

        st.markdown("### Global Feature Importance")

        show_feature_importance(predictor)

        st.info(
            "Feature importance indicates which claim, beneficiary, and provider attributes contribute most to fraud prediction."
        )

    elif page == "AI Assistant":
        show_ai_assistant()

    elif page == "Provider Intelligence":
        show_suspicious_providers()
    
    elif page == "Reports":
        show_reports()   
    else:
        show_history()

    st.markdown("---")
    st.markdown(
        """
        <div style="
            text-align:center;
            color:#c4b5fd;
            padding:18px;
            font-size:14px;
        ">
            Healthcare Fraud Detection Intelligence Platform |
            Model Version 1.0 |
            Developed by Ayush Kumar Pal |
            © 2026
        </div>
        """,
        unsafe_allow_html=True,
    )
        

if __name__ == "__main__":
    main()

