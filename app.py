"""
Disease Outcome Predictor  ·  Streamlit App
──────────────────────────────────────────────
A clinical decision-support tool that predicts whether a patient's
disease outcome is likely to be Positive (requires treatment/attention)
or Negative, with full SHAP-based explainability.
"""

import streamlit as st
import pandas as pd
from consultant import (
    load_and_preprocess,
    train_model,
    predict_patient,
    explain_single_prediction,
    explain_global,
    disease_stats,
    symptom_correlation,
    DATASET_PATH,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Outcome Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

  /* Dark teal accent */
  .stButton > button {
    background: #00d4aa; color: #0f1117;
    font-weight: 700; border: none; border-radius: 6px;
    padding: 0.6rem 1.4rem; font-family: 'Syne', sans-serif;
    font-size: 0.9rem; transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  .metric-card {
    background: #1a1f2e; border: 1px solid #2a3040;
    border-radius: 10px; padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .metric-label { color: #7a8599; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { color: #00d4aa; font-family: 'DM Mono', monospace; font-size: 2rem; font-weight: 500; }

  .result-positive {
    background: linear-gradient(135deg, #1a0a0a, #2a1010);
    border: 1px solid #ff4b4b; border-radius: 10px; padding: 1.2rem;
  }
  .result-negative {
    background: linear-gradient(135deg, #0a1a14, #0a2a1e);
    border: 1px solid #00d4aa; border-radius: 10px; padding: 1.2rem;
  }
  .result-label { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 800; }
  .result-sub   { color: #7a8599; font-size: 0.85rem; margin-top: 0.3rem; }

  .disclaimer {
    background: #1a1f2e; border-left: 3px solid #f0a500;
    border-radius: 4px; padding: 0.8rem 1rem;
    font-size: 0.78rem; color: #9aa3b5; margin-top: 1rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2535; }
  [data-testid="stSidebar"] label { color: #c0c8d8 !important; font-size: 0.85rem; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { font-family: 'Syne', sans-serif; font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)


# ── Data & model (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on dataset…")
def get_model():
    df = load_and_preprocess(DATASET_PATH)
    model, X_test, y_test, features, metrics = train_model(df)
    return model, X_test, y_test, features, metrics, df


model, X_test, y_test, features, metrics, df = get_model()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🩺 Disease Outcome Predictor")
st.markdown(
    "<p style='color:#7a8599; margin-top:-0.5rem; margin-bottom:1.5rem;'>"
    "Enter patient profile → get an ML-powered outcome prediction with SHAP explainability.</p>",
    unsafe_allow_html=True,
)

# Model performance strip
c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "Model Accuracy", f"{metrics['accuracy']}%"),
    (c2, "ROC-AUC",        f"{metrics['roc_auc']}%"),
    (c3, "Training Rows",  f"{len(df):,}"),
    (c4, "Unique Diseases", f"{df['Disease'].nunique()}"),
]:
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_explore, tab_global = st.tabs(
    ["🔬 Predict", "📊 Data Explorer", "🌍 Global Explainability"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Predict
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Patient Symptom Profile")
    st.markdown(
        "<p style='color:#7a8599; font-size:0.85rem;'>Fill in the patient's details. "
        "All fields are required.</p>",
        unsafe_allow_html=True,
    )

    with st.form("patient_form"):
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**🤒 Symptoms**")
            fever      = st.selectbox("Fever",               ["Yes", "No"])
            cough      = st.selectbox("Cough",               ["Yes", "No"])
            fatigue    = st.selectbox("Fatigue",             ["Yes", "No"])
            breathing  = st.selectbox("Difficulty Breathing",["Yes", "No"])

        with col_r:
            st.markdown("**👤 Patient Profile**")
            age        = st.slider("Age", 1, 100, 35)
            gender     = st.selectbox("Gender", ["Female", "Male"])
            bp         = st.selectbox("Blood Pressure",    ["Low", "Normal", "High"])
            cholesterol= st.selectbox("Cholesterol Level", ["Low", "Normal", "High"])

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if submitted:
        patient = {
            "Fever": fever, "Cough": cough,
            "Fatigue": fatigue, "Difficulty Breathing": breathing,
            "Age": age, "Gender": gender,
            "Blood Pressure": bp, "Cholesterol Level": cholesterol,
        }

        result = predict_patient(model, patient)
        label  = result["label"]
        prob   = result["probability"]
        inp_df = result["input_df"]

        # Result card
        css_class = "result-positive" if label == "Positive" else "result-negative"
        colour    = "#ff4b4b" if label == "Positive" else "#00d4aa"
        icon      = "⚠️" if label == "Positive" else "✅"
        advice    = (
            "This patient profile suggests a <strong>positive disease outcome</strong> — "
            "clinical follow-up is recommended."
            if label == "Positive"
            else "This patient profile suggests a <strong>negative (non-critical) outcome</strong>. "
            "Routine monitoring advised."
        )

        st.markdown(
            f"<div class='{css_class}'>"
            f"<div class='result-label' style='color:{colour};'>{icon} {label} Outcome</div>"
            f"<div class='result-sub'>Model confidence: <strong>{prob}%</strong></div>"
            f"<hr style='border-color:#2a3040; margin: 0.8rem 0;'>"
            f"<p style='font-size:0.85rem; color:#c0c8d8; margin:0;'>{advice}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # SHAP waterfall
        st.markdown("<br>**🔍 Why this prediction? (SHAP Explanation)**", unsafe_allow_html=True)
        st.caption(
            "Each bar shows how much a feature pushed the prediction towards Positive (red) "
            "or Negative (blue) compared to the average."
        )
        with st.spinner("Generating SHAP explanation…"):
            shap_img = explain_single_prediction(model, inp_df)
        st.image(
            f"data:image/png;base64,{shap_img}",
            use_container_width=True,
        )

        st.markdown(
            "<div class='disclaimer'>⚠️ <strong>Disclaimer:</strong> This tool is for "
            "educational and research purposes only. It is <em>not</em> a substitute for "
            "professional medical advice, diagnosis, or treatment.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Top 20 Diseases by Frequency")
        stats = disease_stats(df)
        st.dataframe(
            stats.rename(columns={
                "Disease": "Disease",
                "count": "Patient Count",
                "positive_rate": "Positive Outcome %",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with col_b:
        st.markdown("### Symptom–Outcome Correlation")
        corr = symptom_correlation(df)
        corr_df = corr.reset_index()
        corr_df.columns = ["Feature", "Correlation with Positive Outcome"]
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

        st.caption(
            "Positive values → feature associated with positive (worse) outcomes. "
            "Negative values → associated with negative (better) outcomes."
        )

    st.markdown("### Raw Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Global Explainability
# ══════════════════════════════════════════════════════════════════════════════
with tab_global:
    st.markdown("### Global Feature Importance (SHAP)")
    st.markdown(
        "<p style='color:#7a8599; font-size:0.85rem;'>"
        "Across all patients in the test set, which features matter most to the model?</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Computing global SHAP values…"):
        global_img = explain_global(model, X_test)

    st.image(f"data:image/png;base64,{global_img}", use_container_width=True)

    st.markdown("### Model Performance Report")
    report = metrics["report"]
    report_df = pd.DataFrame(report).T
    st.dataframe(
        report_df.style.format("{:.2f}").background_gradient(
            cmap="YlGn", subset=["precision", "recall", "f1-score"]
        ),
        use_container_width=True,
    )

    st.caption(
        f"Accuracy: **{metrics['accuracy']}%** · ROC-AUC: **{metrics['roc_auc']}%** · "
        "Model: Random Forest (200 trees, balanced class weights)"
    )