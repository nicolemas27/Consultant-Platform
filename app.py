"""
Disease Outcome Predictor  ·  Streamlit App
"""
import streamlit as st
import pandas as pd
from consultant import (
    load_and_preprocess, train_best_model, predict_patient,
    explain_single_prediction, explain_global,
    disease_stats, symptom_correlation, DATASET_PATH,
)

st.set_page_config(
    page_title="Disease Outcome Predictor",
    page_icon="🩺", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist+Mono:wght@300;400;500&family=Geist:wght@300;400;500;600&display=swap');

  /* ── Root tokens ── */
  :root {
    --bg:        #f8fafc;
    --bg-2:      #ffffff;
    --bg-3:      #f1f5f9;
    --border:    rgba(0,0,0,0.09);
    --border-hi: rgba(0,0,0,0.18);
    --teal:      #0d9488;
    --teal-dim:  rgba(13,148,136,0.08);
    --red:       #dc2626;
    --red-dim:   rgba(220,38,38,0.06);
    --amber:     #d97706;
    --text-1:    #0f172a;
    --text-2:    #334155;
    --text-3:    #64748b;
    --mono:      'Geist Mono', monospace;
    --serif:     'Instrument Serif', Georgia, serif;
    --sans:      'Geist', system-ui, sans-serif;
  }

  /* ── Global reset ── */
  html, body, [class*="css"] {
    font-family: var(--sans);
    color: var(--text-1);
  }

  /* Subtle noise texture on body */
  .main > div { background: var(--bg); }
  .block-container { padding-top: 2rem !important; max-width: 1100px; }

  /* ── Typography ── */
  h1, h2, h3 {
    font-family: var(--serif) !important;
    font-weight: 400 !important;
    letter-spacing: -0.01em;
    color: var(--text-1) !important;
  }
  h2 { font-size: 2.1rem !important; }
  h3 { font-size: 1.35rem !important; }
  p  { color: var(--text-2); line-height: 1.65; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--bg-2) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-3) !important;
    margin-bottom: 0.6rem;
  }
  [data-testid="stSidebar"] label { color: var(--text-2) !important; font-size: 0.82rem; }
  [data-testid="stSidebar"] .stSelectbox > div { background: #e8eef5 !important; }

  /* ── Inputs ── */
  .stSelectbox > div > div,
  .stSlider > div { background: var(--bg-3) !important; border-color: var(--border) !important; }
  .stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
  }

  /* ── Button ── */
  .stButton > button, .stFormSubmitButton > button {
    background: var(--teal) !important;
    color: #ffffff !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.6rem !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 0 0 0 rgba(45,212,191,0);
  }
  .stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(13,148,136,0.3) !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-3) !important;
    padding: 0.7rem 1.2rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
  }
  .stTabs [aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom-color: var(--teal) !important;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(13,148,136,0.05) 0%, transparent 70%);
    pointer-events: none;
  }
  .metric-card:hover { border-color: var(--border-hi); }
  .metric-label {
    font-family: var(--mono);
    color: var(--text-3);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
  }
  .metric-value {
    font-family: var(--serif);
    color: var(--teal);
    font-size: 2.4rem;
    line-height: 1;
    font-style: italic;
  }

  /* ── Model badge ── */
  .model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--teal-dim);
    border: 1px solid rgba(13,148,136,0.2);
    border-radius: 100px;
    padding: 0.25rem 0.9rem;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--teal);
    letter-spacing: 0.04em;
    margin-bottom: 1.4rem;
  }
  .model-badge::before {
    content: '';
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--teal);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* ── Result boxes ── */
  .result-positive {
    background: var(--red-dim);
    border: 1px solid rgba(220,38,38,0.2);
    border-radius: 14px;
    padding: 1.6rem;
    position: relative;
    overflow: hidden;
  }
  .result-positive::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--red), transparent);
  }
  .result-negative {
    background: var(--teal-dim);
    border: 1px solid rgba(13,148,136,0.2);
    border-radius: 14px;
    padding: 1.6rem;
    position: relative;
    overflow: hidden;
  }
  .result-negative::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--teal), transparent);
  }
  .result-label {
    font-family: var(--serif);
    font-size: 2rem;
    font-style: italic;
    line-height: 1.1;
    margin-bottom: 0.4rem;
  }
  .result-sub {
    font-family: var(--mono);
    color: var(--text-2);
    font-size: 0.78rem;
    letter-spacing: 0.02em;
  }

  /* ── Disclaimer ── */
  .disclaimer {
    background: rgba(217,119,6,0.06);
    border: 1px solid rgba(217,119,6,0.2);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.78rem;
    font-family: var(--mono);
    color: var(--text-2);
    margin-top: 1.2rem;
    line-height: 1.6;
  }

  /* ── Dataframes ── */
  [data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
  }

  /* ── Captions ── */
  .stCaption, [data-testid="stCaptionContainer"] p {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-3) !important;
    letter-spacing: 0.02em;
  }

  /* ── Info / alerts ── */
  .stAlert { border-radius: 10px !important; border-width: 1px !important; }

  /* ── Dividers ── */
  hr { border-color: var(--border) !important; }

  /* ── Form background ── */
  [data-testid="stForm"] {
    background: var(--bg-2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem !important;
  }

  /* ── Section sub-headers ── */
  .section-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-3);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: model selection ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Model Settings")
    model_choice = st.selectbox(
        "Select model (or let app auto-pick best)",
        ["Auto (best by ROC-AUC)", "Random Forest", "XGBoost",
         "LightGBM", "Gradient Boosting", "Logistic Regression"],
    )
    st.caption(
        "Auto runs 5-fold CV on all models and picks the best. "
        "Manual lets you choose a specific algorithm."
    )
    st.markdown("---")
    st.markdown("### ℹ About")
    st.caption(
        "Predicts whether a patient's disease outcome is likely to require "
        "intervention."
    )


# ── Load data + train ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Benchmarking models & training best…")
def get_model(choice: str):
    df = load_and_preprocess(DATASET_PATH)
    model_name = None if choice == "Auto (best by ROC-AUC)" else choice
    model, X_test, y_test, metrics, benchmark_df, best_name = train_best_model(df, model_name)
    return model, X_test, y_test, metrics, benchmark_df, best_name, df

model, X_test, y_test, metrics, benchmark_df, best_name, df = get_model(model_choice)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='display:flex; align-items:baseline; gap:0.8rem; margin-bottom:0.3rem;'>"
    "<h2 style='margin:0;'>Disease Outcome Predictor</h2>"
    "<span style=\"font-family:'Geist Mono',monospace; font-size:0.72rem; color:#556070; "
    "letter-spacing:0.1em; text-transform:uppercase;\">v2.0</span>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='margin-top:0; margin-bottom:1.2rem; font-size:0.9rem;'>"
    "Enter a patient profile and get an ML-powered outcome prediction with SHAP explainability.</p>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='model-badge'>Active model: {metrics['model_name']}</div>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "Accuracy",      f"{metrics['accuracy']}%"),
    (c2, "ROC-AUC",       f"{metrics['roc_auc']}%"),
    (c3, "F1 Score",      f"{metrics['f1']}%"),
    (c4, "Training Rows", f"{len(df):,}"),
]:
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"</div>", unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_benchmark, tab_explore, tab_global = st.tabs([
    "🔬 Predict", "📈 Model Benchmark", "📊 Data Explorer", "🌍 Global Explainability"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Predict
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Patient Symptom Profile")
    with st.form("patient_form"):
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("<div class='section-label'>Symptoms</div>", unsafe_allow_html=True)
            fever     = st.selectbox("Fever",                ["Yes", "No"])
            cough     = st.selectbox("Cough",                ["Yes", "No"])
            fatigue   = st.selectbox("Fatigue",              ["Yes", "No"])
            breathing = st.selectbox("Difficulty Breathing", ["Yes", "No"])
        with col_r:
            st.markdown("<div class='section-label'>Patient Profile</div>", unsafe_allow_html=True)
            age         = st.slider("Age", 1, 100, 35)
            gender      = st.selectbox("Gender",             ["Female", "Male"])
            bp          = st.selectbox("Blood Pressure",     ["Low", "Normal", "High"])
            cholesterol = st.selectbox("Cholesterol Level",  ["Low", "Normal", "High"])
        submitted = st.form_submit_button("Run Prediction →", use_container_width=True)

    if submitted:
        patient = {
            "Fever": fever, "Cough": cough, "Fatigue": fatigue,
            "Difficulty Breathing": breathing, "Age": age,
            "Gender": gender, "Blood Pressure": bp, "Cholesterol Level": cholesterol,
        }
        result = predict_patient(model, patient)
        label, prob, inp_df = result["label"], result["probability"], result["input_df"]

        css_class = "result-positive" if label == "Positive" else "result-negative"
        colour    = "#f87171"         if label == "Positive" else "#2dd4bf"
        icon      = "⚠"             if label == "Positive" else "✓"
        advice    = (
            "This patient profile suggests a <strong>positive disease outcome</strong> — clinical follow-up recommended."
            if label == "Positive" else
            "This patient profile suggests a <strong>negative (non-critical) outcome</strong>. Routine monitoring advised."
        )
        st.markdown(
            f"<div class='{css_class}'>"
            f"<div class='result-label' style='color:{colour};'>{icon} {label} Outcome</div>"
            f"<div class='result-sub' style='margin-top:0.4rem;'>Confidence: <strong>{prob}%</strong> "
            f"&nbsp;·&nbsp; Model: <strong>{metrics['model_name']}</strong></div>"
            f"<hr style='border-color:rgba(255,255,255,0.07); margin:0.9rem 0;'>"
            f"<p style='font-size:0.84rem; color:#8899aa; margin:0; line-height:1.6;'>{advice}</p>"
            f"</div>", unsafe_allow_html=True,
        )

        st.markdown("<br><div class='section-label'>SHAP Explanation</div>", unsafe_allow_html=True)
        st.caption("Teal bars push toward Negative · Red bars push toward Positive outcome.")
        with st.spinner("Generating SHAP explanation…"):
            shap_img = explain_single_prediction(model, inp_df)
        st.image(f"data:image/png;base64,{shap_img}", use_container_width=True)

        st.markdown(
            "<div class='disclaimer'>⚠ <strong>Disclaimer:</strong> For educational "
            "and research purposes only. Not a substitute for professional medical advice.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Model Benchmark
# ══════════════════════════════════════════════════════════════════════════════
with tab_benchmark:
    st.markdown("### Model Comparison  <span style=\"font-family:'Geist Mono',monospace; font-size:0.72rem; color:#556070; font-weight:400;\">5-fold stratified CV</span>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.84rem;'>"
        "Every model is evaluated with 5-fold cross-validation — much more reliable than a "
        "single train/test split. Sorted by ROC-AUC. Lower AUC Std = more stable across folds.</p>",
        unsafe_allow_html=True,
    )

    # Highlight best row
    def highlight_best(row):
        if row["Model"] == benchmark_df.iloc[0]["Model"]:
            return ["background-color: #0d2b24; color: #00d4aa"] * len(row)
        return [""] * len(row)

    st.dataframe(
        benchmark_df.style.apply(highlight_best, axis=1).format({
            "Accuracy": "{:.1f}%",
            "ROC-AUC":  "{:.1f}%",
            "F1 Score": "{:.1f}%",
            "AUC Std":  "±{:.1f}%",
        }),
        use_container_width=True, hide_index=True,
    )

    st.info(
        f"**Why ROC-AUC over Accuracy?**  With a small, imbalanced dataset, accuracy can be "
        f"misleading — a model that always predicts the majority class looks decent. "
        f"ROC-AUC measures how well the model separates Positive from Negative outcomes "
        f"regardless of class balance. AUC Std tells you how consistent it is across folds.",
        icon="💡",
    )

    st.markdown("### Change Active Model")
    st.caption("Use the sidebar to switch models — the app will retrain and update all metrics.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab_explore:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Top 20 Diseases by Frequency")
        stats = disease_stats(df)
        st.dataframe(
            stats.rename(columns={"Disease": "Disease", "count": "Count", "positive_rate": "Positive Outcome %"}),
            use_container_width=True, hide_index=True,
        )
    with col_b:
        st.markdown("### Symptom–Outcome Correlation")
        corr = symptom_correlation(df)
        corr_df = corr.reset_index()
        corr_df.columns = ["Feature", "Correlation with Positive Outcome"]
        st.dataframe(corr_df, use_container_width=True, hide_index=True)
        st.caption("Positive = associated with worse outcomes. Negative = associated with better outcomes.")

    st.markdown("### Raw Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Global Explainability
# ══════════════════════════════════════════════════════════════════════════════
with tab_global:
    st.markdown("### Global Feature Importance  <span style=\"font-family:'Geist Mono',monospace; font-size:0.72rem; color:#556070; font-weight:400;\">SHAP summary</span>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.84rem;'>"
        "Across all patients in the test set, which features drive predictions most?</p>",
        unsafe_allow_html=True,
    )
    with st.spinner("Computing global SHAP values…"):
        global_img = explain_global(model, X_test)
    st.image(f"data:image/png;base64,{global_img}", use_container_width=True)

    st.markdown("### Full Classification Report")
    report_df = pd.DataFrame(metrics["report"]).T
    st.dataframe(
        report_df.style.format("{:.2f}").background_gradient(
            cmap="YlGn", subset=["precision", "recall", "f1-score"]
        ),
        use_container_width=True,
    )
    st.caption(
        f"Active model: **{metrics['model_name']}** · "
        f"Accuracy: **{metrics['accuracy']}%** · ROC-AUC: **{metrics['roc_auc']}%** · F1: **{metrics['f1']}%**"
    )