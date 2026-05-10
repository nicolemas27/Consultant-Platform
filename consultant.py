import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset/Disease_symptom_and_patient_profile_dataset.csv"

BINARY_COLS = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
ORDINAL_COLS = {
    "Blood Pressure":    {"Low": 0, "Normal": 1, "High": 2},
    "Cholesterol Level": {"Low": 0, "Normal": 1, "High": 2},
}
TARGET_COL   = "Outcome Variable"
DISEASE_COL  = "Disease"
FEATURE_COLS = [
    "Fever", "Cough", "Fatigue", "Difficulty Breathing",
    "Age", "Gender", "Blood Pressure", "Cholesterol Level",
]


# ── Data loading & preprocessing ─────────────────────────────────────────────
def load_and_preprocess(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the CSV and encode all categorical columns."""
    df = pd.read_csv(path)

    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    for col, mapping in ORDINAL_COLS.items():
        df[col] = df[col].map(mapping)

    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])   # Female=0, Male=1

    df[TARGET_COL] = df[TARGET_COL].map({"Positive": 1, "Negative": 0})

    return df


# ── Model training ────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    """
    Train a Random Forest classifier and return
    (model, X_test, y_test, feature_names, metrics_dict).
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "roc_auc":  round(roc_auc_score(y_test, y_proba) * 100, 2),
        "report":   classification_report(y_test, y_pred, output_dict=True),
    }

    return model, X_test, y_test, FEATURE_COLS, metrics


# ── SHAP explanation ──────────────────────────────────────────────────────────
def explain_single_prediction(model, input_df: pd.DataFrame) -> str:
    """
    Waterfall plot for a single patient.

    shap_values() returns shape (1, n_features, 2) for a binary RF.
    Index [0, :, 1] gives the 1-D positive-class SHAP values for row 0.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(input_df)    # shape: (1, n_features, 2)
    sv_pos = sv[0, :, 1]                    # shape: (n_features,)  — positive class
    ev_pos = float(explainer.expected_value[1])  # scalar base value

    explanation = shap.Explanation(
        values        = sv_pos,
        base_values   = ev_pos,
        data          = input_df.iloc[0].values,
        feature_names = list(input_df.columns),
    )

    plt.style.use("dark_background")
    shap.plots.waterfall(explanation, max_display=8, show=False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0f1117")
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode()


def explain_global(model, X_test: pd.DataFrame) -> str:
    """
    Global feature importance bar chart.

    shap_values() returns shape (n_samples, n_features, 2).
    Index [:, :, 1] gives the positive-class SHAP matrix.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)     # shape: (n_samples, n_features, 2)
    sv_pos = sv[:, :, 1]                   # shape: (n_samples, n_features)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.summary_plot(sv_pos, X_test, plot_type="bar",
                      show=False, color="#00d4aa")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0f1117")
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode()


# ── Disease statistics helpers ────────────────────────────────────────────────
def disease_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Top diseases by frequency with positive-outcome rate."""
    stats = (
        df.groupby(DISEASE_COL)
        .agg(
            count=(TARGET_COL, "count"),
            positive_rate=(TARGET_COL, "mean"),
        )
        .sort_values("count", ascending=False)
        .head(20)
        .reset_index()
    )
    stats["positive_rate"] = (stats["positive_rate"] * 100).round(1)
    return stats


def symptom_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of symptom features with outcome."""
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    return corr.sort_values(ascending=False)


# ── Single-patient prediction ─────────────────────────────────────────────────
def predict_patient(model, patient: dict) -> dict:
    """
    Accept a dict of raw patient inputs and return prediction + probability.

    Example:
        {
          "Fever": "Yes", "Cough": "No", "Fatigue": "Yes",
          "Difficulty Breathing": "Yes", "Age": 45,
          "Gender": "Female", "Blood Pressure": "High",
          "Cholesterol Level": "Normal"
        }
    """
    row = {}
    for col in BINARY_COLS:
        row[col] = 1 if str(patient[col]).strip().lower() == "yes" else 0
    for col, mapping in ORDINAL_COLS.items():
        row[col] = mapping.get(str(patient[col]).strip(), 1)
    row["Gender"] = 0 if str(patient["Gender"]).strip().lower() == "female" else 1
    row["Age"]    = int(patient["Age"])

    input_df = pd.DataFrame([row])[FEATURE_COLS]
    prob     = model.predict_proba(input_df)[0][1]
    label    = "Positive" if prob >= 0.5 else "Negative"

    return {"label": label, "probability": round(prob * 100, 1), "input_df": input_df}