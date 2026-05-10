import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import io, base64, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import lightgbm as lgb

# ── Constants ─────────────────────────────────────────────────────────────────
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

# ── Preprocessing ─────────────────────────────────────────────────────────────
def load_and_preprocess(path: str = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in BINARY_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    for col, mapping in ORDINAL_COLS.items():
        df[col] = df[col].map(mapping)
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    df[TARGET_COL] = df[TARGET_COL].map({"Positive": 1, "Negative": 0})
    return df


# ── Model candidates ──────────────────────────────────────────────────────────
def _candidate_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=42,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            class_weight="balanced", random_state=42, verbose=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42,
            )),
        ]),
    }


# ── Benchmarking (cross-validation) ──────────────────────────────────────────
def benchmark_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run 5-fold stratified CV on all candidate models.
    Returns a DataFrame of mean CV scores, sorted by ROC-AUC.
    This gives a much more honest accuracy estimate than a single split,
    especially important with only ~349 rows.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []
    for name, model in _candidate_models().items():
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["accuracy", "roc_auc", "f1"],
            return_train_score=False,
        )
        rows.append({
            "Model":    name,
            "Accuracy": round(scores["test_accuracy"].mean() * 100, 1),
            "ROC-AUC":  round(scores["test_roc_auc"].mean() * 100, 1),
            "F1 Score": round(scores["test_f1"].mean() * 100, 1),
            # std tells us how stable the model is across folds
            "AUC Std":  round(scores["test_roc_auc"].std() * 100, 1),
        })

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return results


# ── Train the best model on full data ────────────────────────────────────────
def train_best_model(df: pd.DataFrame, model_name: str = None):
    """
    If model_name is given, train that specific model.
    Otherwise run benchmark first and pick the best by ROC-AUC.
    Returns (model, X_test, y_test, metrics, benchmark_df).
    """
    benchmark_df = benchmark_models(df)

    if model_name is None:
        model_name = benchmark_df.iloc[0]["Model"]

    candidates = _candidate_models()
    model = candidates[model_name]

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy":   round(accuracy_score(y_test, y_pred) * 100, 2),
        "roc_auc":    round(roc_auc_score(y_test, y_proba) * 100, 2),
        "f1":         round(f1_score(y_test, y_pred) * 100, 2),
        "report":     classification_report(y_test, y_pred, output_dict=True),
    }

    return model, X_test, y_test, metrics, benchmark_df, model_name


# ── SHAP helpers ──────────────────────────────────────────────────────────────
def _get_shap_values_pos(model, X: pd.DataFrame):
    """
    Extract positive-class SHAP values robustly for any model type.
    Handles both tree models (TreeExplainer) and linear/pipeline models
    (LinearExplainer / KernelExplainer fallback).
    Returns (sv_pos, expected_value_pos) where sv_pos shape = (n, n_features).
    """
    # Unwrap sklearn Pipeline to get the actual estimator
    base_model = model
    if hasattr(model, "named_steps"):
        # Pipeline — get the final estimator; SHAP needs transformed X
        base_model = model.named_steps[list(model.named_steps)[-1]]

    tree_types = (
        RandomForestClassifier,
        GradientBoostingClassifier,
        xgb.XGBClassifier,
        lgb.LGBMClassifier,
    )

    if isinstance(base_model, tree_types):
        explainer = shap.TreeExplainer(base_model)
        # Transform X if pipeline
        X_in = model[:-1].transform(X) if hasattr(model, "named_steps") else X
        sv = explainer.shap_values(X_in)   # shape (n, n_features, 2) or list

        if isinstance(sv, list):
            sv_pos = sv[1]
            ev_pos = float(explainer.expected_value[1])
        elif sv.ndim == 3:
            sv_pos = sv[:, :, 1]
            ev_pos = float(explainer.expected_value[1])
        else:
            sv_pos = sv
            ev_pos = float(explainer.expected_value)
    else:
        # Linear model inside pipeline
        X_transformed = model[:-1].transform(X) if hasattr(model, "named_steps") else X
        explainer = shap.LinearExplainer(base_model, X_transformed)
        sv_pos = explainer.shap_values(X_transformed)
        ev_pos = float(explainer.expected_value)

    return sv_pos, ev_pos, explainer


def explain_single_prediction(model, input_df: pd.DataFrame) -> str:
    """Waterfall chart for a single patient — works for any model type."""
    sv_pos, ev_pos, explainer = _get_shap_values_pos(model, input_df)

    if sv_pos.ndim == 2:
        row_vals = sv_pos[0]
    else:
        row_vals = sv_pos

    explanation = shap.Explanation(
        values        = row_vals,
        base_values   = ev_pos,
        data          = input_df.iloc[0].values,
        feature_names = list(input_df.columns),
    )

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.default"] = "regular"
    plt.style.use("dark_background")
    shap.plots.waterfall(explanation, max_display=8, show=False)
    plt.savefig(io.BytesIO())  # force render before tight_layout
    plt.gcf().set_facecolor("#0f1117")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f1117")
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode()


def explain_global(model, X_test: pd.DataFrame) -> str:
    """Global SHAP bar chart — works for any model type."""
    sv_pos, _, _ = _get_shap_values_pos(model, X_test)

    plt.rcParams["text.usetex"] = False
    plt.rcParams["mathtext.default"] = "regular"
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.summary_plot(sv_pos, X_test, feature_names=list(X_test.columns),
                      plot_type="bar", show=False, color="#00d4aa")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0f1117")
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.read()).decode()


# ── Disease statistics helpers ────────────────────────────────────────────────
def disease_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby(DISEASE_COL)
        .agg(count=(TARGET_COL, "count"), positive_rate=(TARGET_COL, "mean"))
        .sort_values("count", ascending=False)
        .head(20)
        .reset_index()
    )
    stats["positive_rate"] = (stats["positive_rate"] * 100).round(1)
    return stats


def symptom_correlation(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    return corr.sort_values(ascending=False)


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_patient(model, patient: dict) -> dict:
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