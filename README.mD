# 🩺 Disease Outcome Predictor

> **An explainable ML clinical decision-support tool** — predicts whether a patient's disease outcome is likely to require medical intervention, with full SHAP-based reasoning behind every prediction.

---

## 📌 What It Does

A clinician (or researcher) enters a patient's symptom profile — fever, cough, fatigue, breathing difficulty, age, gender, blood pressure, cholesterol — and the app returns:

- ✅ **Positive / Negative outcome prediction** with confidence score
- 🔍 **Per-patient SHAP waterfall chart** — shows exactly *why* the model made that call
- 🌍 **Global feature importance** — which symptoms matter most across all patients
- 📊 **Data explorer** — disease frequency, outcome rates, symptom correlations

**The key differentiator:** this isn't a black-box predictor. Every prediction is explained in plain terms using SHAP (SHapley Additive exPlanations), a game-theory-based technique widely used in production ML systems.

---

## Technical Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Model | Random Forest (scikit-learn) |
| Explainability | SHAP (TreeExplainer + Waterfall plots) |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib |

**Model details:**
- 200 estimators, `max_depth=10`, balanced class weights (handles class imbalance)
- 80/20 stratified train/test split
- Evaluated on Accuracy + ROC-AUC (more robust than accuracy alone for medical data)

---

## 📂 Project Structure

```
disease-outcome-predictor/
│
├── app.py                                           # Streamlit UI (3 tabs: Predict, Explore, Explain)
├── consultant.py                                    # ML pipeline: load → preprocess → train → explain
├── Disease_symptom_and_patient_profile_dataset.csv  # 
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/nicolemas27/Consultant-Platform.git
cd disease-outcome-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download the dataset 

Disease_symptom_and_patient_profile_dataset.csv
```

### 4. Run the app
```bash
streamlit run app.py
```
---

## 📊 Dataset

**Disease Symptom and Patient Profile Dataset**

| Column | Type | Values |
|---|---|---|
| Disease | string | 40+ disease names |
| Fever | binary | Yes / No |
| Cough | binary | Yes / No |
| Fatigue | binary | Yes / No |
| Difficulty Breathing | binary | Yes / No |
| Age | integer | 1–100 |
| Gender | categorical | Male / Female |
| Blood Pressure | ordinal | Low / Normal / High |
| Cholesterol Level | ordinal | Low / Normal / High |
| Outcome Variable | target | Positive / Negative |

---

## 🔍 Why SHAP?

Most ML models are black boxes — they give a prediction but no reasoning. SHAP solves this:

- Each feature gets a **SHAP value** = its contribution to pushing the prediction above or below the average
- Red bars → pushed towards Positive outcome
- Blue bars → pushed towards Negative outcome
- The waterfall chart tells a clinician *exactly* which symptoms drove the result

This is the standard explainability technique used in healthcare ML research and production systems.

---

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**. It is not a medical device and must not be used for actual clinical decision-making. Always consult a qualified healthcare professional.

---
