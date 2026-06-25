import os
import pandas as pd
import numpy as np
import joblib
from typing import List, Dict, Optional, Tuple
import os

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ml"
)

model_path = os.path.join(MODEL_DIR, "model.pkl")
encoders_path = os.path.join(MODEL_DIR, "encoders.pkl")
features_path = os.path.join(MODEL_DIR, "features.pkl")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

_model = None
_encoder = None
_features = None
_symptom_severity = None
_description_df = None
_medications_df = None
_diets_df = None
_precautions_df = None
_workout_df = None


def _load_model():
    global _model, _encoder, _features

    if _model is None:
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        encoder_path = os.path.join(MODEL_DIR, "encoders.pkl")
        features_path = os.path.join(MODEL_DIR, "features.pkl")

        print("MODEL PATH =", model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run train_model.py first."
            )

        _model = joblib.load(model_path)
        _encoder = joblib.load(encoder_path)
        _features = joblib.load(features_path)

    return _model, _encoder, _features


def _load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path)


def _get_severity_df():
    global _symptom_severity
    if _symptom_severity is None:
        df = _load_csv("Symptom-severity.csv")
        df.columns = df.columns.str.strip()
        _symptom_severity = df
    return _symptom_severity


def _get_description_df():
    global _description_df
    if _description_df is None:
        _description_df = _load_csv("description.csv")
        _description_df.columns = _description_df.columns.str.strip()
    return _description_df


def _get_medications_df():
    global _medications_df
    if _medications_df is None:
        _medications_df = _load_csv("medications.csv")
        _medications_df.columns = _medications_df.columns.str.strip()
    return _medications_df


def _get_diets_df():
    global _diets_df
    if _diets_df is None:
        _diets_df = _load_csv("diets.csv")
        _diets_df.columns = _diets_df.columns.str.strip()
    return _diets_df


def _get_precautions_df():
    global _precautions_df
    if _precautions_df is None:
        _precautions_df = _load_csv("precautions_df.csv")
        _precautions_df.columns = _precautions_df.columns.str.strip()

        if "Unnamed: 0" in _precautions_df.columns:
            _precautions_df = _precautions_df.drop(columns=["Unnamed: 0"])

    return _precautions_df


def _get_workout_df():
    global _workout_df
    if _workout_df is None:
        _workout_df = _load_csv("workout_df.csv")
        _workout_df.columns = _workout_df.columns.str.strip()

        if "Unnamed: 0" in _workout_df.columns:
            _workout_df = _workout_df.drop(columns=["Unnamed: 0"])

    return _workout_df


def get_all_symptoms() -> List[str]:
    _, _, features = _load_model()
    return sorted([f.replace("_", " ") for f in features])


def symptoms_to_vector(symptoms: List[str]) -> np.ndarray:
    _, _, features = _load_model()
    vector = np.zeros(len(features))
    symptom_map = {f.replace("_", " ").lower(): i for i, f in enumerate(features)}
    for symptom in symptoms:
        key = symptom.strip().lower()
        if key in symptom_map:
            vector[symptom_map[key]] = 1
    return vector


def calculate_severity(symptoms: List[str]) -> Tuple[float, str]:
    sev_df = _get_severity_df()
    print(sev_df.head())
    print(sev_df.columns)
    sev_col = sev_df.columns[0]
    weight_col = sev_df.columns[1]
    print("SEVERITY INPUT =", symptoms)
    sev_map = dict(zip(
        sev_df[sev_col].str.strip().str.lower(),
        sev_df[weight_col]
    ))
    total = 0.0
    count = 0
    for s in symptoms:
        key = s.strip().lower().replace(" ", "_")
        if key in sev_map:
            total += float(sev_map[key])
            count += 1
    avg = total / max(count, 1)
    if avg <= 2:
        label = "Low"
    elif avg <= 4:
        label = "Moderate"
    elif avg <= 6:
        label = "High"
    else:
        label = "Critical"
    return round(avg, 2), label


def calculate_health_score(symptoms: List[str]) -> float:
    severity, _ = calculate_severity(symptoms)
    n = len(symptoms)
    # Normalize: more symptoms + higher severity = lower health score
    base = 100 - (n * 3) - (severity * 5)
    return max(0.0, min(100.0, round(base, 1)))


def get_disease_info(disease: str) -> Dict:
    result: Dict[str, list] = {}

    disease = disease.strip().lower()

    # if disease == "peptic ulcer diseae":
    #    disease = "peptic ulcer disease"

    print("LOOKING FOR DISEASE =", repr(disease))
    if disease == "peptic ulcer disease":
        disease = "peptic ulcer disease"

    # ---------------- Description ----------------
    desc_df = _get_description_df()
    print("DESCRIPTION DISEASES =", desc_df["Disease"].head(20).tolist())
    desc_row = desc_df[
        desc_df["Disease"].astype(str).str.strip().str.lower() == disease
    ]
    print("DESCRIPTION DISEASES =", desc_df.iloc[:20,0].tolist())

    if len(desc_row):
        result["description"] = str(desc_row["Description"].iloc[0])
    else:
        result["description"] = "No description available."


    # ---------------- Medications ----------------
    med_df = _get_medications_df()
    print(med_df["Disease"].head(20).tolist())
    med_row = med_df[
        med_df["Disease"].astype(str).str.strip().str.lower() == disease
    ]

    if len(med_row):
        meds = str(med_row["Medication"].iloc[0])

        meds = (
            meds.replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
        )

        result["medications"] = [
            m.strip() for m in meds.split(",") if m.strip()
        ]
    else:
        result["medications"] = []

    # ---------------- Diet ----------------
    diet_df = _get_diets_df()
    print(diet_df["Disease"].head(20).tolist())
    diet_row = diet_df[
        diet_df["Disease"].astype(str).str.strip().str.lower() == disease
    ]

    if len(diet_row):
        diets = str(diet_row["Diet"].iloc[0])

        diets = (
            diets.replace("[", "")
            .replace("]", "")
            .replace("'", "")
            .replace('"', "")
        )

        result["diet"] = [
            d.strip() for d in diets.split(",") if d.strip()
        ]
    else:
        result["diet"] = []

    # ---------------- Precautions ----------------
    prec_df = _get_precautions_df()
    print(prec_df["Disease"].head(20).tolist())
    prec_row = prec_df[
        prec_df["Disease"].astype(str).str.strip().str.lower() == disease
    ]

    precautions = []

    if len(prec_row):
        for col in prec_df.columns:
            if "Precaution" in col:
                val = prec_row[col].iloc[0]

                if pd.notna(val) and str(val).strip():
                    precautions.append(str(val).strip())

    result["precautions"] = precautions

    # ---------------- Workout ----------------
    work_df = _get_workout_df()

    disease_col = "disease" if "disease" in work_df.columns else "Disease"

    work_rows = work_df[
        work_df[disease_col].astype(str).str.strip().str.lower() == disease
    ]

    workouts = []

    if len(work_rows):
        for _, row in work_rows.iterrows():
            val = row.get("workout", None)

            if pd.notna(val) and str(val).strip():
                workouts.append(str(val).strip())

    result["workout"] = workouts

    return result


def predict(symptoms: List[str]) -> Dict:
    model, encoder, features = _load_model()
    vector = symptoms_to_vector(symptoms)
    vector_2d = vector.reshape(1, -1)
    prediction_idx = model.predict(vector_2d)[0]
    probabilities = model.predict_proba(vector_2d)[0]
    confidence = float(max(probabilities))
    disease = encoder.inverse_transform([prediction_idx])[0]

    print(confidence)
    print("INPUT SYMPTOMS =", symptoms)
    print("PREDICTED DISEASE =", disease)
    print("PROBABILITIES =", probabilities)
    print("MAX PROBABILITY =", max(probabilities))

    severity_score, risk_level = calculate_severity(symptoms)
    health_score = calculate_health_score(symptoms)
    disease_info = get_disease_info(disease)

    return {
        "disease": disease,
        "confidence": round(confidence * 100, 2),
        "severity_score": severity_score,
        "risk_level": risk_level,
        "health_score": health_score,
        **disease_info,
    }
