import joblib
import numpy as np
from pathlib import Path

from src.schemas import HeartDiseaseRequest, DiabetesRequest

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

HEART_MODEL_PATH = MODEL_DIR / "heart_disease_model_v1.pkl"
DIABETES_MODEL_PATH = MODEL_DIR / "diabetes_model_v1.pkl"


# =========================
# Loaders
# =========================
def load_heart_model():
    return joblib.load(HEART_MODEL_PATH)


def load_diabetes_model():
    return joblib.load(DIABETES_MODEL_PATH)


# =========================
# Predictors
# =========================
def predict_heart_disease(model, data: HeartDiseaseRequest):
    features = np.array([[
        data.Age,
        data.Sex,
        data.Chest_pain_type,
        data.BP,
        data.Cholesterol,
        data.FBS_over_120,
        data.EKG_results,
        data.Max_HR,
        data.Exercise_angina,
        data.ST_depression,
        data.Slope_of_ST,
        data.Number_of_vessels_fluro,
        data.Thallium
    ]])

    prob = model.predict_proba(features)[0][1]
    prediction = "Presence" if prob >= 0.5 else "Absence"
    return prediction, round(float(prob), 3)


def predict_diabetes(model, data: DiabetesRequest):
    features = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    prob = model.predict_proba(features)[0][1]
    prediction = "Positive" if prob >= 0.5 else "Negative"
    return prediction, round(float(prob), 3)
