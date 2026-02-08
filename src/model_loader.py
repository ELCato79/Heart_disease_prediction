print("🔥 LOADING src/model_loader.py 🔥")

import joblib
import numpy as np
from pathlib import Path

from src.schemas import HeartDiseaseRequest

# =========================
# Model paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

HEART_MODEL_PATH = MODEL_DIR / "heart_disease_model_v1.pkl"


# =========================
# Loaders
# =========================
def load_model():
    if not HEART_MODEL_PATH.exists():
        raise FileNotFoundError(f"Heart model not found at {HEART_MODEL_PATH}")

    return joblib.load(HEART_MODEL_PATH)


# =========================
# Predictors
# =========================
def predict_heart_disease(model, data: HeartDiseaseRequest):
    """
    Predict heart disease using trained model.
    """

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


print("🔥 predict_heart_disease exists:", "predict_heart_disease" in globals())
