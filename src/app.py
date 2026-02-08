from fastapi import FastAPI, HTTPException
from typing import Dict

from src.schemas import (
    HeartDiseaseRequest,
    DiabetesRequest,
    ScreeningInput
)
from src.model_loader import (
    load_heart_model,
    load_diabetes_model,
    predict_heart_disease,
    predict_diabetes
)

app = FastAPI(
    title="Multi-Disease Prediction API",
    version="0.2.0"
)

# =========================
# Load models
# =========================
heart_model = load_heart_model()
diabetes_model = load_diabetes_model()


# =========================
# Root & Health
# =========================
@app.get("/")
def root():
    return {"message": "Multi-Disease Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# Heart Prediction
# =========================
@app.post("/predict")
def predict_heart(data: HeartDiseaseRequest):
    try:
        pred, prob = predict_heart_disease(heart_model, data)
        return {"prediction": pred, "probability": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Diabetes Prediction
# =========================
@app.post("/predict/diabetes")
def predict_diabetes_api(data: DiabetesRequest):
    try:
        pred, prob = predict_diabetes(diabetes_model, data)
        return {"prediction": pred, "probability": prob}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Screening
# =========================
@app.post("/screen")
def screen_diseases(data: ScreeningInput) -> Dict[str, float]:
    risks = {}

    risks["heart"] = 0.4 if data.age >= 50 else 0.15
    risks["diabetes"] = 0.6 if data.glucose and data.glucose > 140 else 0.2
    risks["hypertension"] = 0.7 if data.systolic_bp and data.systolic_bp > 140 else 0.25
    risks["stroke"] = 0.3 if data.age >= 55 else 0.1
    risks["ckd"] = 0.15
    risks["parkinsons"] = 0.05

    return risks
