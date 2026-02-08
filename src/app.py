print("🔥 LOADING src/app.py 🔥")

from fastapi import FastAPI, HTTPException
from typing import Dict

from src.schemas import HeartDiseaseRequest, ScreeningInput
from src.model_loader import load_model, predict_heart_disease

app = FastAPI(
    title="Heart Disease Prediction API",
    version="0.1.0"
)

# =========================
# Load heart disease model
# =========================
model = load_model()


# =========================
# Root & Health
# =========================
@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# Heart Disease Prediction
# =========================
@app.post("/predict")
def predict(data: HeartDiseaseRequest):
    try:
        prediction, probability = predict_heart_disease(model, data)
        return {
            "prediction": prediction,
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Generic Disease Screening
# =========================
@app.post("/screen")
def screen_diseases(data: ScreeningInput) -> Dict[str, float]:
    """
    Coarse screening layer (rule-based).
    Returns probabilities for multiple diseases.
    """

    risks = {}

    # Heart & Stroke
    if data.age >= 50:
        risks["heart"] = 0.4
        risks["stroke"] = 0.3
    else:
        risks["heart"] = 0.15
        risks["stroke"] = 0.1

    # Diabetes
    if data.glucose and data.glucose > 140:
        risks["diabetes"] = 0.6
    else:
        risks["diabetes"] = 0.2

    # Hypertension
    if data.systolic_bp and data.systolic_bp > 140:
        risks["hypertension"] = 0.7
    else:
        risks["hypertension"] = 0.25

    # CKD baseline
    risks["ckd"] = 0.15

    # Parkinsons baseline (screening only)
    risks["parkinsons"] = 0.05

    return risks
