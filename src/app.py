from fastapi import FastAPI, HTTPException
import pandas as pd

from src.model_loader import load_model
from src.schemas import HeartDiseaseRequest

app = FastAPI(title="Heart Disease Prediction API")


# ---------- Health & Root ----------
@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Column Mapping ----------
COLUMN_MAPPING = {
    "Age": "Age",
    "Sex": "Sex",
    "Chest_pain_type": "Chest pain type",
    "BP": "BP",
    "Cholesterol": "Cholesterol",
    "FBS_over_120": "FBS over 120",
    "EKG_results": "EKG results",
    "Max_HR": "Max HR",
    "Exercise_angina": "Exercise angina",
    "ST_depression": "ST depression",
    "Slope_of_ST": "Slope of ST",
    "Number_of_vessels_fluro": "Number of vessels fluro",
    "Thallium": "Thallium",
}

FEATURE_ORDER = list(COLUMN_MAPPING.values())


# ---------- Prediction Endpoint ----------
@app.post("/predict")
def predict(request: HeartDiseaseRequest):
    try:
        model = load_model()

        # Convert request to dict
        data = request.dict()

        # Map API fields to training column names
        mapped_data = {
            COLUMN_MAPPING[key]: value for key, value in data.items()
        }

        # Create DataFrame with correct column order
        df = pd.DataFrame([mapped_data], columns=FEATURE_ORDER)

        # Predict
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        return {
            "prediction": "Presence" if prediction == 1 else "Absence",
            "probability": round(probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
