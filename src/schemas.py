from pydantic import BaseModel
from typing import Optional


# =========================
# Heart Disease
# =========================
class HeartDiseaseRequest(BaseModel):
    Age: int
    Sex: int
    Chest_pain_type: int
    BP: int
    Cholesterol: int
    FBS_over_120: int
    EKG_results: int
    Max_HR: int
    Exercise_angina: int
    ST_depression: float
    Slope_of_ST: int
    Number_of_vessels_fluro: int
    Thallium: int


# =========================
# Diabetes
# =========================
class DiabetesRequest(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# =========================
# Generic Screening
# =========================
class ScreeningInput(BaseModel):
    age: int
    sex: Optional[int] = None
    bmi: Optional[float] = None
    systolic_bp: Optional[int] = None
    glucose: Optional[float] = None
    smoker: Optional[int] = None
