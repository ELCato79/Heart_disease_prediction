from pydantic import BaseModel
from typing import Optional


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




class ScreeningInput(BaseModel):
    age: int
    sex: Optional[int] = None        # 0=female, 1=male
    bmi: Optional[float] = None
    systolic_bp: Optional[int] = None
    glucose: Optional[float] = None
    smoker: Optional[int] = None     # 0/1

