import os
import joblib


MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "model/heart_disease_model_v1.pkl"
)

_model = None


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model
