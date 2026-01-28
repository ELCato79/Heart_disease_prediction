from fastapi import FastAPI

app = FastAPI(title="Heart Disease Prediction API")

@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}
