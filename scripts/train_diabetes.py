import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "diabetes.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "diabetes_model_v1.pkl"

# =========================
# Load data
# =========================
df = pd.read_csv(DATA_PATH)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =========================
# Train
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

# =========================
# Save model
# =========================
joblib.dump(pipeline, MODEL_PATH)

print(f"✅ Diabetes model saved at {MODEL_PATH}")
