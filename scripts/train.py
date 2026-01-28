import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "data/Heart_Disease_Prediction.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "heart_disease_model_v1.pkl")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame):
    # Target mapping
    df["Heart Disease"] = df["Heart Disease"].map(
        {"Presence": 1, "Absence": 0}
    )

    X = df.drop(columns=["Heart Disease"])
    y = df["Heart Disease"]

    return X, y


def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\nModel Evaluation")
    print("----------------")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\nModel saved at: {path}")


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preparing data...")
    X, y = prepare_data(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate(model, X_test, y_test)

    print("Saving model...")
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
