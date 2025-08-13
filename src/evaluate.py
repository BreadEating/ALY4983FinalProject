from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA = Path("data/staged/data.csv")
MODEL = Path("artifacts/model.joblib")
METRICS = Path("artifacts/metrics.json")
METRICS.parent.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42

def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = joblib.load(MODEL)
    y_pred = model.predict(X_te)
    metrics = {"accuracy": float(accuracy_score(y_te, y_pred))}
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] {metrics} -> {METRICS}")

if __name__ == "__main__":
    main()