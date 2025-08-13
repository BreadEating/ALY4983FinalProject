from pathlib import Path
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

DATA = Path("data/staged/data.csv")
MODEL = Path("artifacts/model.joblib")
TRAIN_REPORT = Path("artifacts/train_report.json")
for p in [MODEL.parent, TRAIN_REPORT.parent]:
    p.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV = 3
N_ITER = 5
GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 6, 12],
    "min_samples_split": [2, 10],
}

def main():
    df = pd.read_csv(DATA)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    search = RandomizedSearchCV(
        base, param_distributions=GRID, n_iter=N_ITER, cv=CV,
        scoring="roc_auc", n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    joblib.dump(best, MODEL)
    
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/feature_columns.json").write_text(json.dumps(list(X.columns)))

    TRAIN_REPORT.write_text(json.dumps({
        "best_params": search.best_params_,
        "cv_best_score_roc_auc": float(search.best_score_),
    }, indent=2))

    print(f"[train] saved -> {MODEL}")
    print(f"[train] report -> {TRAIN_REPORT}")

if __name__ == "__main__":
    main()