from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import joblib, json, pandas as pd

MODEL_PATH = Path("artifacts/model.joblib")
FEATS_PATH = Path("artifacts/feature_columns.json")

app = FastAPI()

# Load model + feature columns
model = joblib.load(MODEL_PATH)
feature_columns = []
if FEATS_PATH.exists():
    try:
        feature_columns = json.loads(FEATS_PATH.read_text())
    except Exception:
        feature_columns = []

if not feature_columns:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        feature_columns = [str(c) for c in list(names)]

if not feature_columns:
    # Fail early with a clear message rather than 500 later
    raise RuntimeError("No feature column list available. Rebuild after running the training stage.")

@app.get("/ping")
def ping():
    # SageMaker health-check expects 200 if ready
    return PlainTextResponse("OK", status_code=200)

@app.post("/invocations")
async def invocations(request: Request):
    payload = await request.json()
    # Accept {"instances":[{...}, {...}]} or {"features": {...}}
    items = payload.get("instances") or [payload.get("features", {})]
    if isinstance(items, dict):
        items = [items]
    rows = [{c: (row.get(c, 0)) for c in feature_columns} for row in items]
    X = pd.DataFrame(rows, columns=feature_columns)
    preds = model.predict(X)
    # Ensure plain Python ints (avoid numpy types in JSON)
    preds_py = [int(x) for x in preds.tolist()]
    return JSONResponse({"predictions": preds_py})