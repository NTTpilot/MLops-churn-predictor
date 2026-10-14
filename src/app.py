import glob
import os
from typing import List, Literal, Optional, Union

import mlflow
import mlflow.xgboost
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from mlflow.models import Model
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# FastAPI app setup
app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Type-safe REST API serving the ChurnXGBoost model from MLflow",
    version="1.0.0"
)

# Allow the frontend to call the API from other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend from src/static (path is relative to this file, not the cwd)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Send the browser to the main page when it hits the root route
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")


# ---------- model loading ----------
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}",
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"📡 MLflow tracking URI: {MLFLOW_TRACKING_URI}")

# Find logged model artifacts for experiment 1.
artifact_pattern = os.path.join(BASE_DIR, "mlruns", "1", "models", "m-*", "artifacts")
artifact_dirs = glob.glob(artifact_pattern)

if not artifact_dirs:
    raise RuntimeError(
        f"No model artifacts found.\n"
        f"Searched: {artifact_pattern}\n"
        f"Make sure you ran training and registered the model."
    )

# Load the most recently modified matching artifact.
artifact_dirs.sort(key=os.path.getmtime, reverse=True)
model_path = artifact_dirs[0]
print(f"🔧 Loading model from: {model_path}")

model = mlflow.xgboost.load_model(model_path)
print("✅ Model loaded successfully")

EXPECTED_COLUMNS = model.get_booster().feature_names
if not EXPECTED_COLUMNS:
    raise RuntimeError("Could not read feature names from the XGBoost model.")
print(f"📋 Expected features ({len(EXPECTED_COLUMNS)}): {EXPECTED_COLUMNS[:5]}...")


def load_threshold(path: str):
    """Decision threshold chosen at training time, stored in the model's MLmodel metadata."""
    try:
        value = (Model.load(path).metadata or {}).get("threshold")
        if value is not None and 0.0 < float(value) < 1.0:
            return float(value), "model metadata"
    except Exception as e:
        print(f"⚠️  Could not read model metadata: {e}")
    print("⚠️  No threshold stored with this model, falling back to 0.5. "
          "Retrain with the updated train.py to store one.")
    return 0.5, "default"


THRESHOLD, THRESHOLD_SOURCE = load_threshold(model_path)
print(f"🎯 Decision threshold: {THRESHOLD:.3f} ({THRESHOLD_SOURCE})")


# ---------- schemas ----------
class ChurnInput(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., example="Female")
    SeniorCitizen: Literal[0, 1] = Field(..., example=0)
    Partner: Literal["Yes", "No"] = Field(..., example="Yes")
    Dependents: Literal["Yes", "No"] = Field(..., example="No")
    tenure: int = Field(..., ge=0, le=100, example=12)
    PhoneService: Literal["Yes", "No"] = Field(..., example="Yes")
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(..., example="No")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., example="DSL")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., example="Yes")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., example="No")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., example="Month-to-month")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., example="Yes")
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=29.85)
    TotalCharges: Union[str, float] = Field(..., example="29.85")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": "29.85"
            }
        }


class ChurnOutput(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: int = Field(..., ge=0, le=1)
    churn_label: Literal["Yes", "No"]
    threshold: float
    model_name: str = "ChurnXGBoost"
    model_version: Optional[str] = "artifact"


class BatchInput(BaseModel):
    customers: List[ChurnInput]


class BatchOutput(BaseModel):
    predictions: List[ChurnOutput]


# ---------- preprocessing ----------
NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def encode(rows: List[dict]) -> pd.DataFrame:
    """Rebuild the exact training matrix.

    Training used get_dummies(drop_first=True) over the whole dataset. Doing that on
    the request itself is wrong: with one row, every categorical has a single level,
    so drop_first removes it and every dummy comes out 0. Instead we build the FULL
    dummies and keep only the columns the model was trained on, which drops the same
    baseline levels as training did.
    """
    df = pd.DataFrame(rows)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    categorical = [c for c in df.columns if c not in NUMERIC_COLUMNS]
    dummy_columns = [c for c in EXPECTED_COLUMNS if c not in NUMERIC_COLUMNS]
    dummies = pd.get_dummies(df[categorical]).reindex(columns=dummy_columns, fill_value=False)

    return pd.concat([df[NUMERIC_COLUMNS], dummies], axis=1)[EXPECTED_COLUMNS]


def score(rows: List[dict]) -> List[ChurnOutput]:
    probas = model.predict_proba(encode(rows))[:, 1]
    results = []
    for p in probas:
        churn = int(p >= THRESHOLD)
        results.append(ChurnOutput(
            churn_probability=float(p),
            churn_prediction=churn,
            churn_label="Yes" if churn else "No",
            threshold=THRESHOLD,
        ))
    return results


# ---------- endpoints ----------
@app.post("/predict", response_model=ChurnOutput)
def predict(input_data: ChurnInput):
    try:
        return score([input_data.model_dump()])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchOutput)
def predict_batch(batch: BatchInput):
    try:
        if not batch.customers:
            return BatchOutput(predictions=[])
        return BatchOutput(predictions=score([c.model_dump() for c in batch.customers]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "ChurnXGBoost",
        "features": len(EXPECTED_COLUMNS),
        "threshold": THRESHOLD,
        "threshold_source": THRESHOLD_SOURCE,
    }


# Start the app when this file is run directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)