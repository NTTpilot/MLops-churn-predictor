import os
import glob
import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union, List
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# ─────────────────────────────────────────────
# 1. FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Type-safe REST API serving the ChurnXGBoost model from MLflow",
    version="1.0.0"
)

# Enable CORS (so the frontend can call the API from anywhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML/CSS/JS) from src/static/
app.mount("/static", StaticFiles(directory="static"), name="static")

# Redirect root "/" to the landing page
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# ─────────────────────────────────────────────
# 2. Load Model Directly from Run Artifacts
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"📡 MLflow tracking URI: {MLFLOW_TRACKING_URI}")

# Find model artifacts in registry copies: mlruns/1/models/m-*/artifacts/
artifact_pattern = os.path.join(
    BASE_DIR, "mlruns", "1", "models", "m-*", "artifacts"
)
artifact_dirs = glob.glob(artifact_pattern)

if not artifact_dirs:
    raise RuntimeError(
        f"No model artifacts found.\n"
        f"Searched: {artifact_pattern}\n"
        f"Make sure you ran training and registered the model."
    )

# Pick the most recently modified one
artifact_dirs.sort(key=os.path.getmtime, reverse=True)
model_path = artifact_dirs[0]
print(f"🔧 Loading model from: {model_path}")

model = mlflow.xgboost.load_model(model_path)
print(f"✅ Model loaded successfully")

try:
    EXPECTED_COLUMNS = model.get_booster().feature_names
    print(f"📋 Expected features ({len(EXPECTED_COLUMNS)}): {EXPECTED_COLUMNS[:5]}...")
except Exception:
    raise RuntimeError("Could not read feature names from the XGBoost model.")

model = mlflow.xgboost.load_model(model_path)
print(f"✅ Model loaded successfully")

# Extract expected feature names
try:
    EXPECTED_COLUMNS = model.get_booster().feature_names
    print(f"📋 Expected features ({len(EXPECTED_COLUMNS)}): {EXPECTED_COLUMNS[:5]}...")
except Exception:
    raise RuntimeError("Could not read feature names from the XGBoost model.")

# ─────────────────────────────────────────────
# 3. Pydantic Input Schema (Raw Customer Data)
# ─────────────────────────────────────────────
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
    model_name: str = "ChurnXGBoost"
    model_version: Optional[str] = "artifact"


class BatchInput(BaseModel):
    customers: List[ChurnInput]


class BatchOutput(BaseModel):
    predictions: List[ChurnOutput]


# ─────────────────────────────────────────────
# 4. Preprocessing (Must Match train.py Exactly)
# ─────────────────────────────────────────────
def preprocess(raw_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_dict])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded


def align_columns(df_encoded: pd.DataFrame) -> pd.DataFrame:
    for col in EXPECTED_COLUMNS:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[EXPECTED_COLUMNS]
    return df_encoded


# ─────────────────────────────────────────────
# 5. Prediction Endpoints
# ─────────────────────────────────────────────
@app.post("/predict", response_model=ChurnOutput)
async def predict(input_data: ChurnInput):
    try:
        raw = input_data.model_dump()
        df_processed = preprocess(raw)
        df_aligned = align_columns(df_processed)

        proba = float(model.predict_proba(df_aligned)[:, 1][0])
        pred = int(model.predict(df_aligned)[0])

        return ChurnOutput(
            churn_probability=proba,
            churn_prediction=pred,
            churn_label="Yes" if pred == 1 else "No"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(batch: BatchInput):
    try:
        results = []
        for customer in batch.customers:
            raw = customer.model_dump()
            df_processed = preprocess(raw)
            df_aligned = align_columns(df_processed)

            proba = float(model.predict_proba(df_aligned)[:, 1][0])
            pred = int(model.predict(df_aligned)[0])

            results.append(ChurnOutput(
                churn_probability=proba,
                churn_prediction=pred,
                churn_label="Yes" if pred == 1 else "No"
            ))
        return BatchOutput(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "ChurnXGBoost",
        "features": len(EXPECTED_COLUMNS)
    }


@app.get("/")
async def root():
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "predict": "POST /predict",
        "batch": "POST /predict/batch"
    }


# ─────────────────────────────────────────────
# 6. Entrypoint
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)