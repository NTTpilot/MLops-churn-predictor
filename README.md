# Telecom Customer Churn Prediction

This project trains an XGBoost model to estimate customer churn and serves predictions through a FastAPI application.

## Live Demo

This link allows for easy usage:

https://telecom-customer-churn-prediction-rtaj.onrender.com/

## Project Overview

- FastAPI backend for prediction requests
- XGBoost model trained on the included Telco Customer Churn dataset
- MLflow tracking, model artifact logging, and model registration
- Simple frontend for quick testing

## Training and results

Run the training script from the project root:

```bash
python src/train.py
```

The script uses a stratified validation split to choose a decision threshold by F1, then evaluates once on a held-out test split. Runs are tracked in the local `src/mlflow.db` store and the trained model is registered as `ChurnXGBoost`.

An earlier recorded run on this dataset achieved about **75.2% accuracy** and **0.842 ROC-AUC**. Treat these as a reference baseline for this dataset, not a universal benchmark or guaranteed result: scores vary with the split, preprocessing, model settings, and threshold-selection method.

## Local Setup

1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train a model:

```bash
python src/train.py
```

4. Start the API:

```bash
python src/app.py
```

5. Open the app in a browser:

```text
http://localhost:8000
```

## Docker

You can also run the project with Docker:

```bash
docker-compose up --build
```

## API Endpoints

- `GET /` — app landing page
- `GET /health` — health check
- `POST /predict` — single customer prediction
- `POST /predict/batch` — batch prediction
- `GET /docs` — FastAPI Swagger UI

## Notes

The API expects customer data in the same format used during training, including categorical fields and billing information. Restart the API after training so it loads the latest model artifact.
