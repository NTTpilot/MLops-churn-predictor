# Telecom Customer Churn Prediction

This project predicts whether a telecom customer is likely to churn using a trained machine learning model exposed through a FastAPI application.

## Live Demo

This link allows for easy usage:

https://telecom-customer-churn-prediction-rtaj.onrender.com/

## Project Overview

- FastAPI backend for prediction requests
- XGBoost model trained on churn data
- MLflow model tracking and artifact loading
- Simple frontend for quick testing

## Local Setup

1. Create and activate a virtual environment
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python src/app.py
```

4. Open the app in a browser:

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

The model expects customer data in the same format used during training, including categorical fields and billing information.
