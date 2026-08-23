import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import os


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("telecom_churn_prediction")

    # Path to data from the 'src' directory
    data_path = "../data/ChurnData.csv"
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Drop unique identifier
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # 2. Clean TotalCharges (convert whitespace/blank values to numeric float)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # 3. Encode Target Variable ('Yes' -> 1, 'No' -> 0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # 4. Separate Features and Target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # 5. One-Hot Encode Categorical Variables
    X = pd.get_dummies(X, drop_first=True)

    # 6. Train/Test Split (Stratified due to churn imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Data successfully prepared.")
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")

    # Set the MLflow experiment name
    mlflow.set_experiment("telecom_churn_prediction")
    scale_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    # Define hyperparameters
    params = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "scale_pos_weight": scale_weight,  # Penalizes missing Class 1
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss"
    }

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(params)
        
        print("Training XGBoost Classifier...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predict probabilities and class labels
        preds_proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, preds_proba)
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log trained model artifact
        mlflow.xgboost.log_model(model, "churn_xgboost_model")
        
        print(f"Training complete!")
        print(f"Accuracy: {acc:.4f} | ROC-AUC: {roc_auc:.4f}")

    # Print detailed classification metrics
    print("Classification Report:")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()
