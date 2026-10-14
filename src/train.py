import argparse
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "ChurnData.csv"
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'src' / 'mlflow.db').resolve()}"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a churn prediction model with XGBoost.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the customer churn dataset CSV.",
    )
    parser.add_argument(
        "--experiment-name",
        default="telecom_churn_prediction",
        help="MLflow experiment name to store the training run under.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI. Defaults to the SQLite database shared with the API.",
    )
    parser.add_argument(
        "--model-name",
        default="ChurnXGBoost",
        help="Registered MLflow model name to use for the trained model.",
    )
    return parser.parse_args()


def load_and_prepare_data(data_path: Path):
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X = pd.get_dummies(X, drop_first=True)

    # Keep the test split out of threshold selection.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Use a validation split to choose the decision threshold.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    print(f"Data successfully prepared.")
    print(f"Training features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")
    print(f"Testing features shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model_params(y_train):
    negative_class_count = y_train.value_counts().get(0, 0)
    positive_class_count = y_train.value_counts().get(1, 0)

    if positive_class_count == 0:
        raise ValueError("Training target has no positive examples. Cannot train the model.")

    scale_weight = negative_class_count / positive_class_count
    params = {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 150,
        "scale_pos_weight": scale_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "logloss",
    }
    return params


def main():
    args = parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(args.data_path)
    params = build_model_params(y_train)

    with mlflow.start_run(run_name="churn_model_training"):
        mlflow.log_params(params)

        print("Training XGBoost Classifier...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Choose the threshold that gives the best validation F1.
        print("Finding optimal decision threshold on validation set...")
        val_probs = model.predict_proba(X_val)[:, 1]

        thresholds = np.linspace(0.01, 0.99, 99)
        best_threshold = 0.5
        best_f1 = -1.0

        for t in thresholds:
            val_preds_t = (val_probs >= t).astype(int)
            score = f1_score(y_val, val_preds_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        print(f"Optimal threshold found: {best_threshold:.4f} (Validation F1: {best_f1:.4f})")
        mlflow.log_param("best_threshold", best_threshold)
        mlflow.log_metric("val_f1", best_f1)

        # Evaluate once on the held-out test split.
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= best_threshold).astype(int)

        acc = accuracy_score(y_test, test_preds)
        roc_auc = roc_auc_score(y_test, test_probs)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)

        # Store the selected threshold with the registered model.
        mlflow.xgboost.log_model(
            model,
            "churn_xgboost_model",
            registered_model_name=args.model_name,
            metadata={"threshold": str(best_threshold)},
        )

        print("Training complete!")
        print(f"Test Accuracy: {acc:.4f} | Test ROC-AUC: {roc_auc:.4f} | Threshold: {best_threshold:.4f}")

    print("Classification Report (Test Set):")
    print(classification_report(y_test, test_preds))


if __name__ == "__main__":
    main()