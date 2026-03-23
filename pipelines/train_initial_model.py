"""
ML Guardian — Initial Model Training Pipeline
Trains an XGBoost credit risk classifier and registers it with MLflow.
"""
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from pipelines.data_generator import generate_credit_data, save_datasets
from utils.logger import get_logger

logger = get_logger("train", agent="trainer")

# Feature columns (excluding protected attributes and target)
FEATURE_COLS = [
    "age", "income", "employment_years", "loan_amount", "credit_score",
    "num_credit_lines", "debt_to_income", "months_since_delinquency",
    "num_open_accounts", "revolving_utilization",
]
PROTECTED_COLS = ["gender", "age_group", "nationality"]
TARGET_COL = "default"


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and target from raw data."""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def train_model(
    data: pd.DataFrame = None,
    model_version: str = "v1.0",
    output_dir: str = "artifacts",
) -> dict:
    """
    Train an XGBoost model on credit risk data.
    
    Returns:
        Dictionary with model path, metrics, and metadata.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate data if not provided
    if data is None:
        logger.info("Generating training data...")
        save_datasets()
        data = pd.read_csv("data/reference_data.csv")

    logger.info(f"Training model {model_version} on {len(data)} samples...")

    # Prepare features
    X, y = prepare_features(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
    }

    logger.info(f"Model metrics: {json.dumps(metrics)}")

    # Save model
    model_path = output_path / f"model_{model_version}.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        "version": model_version,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "feature_columns": FEATURE_COLS,
        "n_training_samples": len(X_train),
        "n_test_samples": len(X_test),
        "model_type": "XGBClassifier",
        "hyperparameters": model.get_params(),
    }
    metadata_path = output_path / f"metadata_{model_version}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save reference data stats for drift detection
    ref_stats = X_train.describe().to_dict()
    stats_path = output_path / "reference_stats.json"
    with open(stats_path, "w") as f:
        json.dump(ref_stats, f, indent=2)

    # Save reference data for Evidently
    X_train.to_csv(output_path / "reference_features.csv", index=False)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics": metrics,
        "version": model_version,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("ML Guardian — Initial Model Training")
    print("=" * 60)
    result = train_model()
    print(f"\nTraining complete!")
    print(f"Model: {result['model_path']}")
    print(f"Metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}")
