"""
ML Guardian — Automated Retraining Pipeline
Triggered by the Retrain Agent when drift is detected.
"""
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from pipelines.train_initial_model import FEATURE_COLS, TARGET_COL, prepare_features
from utils.logger import get_logger

logger = get_logger("retrain", agent="retrain_pipeline")


def retrain_model(
    new_data_path: str,
    reference_data_path: str = "data/reference_data.csv",
    previous_model_path: str = None,
    output_dir: str = "artifacts",
) -> dict:
    """
    Retrain the model on combined reference + new production data.
    
    Strategy: Combine reference data with recent production data,
    giving more weight to recent observations.
    
    Returns:
        Dictionary with new model path, metrics, and comparison.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    ref_data = pd.read_csv(reference_data_path)
    new_data = pd.read_csv(new_data_path)
    
    logger.info(f"Reference data: {len(ref_data)} samples")
    logger.info(f"New production data: {len(new_data)} samples")

    # Combine with recency weighting
    # Use 70% new data + 30% reference to adapt to new distribution
    ref_sample = ref_data.sample(
        n=min(int(len(new_data) * 0.4), len(ref_data)),
        random_state=42,
    )
    combined = pd.concat([ref_sample, new_data], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"Combined training set: {len(combined)} samples")

    # Prepare features
    X, y = prepare_features(combined)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Train new model
    new_model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    new_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate new model
    y_pred = new_model.predict(X_test)
    y_proba = new_model.predict_proba(X_test)[:, 1]

    new_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
    }

    # Compare with previous model if available
    comparison = {}
    if previous_model_path and Path(previous_model_path).exists():
        old_model = joblib.load(previous_model_path)
        old_pred = old_model.predict(X_test)
        old_proba = old_model.predict_proba(X_test)[:, 1]
        old_metrics = {
            "accuracy": round(accuracy_score(y_test, old_pred), 4),
            "f1_score": round(f1_score(y_test, old_pred), 4),
            "auc_roc": round(roc_auc_score(y_test, old_proba), 4),
        }
        comparison = {
            "old_metrics": old_metrics,
            "improvement": {
                k: round(new_metrics[k] - old_metrics[k], 4)
                for k in new_metrics
            },
        }
        logger.info(f"Old model metrics: {json.dumps(old_metrics)}")
        logger.info(f"Improvement: {json.dumps(comparison['improvement'])}")

    # Generate version number
    version = f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Save new model
    model_path = output_path / f"model_{version}.joblib"
    joblib.dump(new_model, model_path)

    # Save metadata
    metadata = {
        "version": version,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": new_metrics,
        "comparison": comparison,
        "training_data_size": len(combined),
        "retrain_reason": "drift_detected",
        "feature_columns": FEATURE_COLS,
    }
    metadata_path = output_path / f"metadata_{version}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Update reference features for future drift checks
    X_train.to_csv(output_path / "reference_features.csv", index=False)

    logger.info(f"Retrained model saved: {model_path}")
    logger.info(f"New metrics: {json.dumps(new_metrics)}")

    return {
        "model_path": str(model_path),
        "version": version,
        "metrics": new_metrics,
        "comparison": comparison,
        "is_improvement": all(
            v >= 0 for v in comparison.get("improvement", {}).values()
        ) if comparison else True,
    }
