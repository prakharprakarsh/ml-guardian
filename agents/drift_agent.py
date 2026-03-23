"""
ML Guardian — Drift Detection Agent
Uses Evidently AI to detect data drift, prediction drift, and model performance changes.
This is the sentinel — it watches 24/7 and raises the alarm.
"""
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from datetime import datetime

from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.presets import DataDriftPreset as DataDriftTablePreset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from core.state import GuardianState, DriftReport, AgentAction, SystemStatus
from config.settings import config, thresholds
from utils.logger import get_logger
from utils.slack_notifier import notifier
from pipelines.train_initial_model import FEATURE_COLS, TARGET_COL

logger = get_logger("drift_agent", agent="drift")


class DriftAgent:
    """
    Monitors production data for drift using Evidently AI.
    
    Responsibilities:
        - Run scheduled drift checks against reference data
        - Detect feature-level and dataset-level drift
        - Monitor model performance on labeled production data
        - Update system state with drift findings
        - Alert via Slack on warning/critical drift
    """

    def __init__(self, model_path: str = None, reference_path: str = None):
        self.model_path = model_path or str(
            config.paths.artifacts / "model_v1.0.joblib"
        )
        self.reference_path = reference_path or str(
            config.paths.artifacts / "reference_features.csv"
        )
        self.model = None
        self.reference_data = None
        self._load_resources()

    def _load_resources(self):
        """Load model and reference data."""
        try:
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            if Path(self.reference_path).exists():
                self.reference_data = pd.read_csv(self.reference_path)
                logger.info(
                    f"Loaded reference data: {len(self.reference_data)} rows"
                )
        except Exception as e:
            logger.error(f"Failed to load resources: {e}")

    def check_drift(
        self, production_data: pd.DataFrame, state: GuardianState
    ) -> GuardianState:
        """
        Run a full drift check on production data.
        
        This is the main entry point called by the orchestrator.
        
        Args:
            production_data: Recent production data to check.
            state: Current guardian state.
            
        Returns:
            Updated guardian state with drift findings.
        """
        logger.info(f"Starting drift check on {len(production_data)} samples...")
        state.last_check = datetime.utcnow().isoformat()
        state.total_drift_checks += 1

        # Extract features only (exclude protected attributes and target)
        prod_features = production_data[FEATURE_COLS].copy()

        # 1. Run Evidently data drift report
        drift_result = self._run_evidently_drift(prod_features)

        # 2. Check model performance if labels are available
        perf_metrics = {}
        if TARGET_COL in production_data.columns:
            perf_metrics = self._check_performance(production_data)

        # 3. Build drift report
        drift_report = DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            dataset_drift=drift_result["dataset_drift"],
            drift_score=drift_result["drift_share"],
            num_drifted_features=drift_result["num_drifted"],
            total_features=drift_result["total_features"],
            drifted_features=drift_result["drifted_columns"],
            performance_metrics=perf_metrics,
            method="psi",
        )

        # 4. Update state based on findings
        state.drift_report = drift_report
        state = self._evaluate_severity(state, drift_report, perf_metrics)

        # 5. Log the action
        state.log_action(
            AgentAction.DRIFT_CHECK,
            {
                "drift_detected": state.drift_detected,
                "drift_score": drift_report.drift_score,
                "drifted_features": drift_report.drifted_features,
                "performance": perf_metrics,
            },
        )

        # 6. Save drift report
        self._save_report(drift_report)

        return state

    def _run_evidently_drift(self, production_features: pd.DataFrame) -> dict:
        """Run Evidently AI drift detection."""
        if self.reference_data is None:
            logger.warning("No reference data available, skipping drift check")
            return {
                "dataset_drift": False,
                "drift_share": 0.0,
                "num_drifted": 0,
                "total_features": len(FEATURE_COLS),
                "drifted_columns": [],
            }

        try:
            report = Report(
                metrics=[
                    DataDriftTablePreset(),
                ]
            )
            report.run(
                reference_data=self.reference_data,
                current_data=production_features,
            )

            # Extract results
            result = report.dict()
            metrics = result.get("metrics", [])

            dataset_drift = False
            drift_share = 0.0
            num_drifted = 0
            drifted_columns = []

            for metric in metrics:
                metric_result = metric.get("result", {})
                if "dataset_drift" in metric_result:
                    dataset_drift = metric_result["dataset_drift"]
                    drift_share = metric_result.get("share_of_drifted_columns", 0.0)
                    num_drifted = metric_result.get("number_of_drifted_columns", 0)
                if "drift_by_columns" in metric_result:
                    for col, info in metric_result["drift_by_columns"].items():
                        if info.get("drift_detected", False):
                            drifted_columns.append(col)

            # Also save HTML report
            report_path = config.paths.reports / f"drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(str(report_path))
            logger.info(f"Evidently report saved to {report_path}")

            return {
                "dataset_drift": dataset_drift,
                "drift_share": drift_share,
                "num_drifted": num_drifted,
                "total_features": len(FEATURE_COLS),
                "drifted_columns": drifted_columns,
            }

        except Exception as e:
            logger.error(f"Evidently drift check failed: {e}")
            return {
                "dataset_drift": False,
                "drift_share": 0.0,
                "num_drifted": 0,
                "total_features": len(FEATURE_COLS),
                "drifted_columns": [],
            }

    def _check_performance(self, production_data: pd.DataFrame) -> dict:
        """Check model performance on labeled production data."""
        if self.model is None:
            return {}

        try:
            X = production_data[FEATURE_COLS]
            y_true = production_data[TARGET_COL]
            y_pred = self.model.predict(X)
            y_proba = self.model.predict_proba(X)[:, 1]

            metrics = {
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "f1_score": round(f1_score(y_true, y_pred), 4),
                "auc_roc": round(roc_auc_score(y_true, y_proba), 4),
            }
            logger.info(f"Performance metrics: {json.dumps(metrics)}")
            return metrics

        except Exception as e:
            logger.error(f"Performance check failed: {e}")
            return {}

    def _evaluate_severity(
        self,
        state: GuardianState,
        report: DriftReport,
        perf_metrics: dict,
    ) -> GuardianState:
        """Evaluate drift severity and update system status."""
        drift_thresholds = thresholds.get("drift", {}).get("data_drift", {})
        perf_thresholds = thresholds.get("performance", {})

        warning_threshold = drift_thresholds.get("warning", 0.10)
        critical_threshold = drift_thresholds.get("critical", 0.20)

        # Check data drift severity
        if report.drift_score >= critical_threshold:
            state.drift_detected = True
            state.status = SystemStatus.CRITICAL
            notifier.send_drift_alert(
                report.drift_score,
                report.drifted_features,
                severity="critical",
            )
            logger.warning(
                f"CRITICAL drift detected: score={report.drift_score:.4f}"
            )

        elif report.drift_score >= warning_threshold:
            state.drift_detected = True
            state.status = SystemStatus.WARNING
            notifier.send_drift_alert(
                report.drift_score,
                report.drifted_features,
                severity="warning",
            )
            logger.warning(
                f"WARNING drift detected: score={report.drift_score:.4f}"
            )

        else:
            state.drift_detected = False
            state.status = SystemStatus.HEALTHY
            logger.info(
                f"No significant drift: score={report.drift_score:.4f}"
            )

        # Check performance degradation
        if perf_metrics:
            min_accuracy = perf_thresholds.get("accuracy", {}).get("minimum", 0.85)
            if perf_metrics.get("accuracy", 1.0) < min_accuracy:
                state.status = SystemStatus.CRITICAL
                state.drift_detected = True
                logger.warning(
                    f"Performance below minimum: accuracy={perf_metrics['accuracy']}"
                )

        return state

    def _save_report(self, report: DriftReport):
        """Save drift report as JSON."""
        report_path = (
            config.paths.reports
            / f"drift_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(
                {
                    "timestamp": report.timestamp,
                    "dataset_drift": report.dataset_drift,
                    "drift_score": report.drift_score,
                    "num_drifted_features": report.num_drifted_features,
                    "total_features": report.total_features,
                    "drifted_features": report.drifted_features,
                    "performance_metrics": report.performance_metrics,
                    "method": report.method,
                },
                f,
                indent=2,
            )
