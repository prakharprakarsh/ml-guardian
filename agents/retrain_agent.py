"""
ML Guardian — Retrain Agent
Automatically triggers model retraining when drift is detected.
"""
import json
from datetime import datetime
from pathlib import Path

from core.state import GuardianState, ModelVersion, AgentAction, SystemStatus
from pipelines.retrain_pipeline import retrain_model
from config.settings import config
from utils.logger import get_logger
from utils.slack_notifier import notifier

logger = get_logger("retrain_agent", agent="retrain")


class RetrainAgent:
    """
    Manages automated model retraining.
    
    Responsibilities:
        - Trigger retraining when drift agent flags issues
        - Use combined reference + production data for training
        - Compare new model against champion
        - Register new model as challenger for A/B testing
    """

    def __init__(self):
        self.artifacts_dir = config.paths.artifacts

    def should_retrain(self, state: GuardianState) -> bool:
        """Determine if retraining should be triggered."""
        if not state.drift_detected:
            return False
        if state.status == SystemStatus.RETRAINING:
            logger.info("Already retraining, skipping.")
            return False
        if state.ab_test_active:
            logger.info("A/B test in progress, deferring retrain.")
            return False
        return True

    def execute(self, state: GuardianState,
                production_data_path: str = None) -> GuardianState:
        """
        Execute the retraining pipeline.
        
        Args:
            state: Current guardian state.
            production_data_path: Path to recent production data.
            
        Returns:
            Updated state with new challenger model.
        """
        if not self.should_retrain(state):
            return state

        logger.info("Retrain agent triggered — starting retraining pipeline...")
        state.status = SystemStatus.RETRAINING

        # Determine data paths
        if production_data_path is None:
            production_data_path = str(config.paths.data / "production_drifted.csv")

        champion_path = None
        if state.champion_model:
            champion_path = state.champion_model.model_uri

        try:
            # Run retraining
            result = retrain_model(
                new_data_path=production_data_path,
                reference_data_path=str(config.paths.data / "reference_data.csv"),
                previous_model_path=champion_path,
                output_dir=str(self.artifacts_dir),
            )

            # Create challenger model version
            challenger = ModelVersion(
                version=result["version"],
                model_uri=result["model_path"],
                metrics=result["metrics"],
                trained_at=datetime.utcnow().isoformat(),
                is_champion=False,
            )
            state.challenger_model = challenger
            state.total_retrains += 1

            # Log action
            state.log_action(
                AgentAction.RETRAIN_COMPLETED,
                {
                    "new_version": result["version"],
                    "metrics": result["metrics"],
                    "is_improvement": result.get("is_improvement", True),
                    "comparison": result.get("comparison", {}),
                },
            )

            # Notify
            notifier.send_retrain_alert(
                old_version=state.champion_model.version if state.champion_model else "N/A",
                new_version=result["version"],
                metrics=result["metrics"],
            )

            logger.info(
                f"Retraining complete: {result['version']} — "
                f"accuracy={result['metrics']['accuracy']}"
            )

            # Transition to A/B testing
            state.status = SystemStatus.AB_TESTING

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            state.status = SystemStatus.WARNING
            state.log_action(
                AgentAction.RETRAIN_TRIGGERED,
                {"error": str(e), "status": "failed"},
            )

        return state
