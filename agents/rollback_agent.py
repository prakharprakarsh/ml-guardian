"""
ML Guardian — Rollback Agent
Automatically reverts to the last known good model when things go wrong.
"""
import json
import joblib
import shutil
from datetime import datetime
from pathlib import Path

from core.state import GuardianState, ModelVersion, AgentAction, SystemStatus
from config.settings import config, thresholds
from utils.logger import get_logger
from utils.slack_notifier import notifier

logger = get_logger("rollback_agent", agent="rollback")


class RollbackAgent:
    """
    Manages automatic model rollbacks.
    
    Responsibilities:
        - Maintain a history of model versions
        - Detect when rollback is needed (A/B test failure, critical degradation)
        - Execute rollback to previous champion
        - Notify stakeholders
        - Enforce cooldown periods
    """

    def __init__(self):
        self.model_history: list = []
        self.cooldown_hours = thresholds.get("rollback", {}).get(
            "cooldown_hours", 6
        )
        self.artifacts_dir = config.paths.artifacts

    def register_model(self, model: ModelVersion):
        """Add a model to the rollback history."""
        self.model_history.append({
            "version": model.version,
            "model_uri": model.model_uri,
            "metrics": model.metrics,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep last 10 versions
        if len(self.model_history) > 10:
            self.model_history = self.model_history[-10:]

    def should_rollback(self, state: GuardianState) -> bool:
        """Determine if a rollback should be executed."""
        # Check cooldown
        if state.last_rollback:
            from datetime import datetime as dt
            last = dt.fromisoformat(state.last_rollback)
            hours_since = (dt.utcnow() - last).total_seconds() / 3600
            if hours_since < self.cooldown_hours:
                logger.info(
                    f"Rollback cooldown active ({hours_since:.1f}h / "
                    f"{self.cooldown_hours}h)"
                )
                return False

        # Trigger conditions
        if state.status == SystemStatus.CRITICAL and not state.ab_test_active:
            return True

        # Check A/B test results for degradation
        if state.ab_test_results:
            decision = state.ab_test_results.get("decision", "")
            if decision == "rollback":
                return True

        return False

    def execute(self, state: GuardianState,
                reason: str = "auto") -> GuardianState:
        """
        Execute a model rollback.
        
        Args:
            state: Current guardian state.
            reason: Why the rollback is happening.
            
        Returns:
            Updated state after rollback.
        """
        if len(self.model_history) < 2:
            logger.warning("Not enough model history for rollback.")
            return state

        logger.info("Rollback agent executing...")
        state.status = SystemStatus.ROLLING_BACK

        # Get previous version
        current = self.model_history[-1]
        previous = self.model_history[-2]

        try:
            # Verify previous model exists and loads
            prev_model = joblib.load(previous["model_uri"])
            logger.info(
                f"Rolling back from {current['version']} to "
                f"{previous['version']}"
            )

            # Update champion model
            state.champion_model = ModelVersion(
                version=previous["version"],
                model_uri=previous["model_uri"],
                metrics=previous["metrics"],
                trained_at=previous["timestamp"],
                promoted_at=datetime.utcnow().isoformat(),
                is_champion=True,
            )

            # Clear challenger
            state.challenger_model = None
            state.ab_test_active = False

            # Update tracking
            state.total_rollbacks += 1
            state.last_rollback = datetime.utcnow().isoformat()
            state.status = SystemStatus.WARNING  # Stay in warning after rollback

            # Log action
            state.log_action(
                AgentAction.ROLLBACK_EXECUTED,
                {
                    "from_version": current["version"],
                    "to_version": previous["version"],
                    "reason": reason,
                    "previous_metrics": previous["metrics"],
                },
            )

            # Notify
            notifier.send_rollback_alert(
                from_version=current["version"],
                to_version=previous["version"],
                reason=reason,
            )

            logger.info(
                f"Rollback complete: now running {previous['version']}"
            )

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            state.status = SystemStatus.CRITICAL
            state.log_action(
                AgentAction.ROLLBACK_EXECUTED,
                {"error": str(e), "status": "failed"},
            )

        return state
