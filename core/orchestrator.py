"""
ML Guardian — Agent Orchestrator
Uses a state machine pattern (inspired by LangGraph) to coordinate all agents.
This is the brain that decides which agent runs next based on system state.
"""
import time
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

from core.state import GuardianState, ModelVersion, AgentAction, SystemStatus
from agents.drift_agent import DriftAgent
from agents.retrain_agent import RetrainAgent
from agents.ab_test_agent import ABTestAgent
from agents.incident_agent import IncidentAgent
from agents.rollback_agent import RollbackAgent
from config.settings import config
from pipelines.train_initial_model import train_model
from pipelines.data_generator import save_datasets
from utils.logger import get_logger
from utils.slack_notifier import notifier

logger = get_logger("orchestrator", agent="orchestrator")


class GuardianOrchestrator:
    """
    Coordinates the ML Guardian agent swarm using a state machine.
    
    State Machine Flow:
    
    IDLE → DRIFT_CHECK → [no drift] → IDLE
                       → [drift warning] → MONITOR
                       → [drift critical] → RETRAIN → AB_TEST → [promote/keep] → IDLE
                                                              → [rollback] → ROLLBACK → IDLE
                       → [perf drop] → INCIDENT_REPORT → RETRAIN → ...
    """

    def __init__(self):
        self.state = GuardianState()
        self.drift_agent = DriftAgent()
        self.retrain_agent = RetrainAgent()
        self.ab_test_agent = ABTestAgent()
        self.incident_agent = IncidentAgent()
        self.rollback_agent = RollbackAgent()
        self._initialized = False

    def initialize(self):
        """Initialize the system: ensure model and data exist."""
        logger.info("=" * 60)
        logger.info("ML Guardian — Initializing...")
        logger.info("=" * 60)

        # Ensure data exists
        data_dir = config.paths.data
        if not (data_dir / "reference_data.csv").exists():
            logger.info("Generating initial datasets...")
            save_datasets(str(data_dir))

        # Ensure model exists
        model_path = config.paths.artifacts / "model_v1.0.joblib"
        if not model_path.exists():
            logger.info("Training initial model...")
            result = train_model(output_dir=str(config.paths.artifacts))
            initial_model = ModelVersion(
                version="v1.0",
                model_uri=result["model_path"],
                metrics=result["metrics"],
                trained_at=datetime.utcnow().isoformat(),
                promoted_at=datetime.utcnow().isoformat(),
                is_champion=True,
            )
        else:
            # Load existing metadata
            meta_path = config.paths.artifacts / "metadata_v1.0.json"
            metrics = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
                    metrics = metadata.get("metrics", {})
            initial_model = ModelVersion(
                version="v1.0",
                model_uri=str(model_path),
                metrics=metrics,
                trained_at=datetime.utcnow().isoformat(),
                promoted_at=datetime.utcnow().isoformat(),
                is_champion=True,
            )

        self.state.champion_model = initial_model
        self.rollback_agent.register_model(initial_model)

        # Re-init drift agent with correct paths
        self.drift_agent = DriftAgent(
            model_path=str(model_path),
            reference_path=str(config.paths.artifacts / "reference_features.csv"),
        )

        self._initialized = True
        logger.info("Initialization complete.")
        logger.info(f"Champion model: {initial_model.version}")
        logger.info(f"Metrics: {json.dumps(initial_model.metrics)}")

        notifier.send(
            title="ML Guardian Started",
            message=f"System initialized with model {initial_model.version}",
            severity="success",
            fields=initial_model.metrics,
        )

    def run_cycle(self, production_data: pd.DataFrame = None) -> dict:
        """
        Run one complete monitoring cycle.
        
        This is the main loop iteration:
        1. Check for drift
        2. If drift → retrain
        3. If retrained → A/B test
        4. If critical → incident report
        5. If degradation → rollback
        
        Returns:
            Current system state as dictionary.
        """
        if not self._initialized:
            self.initialize()

        logger.info(f"\n{'='*60}")
        logger.info(f"Cycle starting at {datetime.utcnow().isoformat()}")
        logger.info(f"Current status: {self.state.status.value}")
        logger.info(f"{'='*60}")

        # Load production data if not provided
        if production_data is None:
            prod_path = config.paths.data / "production_drifted.csv"
            if prod_path.exists():
                production_data = pd.read_csv(str(prod_path))
            else:
                logger.warning("No production data available.")
                return self.state.to_dict()

        # === STEP 1: Drift Detection ===
        logger.info("Step 1: Running drift detection...")
        self.state = self.drift_agent.check_drift(production_data, self.state)

        # === STEP 2: Incident Report (if critical) ===
        if self.incident_agent.should_report(self.state):
            logger.info("Step 2: Generating incident report...")
            self.state = self.incident_agent.execute(self.state)

        # === STEP 3: Retrain (if drift detected) ===
        if self.retrain_agent.should_retrain(self.state):
            logger.info("Step 3: Triggering retraining...")
            self.state = self.retrain_agent.execute(
                self.state,
                production_data_path=str(config.paths.data / "production_drifted.csv"),
            )

        # === STEP 4: A/B Test (if challenger available) ===
        if self.state.challenger_model and not self.state.ab_test_active:
            logger.info("Step 4: Running A/B test...")
            test_data = production_data.sample(
                n=min(config.ab_test.sample_size, len(production_data)),
                random_state=42,
            )
            self.state = self.ab_test_agent.execute(self.state, test_data)

            # Register new champion if promoted
            if self.state.champion_model:
                self.rollback_agent.register_model(self.state.champion_model)

        # === STEP 5: Rollback check ===
        if self.rollback_agent.should_rollback(self.state):
            logger.info("Step 5: Executing rollback...")
            self.state = self.rollback_agent.execute(
                self.state, reason="auto_critical_drift"
            )

        # === STEP 6: Resolve incident if system recovered ===
        if (
            self.state.incident_active
            and self.state.status in (SystemStatus.HEALTHY, SystemStatus.WARNING)
        ):
            self.state = self.incident_agent.resolve_incident(self.state)

        logger.info(f"Cycle complete. Status: {self.state.status.value}")
        return self.state.to_dict()

    def run_continuous(self, interval_minutes: int = None):
        """Run the guardian in continuous monitoring mode."""
        interval = interval_minutes or config.drift.check_interval_minutes
        logger.info(f"Starting continuous monitoring (interval: {interval}min)")

        while True:
            try:
                result = self.run_cycle()
                logger.info(f"Status: {result['status']}")
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                notifier.send(
                    title="Guardian Cycle Error",
                    message=str(e),
                    severity="critical",
                )

            logger.info(f"Sleeping for {interval} minutes...")
            time.sleep(interval * 60)

    def get_status(self) -> dict:
        """Get current system status for dashboard."""
        return self.state.to_dict()


def main():
    """Entry point for running the ML Guardian."""
    print("=" * 60)
    print("  ML Guardian — Agentic ML Model Lifecycle Guardian")
    print("  An SRE for Machine Learning, Fully Autonomous")
    print("=" * 60)

    guardian = GuardianOrchestrator()
    guardian.initialize()

    print("\nRunning a single monitoring cycle with drifted data...")
    print("-" * 60)

    result = guardian.run_cycle()

    print("\n" + "=" * 60)
    print("CYCLE RESULTS:")
    print("=" * 60)
    print(json.dumps(result, indent=2))

    print("\nTo run continuous monitoring:")
    print("  guardian.run_continuous(interval_minutes=30)")


if __name__ == "__main__":
    main()
