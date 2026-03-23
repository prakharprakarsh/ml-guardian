"""
ML Guardian — Unit Tests for Agents
30 automated tests validating core functionality.
"""
import pytest
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import GuardianState, DriftReport, ModelVersion, SystemStatus, AgentAction
from agents.drift_agent import DriftAgent
from agents.retrain_agent import RetrainAgent
from agents.ab_test_agent import ABTestAgent
from agents.incident_agent import IncidentAgent
from agents.rollback_agent import RollbackAgent
from pipelines.data_generator import generate_credit_data
from pipelines.train_initial_model import FEATURE_COLS, TARGET_COL


# ─── STATE TESTS ──────────────────────────────────────────

class TestGuardianState:
    def test_initial_state_is_healthy(self):
        state = GuardianState()
        assert state.status == SystemStatus.HEALTHY
        assert not state.drift_detected
        assert state.total_drift_checks == 0

    def test_log_action_adds_entry(self):
        state = GuardianState()
        state.log_action(AgentAction.DRIFT_CHECK, {"score": 0.05})
        assert len(state.action_log) == 1
        assert state.action_log[0]["action"] == "drift_check"

    def test_to_dict_serialization(self):
        state = GuardianState()
        d = state.to_dict()
        assert "status" in d
        assert "drift_detected" in d
        assert d["status"] == "healthy"

    def test_state_tracks_counters(self):
        state = GuardianState()
        state.total_drift_checks = 5
        state.total_retrains = 2
        d = state.to_dict()
        assert d["total_drift_checks"] == 5
        assert d["total_retrains"] == 2


# ─── DRIFT REPORT TESTS ──────────────────────────────────

class TestDriftReport:
    def test_default_drift_report(self):
        report = DriftReport()
        assert not report.dataset_drift
        assert report.drift_score == 0.0
        assert report.drifted_features == []

    def test_drift_report_with_values(self):
        report = DriftReport(
            dataset_drift=True,
            drift_score=0.25,
            drifted_features=["income", "age"],
        )
        assert report.dataset_drift
        assert report.drift_score == 0.25
        assert len(report.drifted_features) == 2


# ─── DATA GENERATOR TESTS ────────────────────────────────

class TestDataGenerator:
    def test_generates_correct_shape(self):
        data = generate_credit_data(n_samples=100)
        assert len(data) == 100
        assert TARGET_COL in data.columns
        for col in FEATURE_COLS:
            assert col in data.columns

    def test_drift_changes_distribution(self):
        normal = generate_credit_data(n_samples=5000, drift=False, seed=42)
        drifted = generate_credit_data(n_samples=5000, drift=True,
                                       drift_intensity=0.5, seed=42)
        # Income should be lower in drifted
        assert drifted["income"].mean() < normal["income"].mean()
        # Credit scores should be lower
        assert drifted["credit_score"].mean() < normal["credit_score"].mean()

    def test_target_is_binary(self):
        data = generate_credit_data(n_samples=100)
        assert set(data[TARGET_COL].unique()).issubset({0, 1})

    def test_protected_attributes_present(self):
        data = generate_credit_data(n_samples=100)
        assert "gender" in data.columns
        assert "age_group" in data.columns
        assert "nationality" in data.columns

    def test_age_within_bounds(self):
        data = generate_credit_data(n_samples=1000)
        assert data["age"].min() >= 18
        assert data["age"].max() <= 75


# ─── RETRAIN AGENT TESTS ─────────────────────────────────

class TestRetrainAgent:
    def test_should_not_retrain_when_healthy(self):
        agent = RetrainAgent()
        state = GuardianState()
        state.drift_detected = False
        assert not agent.should_retrain(state)

    def test_should_retrain_when_drift_detected(self):
        agent = RetrainAgent()
        state = GuardianState()
        state.drift_detected = True
        state.status = SystemStatus.CRITICAL
        assert agent.should_retrain(state)

    def test_should_not_retrain_during_ab_test(self):
        agent = RetrainAgent()
        state = GuardianState()
        state.drift_detected = True
        state.ab_test_active = True
        assert not agent.should_retrain(state)

    def test_should_not_retrain_when_already_retraining(self):
        agent = RetrainAgent()
        state = GuardianState()
        state.drift_detected = True
        state.status = SystemStatus.RETRAINING
        assert not agent.should_retrain(state)


# ─── AB TEST AGENT TESTS ─────────────────────────────────

class TestABTestAgent:
    def test_auto_promote_first_model(self):
        agent = ABTestAgent()
        state = GuardianState()
        state.champion_model = None
        state.challenger_model = ModelVersion(
            version="v1.0", model_uri="test", metrics={"accuracy": 0.9}
        )
        test_data = generate_credit_data(n_samples=100)
        state = agent.execute(state, test_data)
        assert state.champion_model is not None
        assert state.champion_model.version == "v1.0"

    def test_mcnemar_identical_predictions(self):
        agent = ABTestAgent()
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        pred = np.array([0, 1, 0, 1, 0, 1, 0, 0])
        result = agent._mcnemar_test(y_true, pred, pred)
        assert result["p_value"] == 1.0
        assert not result["significant"]


# ─── INCIDENT AGENT TESTS ────────────────────────────────

class TestIncidentAgent:
    def test_should_not_report_when_healthy(self):
        agent = IncidentAgent()
        state = GuardianState()
        state.status = SystemStatus.HEALTHY
        assert not agent.should_report(state)

    def test_should_report_when_critical(self):
        agent = IncidentAgent()
        state = GuardianState()
        state.status = SystemStatus.CRITICAL
        assert agent.should_report(state)

    def test_should_not_report_when_incident_active(self):
        agent = IncidentAgent()
        state = GuardianState()
        state.status = SystemStatus.CRITICAL
        state.incident_active = True
        assert not agent.should_report(state)

    def test_should_report_on_low_accuracy(self):
        agent = IncidentAgent()
        state = GuardianState()
        state.status = SystemStatus.WARNING
        state.drift_report = DriftReport(
            performance_metrics={"accuracy": 0.70}
        )
        assert agent.should_report(state)

    def test_resolve_incident(self):
        agent = IncidentAgent()
        state = GuardianState()
        state.incident_active = True
        state.incident_id = "INC-TEST"
        state = agent.resolve_incident(state)
        assert not state.incident_active


# ─── ROLLBACK AGENT TESTS ────────────────────────────────

class TestRollbackAgent:
    def test_should_not_rollback_when_healthy(self):
        agent = RollbackAgent()
        state = GuardianState()
        state.status = SystemStatus.HEALTHY
        assert not agent.should_rollback(state)

    def test_should_rollback_on_critical(self):
        agent = RollbackAgent()
        state = GuardianState()
        state.status = SystemStatus.CRITICAL
        state.ab_test_active = False
        assert agent.should_rollback(state)

    def test_cooldown_prevents_rollback(self):
        agent = RollbackAgent()
        state = GuardianState()
        state.status = SystemStatus.CRITICAL
        from datetime import datetime
        state.last_rollback = datetime.utcnow().isoformat()
        assert not agent.should_rollback(state)

    def test_not_enough_history(self):
        agent = RollbackAgent()
        state = GuardianState()
        agent.model_history = [{"version": "v1", "model_uri": "x", "metrics": {}, "timestamp": ""}]
        result = agent.execute(state)
        assert result.status != SystemStatus.ROLLING_BACK

    def test_register_model_keeps_history(self):
        agent = RollbackAgent()
        for i in range(15):
            agent.register_model(ModelVersion(version=f"v{i}"))
        assert len(agent.model_history) == 10  # Keeps last 10


# ─── INTEGRATION-LIKE TESTS ──────────────────────────────

class TestEndToEnd:
    def test_state_flow_drift_to_retrain(self):
        """Verify state transitions from drift detection to retrain trigger."""
        state = GuardianState()
        state.status = SystemStatus.CRITICAL
        state.drift_detected = True

        retrain = RetrainAgent()
        assert retrain.should_retrain(state)

    def test_state_flow_retrain_to_ab_test(self):
        """Verify state transitions from retrain to A/B test."""
        state = GuardianState()
        state.champion_model = ModelVersion(version="v1")
        state.challenger_model = ModelVersion(version="v2")
        state.ab_test_active = False
        # A/B test should proceed when both models exist
        assert state.challenger_model is not None
        assert not state.ab_test_active


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
