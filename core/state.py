"""
ML Guardian — Shared State Management
Defines the state that flows between agents in the LangGraph orchestrator.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import json
from pathlib import Path


class SystemStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    RETRAINING = "retraining"
    AB_TESTING = "ab_testing"
    ROLLING_BACK = "rolling_back"
    INCIDENT = "incident"


class AgentAction(str, Enum):
    DRIFT_CHECK = "drift_check"
    RETRAIN_TRIGGERED = "retrain_triggered"
    RETRAIN_COMPLETED = "retrain_completed"
    AB_TEST_STARTED = "ab_test_started"
    AB_TEST_COMPLETED = "ab_test_completed"
    MODEL_PROMOTED = "model_promoted"
    ROLLBACK_EXECUTED = "rollback_executed"
    INCIDENT_REPORTED = "incident_reported"
    ALERT_SENT = "alert_sent"


@dataclass
class DriftReport:
    """Results from a drift detection run."""
    timestamp: str = ""
    dataset_drift: bool = False
    drift_score: float = 0.0
    num_drifted_features: int = 0
    total_features: int = 0
    drifted_features: list = field(default_factory=list)
    performance_metrics: dict = field(default_factory=dict)
    method: str = "psi"


@dataclass
class ModelVersion:
    """Tracks a model version in the system."""
    version: str = ""
    model_uri: str = ""
    metrics: dict = field(default_factory=dict)
    trained_at: str = ""
    promoted_at: str = ""
    is_champion: bool = False


@dataclass
class GuardianState:
    """
    Central state object shared across all agents.
    This is the state that flows through the LangGraph orchestrator.
    """
    # System status
    status: SystemStatus = SystemStatus.HEALTHY
    last_check: str = ""
    
    # Current drift report
    drift_report: Optional[DriftReport] = None
    drift_detected: bool = False
    
    # Model versions
    champion_model: Optional[ModelVersion] = None
    challenger_model: Optional[ModelVersion] = None
    
    # A/B test state
    ab_test_active: bool = False
    ab_test_results: dict = field(default_factory=dict)
    
    # Action history (audit trail)
    action_log: list = field(default_factory=list)
    
    # Incident tracking
    incident_active: bool = False
    incident_id: str = ""
    
    # Rollback state
    rollback_available: bool = False
    last_rollback: str = ""
    
    # Counters
    total_drift_checks: int = 0
    total_retrains: int = 0
    total_rollbacks: int = 0
    total_incidents: int = 0

    def log_action(self, action: AgentAction, details: dict = None):
        """Log an agent action to the audit trail."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action.value,
            "status": self.status.value,
            "details": details or {}
        }
        self.action_log.append(entry)
        self._persist_audit_log(entry)

    def _persist_audit_log(self, entry: dict):
        """Persist audit log entry to JSONL file (EU AI Act compliance)."""
        log_dir = Path("audit_logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def to_dict(self) -> dict:
        """Serialize state for dashboard display."""
        return {
            "status": self.status.value,
            "last_check": self.last_check,
            "drift_detected": self.drift_detected,
            "drift_score": self.drift_report.drift_score if self.drift_report else 0.0,
            "champion_version": self.champion_model.version if self.champion_model else "N/A",
            "ab_test_active": self.ab_test_active,
            "incident_active": self.incident_active,
            "total_drift_checks": self.total_drift_checks,
            "total_retrains": self.total_retrains,
            "total_rollbacks": self.total_rollbacks,
            "total_incidents": self.total_incidents,
            "recent_actions": self.action_log[-10:] if self.action_log else [],
        }
