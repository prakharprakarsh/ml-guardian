"""
ML Guardian — Central Configuration
Loads settings from environment variables and thresholds from YAML.
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


@dataclass
class MLflowConfig:
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "ml-guardian")


@dataclass
class SlackConfig:
    webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    channel: str = os.getenv("SLACK_CHANNEL", "#ml-alerts")
    enabled: bool = bool(os.getenv("SLACK_WEBHOOK_URL", ""))


@dataclass
class DriftConfig:
    check_interval_minutes: int = int(os.getenv("DRIFT_CHECK_INTERVAL_MINUTES", "30"))
    threshold: float = float(os.getenv("DRIFT_THRESHOLD", "0.15"))
    performance_drop_threshold: float = float(os.getenv("PERFORMANCE_DROP_THRESHOLD", "0.05"))
    methods: list = field(default_factory=lambda: ["psi", "ks", "wasserstein"])


@dataclass
class ABTestConfig:
    sample_size: int = int(os.getenv("AB_TEST_SAMPLE_SIZE", "1000"))
    confidence_level: float = float(os.getenv("AB_TEST_CONFIDENCE", "0.95"))
    min_duration_hours: int = 24
    max_duration_hours: int = 168  # 1 week


@dataclass
class EUAIActConfig:
    system_name: str = os.getenv("AI_SYSTEM_NAME", "credit-risk-classifier")
    risk_category: str = os.getenv("AI_RISK_CATEGORY", "high-risk")
    organization: str = os.getenv("ORGANIZATION_NAME", "Your Organization")
    deployer_country: str = os.getenv("DEPLOYER_COUNTRY", "NL")
    annex_iii_category: str = "creditworthiness-assessment"
    requires_conformity_assessment: bool = True


@dataclass
class PathConfig:
    artifacts: Path = BASE_DIR / os.getenv("MODEL_ARTIFACTS_PATH", "artifacts")
    reports: Path = BASE_DIR / os.getenv("REPORTS_PATH", "reports_output")
    audit_logs: Path = BASE_DIR / os.getenv("AUDIT_LOG_PATH", "audit_logs")
    data: Path = BASE_DIR / os.getenv("DATA_PATH", "data")

    def __post_init__(self):
        for path in [self.artifacts, self.reports, self.audit_logs, self.data]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class GuardianConfig:
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    ab_test: ABTestConfig = field(default_factory=ABTestConfig)
    eu_ai_act: EUAIActConfig = field(default_factory=EUAIActConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def load_thresholds() -> dict:
    """Load threshold configuration from YAML."""
    threshold_path = BASE_DIR / "config" / "thresholds.yaml"
    if threshold_path.exists():
        with open(threshold_path) as f:
            return yaml.safe_load(f)
    return {}


# Global config instance
config = GuardianConfig()
thresholds = load_thresholds()
