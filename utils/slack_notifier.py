"""
ML Guardian — Slack Notification Service
Sends alerts to Slack when agents take action.
"""
import json
import requests
from datetime import datetime
from config.settings import config
from utils.logger import get_logger

logger = get_logger("slack", agent="notifier")


class SlackNotifier:
    """Sends formatted notifications to Slack."""

    SEVERITY_EMOJI = {
        "info": "ℹ️",
        "warning": "⚠️",
        "critical": "🚨",
        "success": "✅",
        "rollback": "⏪",
    }

    SEVERITY_COLOR = {
        "info": "#36a64f",
        "warning": "#ff9900",
        "critical": "#ff0000",
        "success": "#2eb886",
        "rollback": "#764FA5",
    }

    def __init__(self):
        self.webhook_url = config.slack.webhook_url
        self.enabled = config.slack.enabled

    def send(self, title: str, message: str, severity: str = "info",
             fields: dict = None, agent: str = "Guardian"):
        """Send a formatted Slack notification."""
        if not self.enabled:
            logger.info(f"[Slack Disabled] {severity.upper()}: {title} — {message}")
            return

        emoji = self.SEVERITY_EMOJI.get(severity, "📋")
        color = self.SEVERITY_COLOR.get(severity, "#cccccc")

        attachment_fields = []
        if fields:
            for key, value in fields.items():
                attachment_fields.append({
                    "title": key,
                    "value": str(value),
                    "short": True,
                })

        payload = {
            "channel": config.slack.channel,
            "username": "ML Guardian",
            "icon_emoji": ":shield:",
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} {title}",
                    "text": message,
                    "fields": attachment_fields,
                    "footer": f"ML Guardian • {agent} Agent",
                    "ts": int(datetime.utcnow().timestamp()),
                }
            ],
        }

        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if response.status_code != 200:
                logger.warning(f"Slack API returned {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def send_drift_alert(self, drift_score: float, drifted_features: list,
                         severity: str = "warning"):
        """Send a drift detection alert."""
        self.send(
            title="Data Drift Detected",
            message=f"Drift score: {drift_score:.4f}\n"
                    f"Drifted features: {', '.join(drifted_features[:5])}",
            severity=severity,
            fields={
                "Drift Score": f"{drift_score:.4f}",
                "Features Affected": len(drifted_features),
            },
            agent="Drift",
        )

    def send_retrain_alert(self, old_version: str, new_version: str,
                           metrics: dict):
        """Send a retraining completion alert."""
        self.send(
            title="Model Retraining Complete",
            message=f"New model version {new_version} trained.\n"
                    f"Previous version: {old_version}",
            severity="info",
            fields=metrics,
            agent="Retrain",
        )

    def send_incident_alert(self, incident_id: str, reason: str,
                            metrics: dict):
        """Send an EU AI Act incident report alert."""
        self.send(
            title=f"Incident Report Generated: {incident_id}",
            message=f"Reason: {reason}\n"
                    f"An EU AI Act compliant incident report has been filed.",
            severity="critical",
            fields=metrics,
            agent="Incident",
        )

    def send_rollback_alert(self, from_version: str, to_version: str,
                            reason: str):
        """Send a rollback notification."""
        self.send(
            title="Model Rollback Executed",
            message=f"Rolled back from v{from_version} to v{to_version}\n"
                    f"Reason: {reason}",
            severity="rollback",
            fields={
                "From Version": from_version,
                "To Version": to_version,
                "Reason": reason,
            },
            agent="Rollback",
        )


# Singleton instance
notifier = SlackNotifier()
