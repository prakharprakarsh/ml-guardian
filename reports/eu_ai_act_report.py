"""
ML Guardian — EU AI Act Compliant Report Generator
Generates structured model cards and compliance documentation.
"""
import json
from datetime import datetime
from pathlib import Path

from config.settings import config


def generate_model_card(model_metadata: dict, output_dir: str = None) -> str:
    """
    Generate an EU AI Act compliant model card.
    Reference: Article 11 — Technical Documentation
    """
    output_path = Path(output_dir or str(config.paths.reports))
    output_path.mkdir(parents=True, exist_ok=True)

    version = model_metadata.get("version", "unknown")
    metrics = model_metadata.get("metrics", {})

    card = f"""
================================================================================
MODEL CARD — EU AI Act Technical Documentation (Article 11)
================================================================================

1. MODEL OVERVIEW
-----------------
Name:               {config.eu_ai_act.system_name}
Version:            {version}
Type:               {model_metadata.get('model_type', 'XGBClassifier')}
Risk Classification: {config.eu_ai_act.risk_category}
Annex III Category:  {config.eu_ai_act.annex_iii_category}
Organization:        {config.eu_ai_act.organization}
Training Date:       {model_metadata.get('trained_at', 'N/A')}

2. INTENDED USE
---------------
Purpose:            Credit risk assessment for loan applications
Users:              Loan officers and automated decision pipelines
Limitations:        Not intended for use outside credit risk context
Deployment Region:  {config.eu_ai_act.deployer_country} (EU)

3. TRAINING DATA
----------------
Training Samples:   {model_metadata.get('n_training_samples', 'N/A')}
Test Samples:       {model_metadata.get('n_test_samples', 'N/A')}
Features Used:      {', '.join(model_metadata.get('feature_columns', []))}

4. PERFORMANCE METRICS
----------------------
Accuracy:           {metrics.get('accuracy', 'N/A')}
F1 Score:           {metrics.get('f1_score', 'N/A')}
AUC-ROC:            {metrics.get('auc_roc', 'N/A')}
Precision:          {metrics.get('precision', 'N/A')}
Recall:             {metrics.get('recall', 'N/A')}

5. FAIRNESS ASSESSMENT
----------------------
Protected Attributes Monitored: gender, age_group, nationality
Fairness Metrics:   Demographic parity, equalized odds, disparate impact
Bias Mitigation:    Feature exclusion of protected attributes from model input

6. RISK MANAGEMENT (Article 9)
------------------------------
Monitoring:         Continuous drift detection via Evidently AI
Retraining:         Automated pipeline triggered on drift detection
Rollback:           Automatic rollback to previous version on critical failure
Incident Reporting: Automated EU AI Act Article 62 compliant reports

7. HUMAN OVERSIGHT (Article 14)
-------------------------------
Escalation:         Slack notifications to ML operations team
Override:           Human can manually trigger rollback via dashboard
Audit Trail:        All actions logged in JSONL format

8. TRANSPARENCY (Article 13)
-----------------------------
Explainability:     SHAP values available for individual predictions
Documentation:      This model card + technical documentation
User Information:   Loan applicants informed of automated decision involvement

================================================================================
Generated: {datetime.utcnow().isoformat()}
ML Guardian — Agentic ML Model Lifecycle Guardian
================================================================================
"""

    card_path = output_path / f"model_card_{version}.txt"
    with open(card_path, "w") as f:
        f.write(card)

    # Also save as JSON
    json_card = {
        "model_overview": {
            "name": config.eu_ai_act.system_name,
            "version": version,
            "type": model_metadata.get("model_type", "XGBClassifier"),
            "risk_classification": config.eu_ai_act.risk_category,
        },
        "performance": metrics,
        "training_data": {
            "n_training": model_metadata.get("n_training_samples"),
            "n_test": model_metadata.get("n_test_samples"),
            "features": model_metadata.get("feature_columns", []),
        },
        "compliance": {
            "article_9_risk_management": True,
            "article_11_technical_docs": True,
            "article_13_transparency": True,
            "article_14_human_oversight": True,
            "article_62_incident_reporting": True,
        },
        "generated_at": datetime.utcnow().isoformat(),
    }

    json_path = output_path / f"model_card_{version}.json"
    with open(json_path, "w") as f:
        json.dump(json_card, f, indent=2)

    return str(card_path)
