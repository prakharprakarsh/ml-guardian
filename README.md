#  ML Guardian — Agentic ML Model Lifecycle Guardian

> An autonomous MLOps agent that watches production ML models 24/7: detects data drift, triggers automatic retraining, runs A/B tests, generates EU AI Act–mandated incident reports, and auto-rolls back bad deployments.

**Essentially an SRE for Machine Learning — but fully autonomous.**

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ML GUARDIAN ORCHESTRATOR                  │
│              (LangGraph Agent State Machine)                 │
├─────────┬──────────┬──────────┬──────────┬─────────────────┤
│  Drift  │ Retrain  │  A/B     │ Incident │  Rollback       │
│  Agent  │  Agent   │  Agent   │  Agent   │  Agent          │
├─────────┴──────────┴──────────┴──────────┴─────────────────┤
│                    SHARED SERVICES                           │
│  MLflow  │  Evidently  │  Slack  │  FastAPI  │  PostgreSQL  │
└─────────────────────────────────────────────────────────────┘
```

##  Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/ml-guardian.git
cd ml-guardian
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings (Slack webhook, MLflow URI, etc.)
```

### 3. Train the Demo Model
```bash
python -m pipelines.train_initial_model
```

### 4. Start the Guardian
```bash
python -m core.orchestrator
```

### 5. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

### 6. Simulate Drift (Demo)
```bash
python -m tests.simulate_drift
```

---

##  Project Structure

```
ml-guardian/
├── agents/                  # Individual autonomous agents
│   ├── drift_agent.py       # Monitors data & model drift via Evidently
│   ├── retrain_agent.py     # Triggers automated retraining pipeline
│   ├── ab_test_agent.py     # Manages A/B tests between model versions
│   ├── incident_agent.py    # Generates EU AI Act incident reports
│   └── rollback_agent.py    # Auto-rolls back failed deployments
├── config/
│   ├── settings.py          # Central configuration
│   └── thresholds.yaml      # Drift & performance thresholds
├── core/
│   ├── orchestrator.py      # LangGraph-based agent orchestrator
│   └── state.py             # Shared state management
├── dashboard/
│   └── app.py               # Streamlit monitoring dashboard
├── models/
│   └── registry.py          # MLflow model registry wrapper
├── pipelines/
│   ├── train_initial_model.py  # Initial model training
│   ├── retrain_pipeline.py     # Automated retraining
│   └── data_generator.py       # Synthetic data for demo
├── reports/
│   └── eu_ai_act_report.py  # EU AI Act compliant report generator
├── tests/
│   ├── simulate_drift.py    # Drift simulation for testing
│   ├── test_agents.py       # Unit tests for all agents
│   └── test_orchestrator.py # Integration tests
├── utils/
│   ├── slack_notifier.py    # Slack webhook integration
│   └── logger.py            # Structured logging
├── .github/workflows/
│   └── ci.yml               # CI/CD with fairness gate
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

##  Agent Details

| Agent | Trigger | Action | EU AI Act Relevance |
|-------|---------|--------|---------------------|
| **Drift Agent** | Scheduled (every 30min) | Runs Evidently data drift + model performance checks | Art. 9: Risk Management |
| **Retrain Agent** | Drift detected above threshold | Launches retraining pipeline with fresh data | Art. 9: Continuous improvement |
| **A/B Test Agent** | New model version available | Deploys shadow model, compares metrics | Art. 15: Accuracy |
| **Incident Agent** | Performance drop > 5% | Generates JSONL audit report + Slack alert | Art. 62: Incident Reporting |
| **Rollback Agent** | A/B test fails or critical drift | Reverts to last known good model version | Art. 14: Human Oversight |

---

## 🇪🇺 EU AI Act Compliance Features

- **Risk classification**: Auto-classifies model under Annex III categories
- **Audit trail**: Every agent action logged in JSONL format
- **Incident reports**: Auto-generated when thresholds breached
- **Fairness auditing**: Bias detection across protected attributes
- **Human oversight**: Escalation to human for critical decisions
- **Technical documentation**: Auto-generated model cards

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph (LangChain) |
| Drift Detection | Evidently AI |
| Model Registry | MLflow |
| ML Framework | Scikit-learn, XGBoost |
| API | FastAPI |
| Dashboard | Streamlit |
| Notifications | Slack Webhooks |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |
| Database | SQLite (demo) / PostgreSQL (prod) |

---

##  License

MIT License — Built by [Prakhar Prakarsh](https://github.com/YOUR_USERNAME)
