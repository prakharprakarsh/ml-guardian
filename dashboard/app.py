"""
ML Guardian — Streamlit Monitoring Dashboard
Real-time visibility into the ML Guardian system.
"""
import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(
    page_title="ML Guardian Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
ARTIFACTS_DIR = Path("artifacts")
REPORTS_DIR = Path("reports_output")
AUDIT_DIR = Path("audit_logs")
DATA_DIR = Path("data")

# Ensure directories exist
for d in [ARTIFACTS_DIR, REPORTS_DIR, AUDIT_DIR, DATA_DIR]:
    d.mkdir(exist_ok=True)


def load_latest_drift_report() -> dict:
    """Load the most recent drift report JSON."""
    reports = sorted(REPORTS_DIR.glob("drift_report_*.json"), reverse=True)
    if reports:
        with open(reports[0]) as f:
            return json.load(f)
    return {}


def load_audit_log() -> list:
    """Load today's audit log."""
    today = datetime.utcnow().strftime("%Y%m%d")
    log_file = AUDIT_DIR / f"audit_{today}.jsonl"
    entries = []
    if log_file.exists():
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def load_model_metadata() -> dict:
    """Load the latest model metadata."""
    metas = sorted(ARTIFACTS_DIR.glob("metadata_*.json"), reverse=True)
    if metas:
        with open(metas[0]) as f:
            return json.load(f)
    return {}


def load_incident_reports() -> list:
    """Load all incident reports."""
    incidents = sorted(REPORTS_DIR.glob("INC-*.json"), reverse=True)
    reports = []
    for inc_path in incidents[:10]:
        with open(inc_path) as f:
            reports.append(json.load(f))
    return reports


# ─── SIDEBAR ─────────────────────────────────────────────
st.sidebar.title("🛡️ ML Guardian")
st.sidebar.markdown("**Agentic ML Lifecycle Monitor**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Drift Analysis", "Model Registry", "Incident Reports", "Audit Trail"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**EU AI Act Compliance**")
st.sidebar.markdown("System: `credit-risk-classifier`")
st.sidebar.markdown("Risk: `High-Risk (Annex III)`")
st.sidebar.markdown("Country: `NL`")


# ─── OVERVIEW PAGE ───────────────────────────────────────
if page == "Overview":
    st.title("🛡️ ML Guardian — System Overview")
    st.markdown("Real-time monitoring of your production ML model lifecycle.")

    drift_report = load_latest_drift_report()
    model_meta = load_model_metadata()
    audit_entries = load_audit_log()

    # Status metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    # Determine system status
    drift_score = drift_report.get("drift_score", 0.0)
    if drift_score > 0.20:
        status_label = "🚨 CRITICAL"
    elif drift_score > 0.10:
        status_label = "⚠️ WARNING"
    else:
        status_label = "✅ HEALTHY"

    col1.metric("System Status", status_label)
    col2.metric("Drift Score", f"{drift_score:.4f}")
    col3.metric(
        "Model Version",
        model_meta.get("version", "N/A"),
    )
    col4.metric(
        "Accuracy",
        f"{model_meta.get('metrics', {}).get('accuracy', 0):.2%}",
    )
    col5.metric("Actions Today", len(audit_entries))

    st.markdown("---")

    # Two column layout
    left, right = st.columns(2)

    with left:
        st.subheader("📊 Current Model Performance")
        metrics = model_meta.get("metrics", {})
        if metrics:
            metrics_df = pd.DataFrame(
                [{"Metric": k, "Value": f"{v:.4f}"} for k, v in metrics.items()]
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        else:
            st.info("No model metrics available. Train a model first.")

    with right:
        st.subheader("🔄 Recent Agent Actions")
        if audit_entries:
            recent = audit_entries[-10:][::-1]
            for entry in recent:
                action = entry.get("action", "unknown")
                ts = entry.get("timestamp", "")[:19]
                icon = {
                    "drift_check": "🔍",
                    "retrain_triggered": "🔄",
                    "retrain_completed": "✅",
                    "ab_test_started": "🧪",
                    "ab_test_completed": "📊",
                    "model_promoted": "🏆",
                    "rollback_executed": "⏪",
                    "incident_reported": "🚨",
                }.get(action, "📋")
                st.markdown(f"`{ts}` {icon} **{action}**")
        else:
            st.info("No actions logged yet. Run a monitoring cycle.")

    # Drift features chart
    st.markdown("---")
    st.subheader("📈 Drifted Features")
    drifted = drift_report.get("drifted_features", [])
    if drifted:
        st.warning(f"{len(drifted)} features showing significant drift:")
        for feat in drifted:
            st.markdown(f"  • `{feat}`")
    else:
        st.success("No significant feature drift detected.")


# ─── DRIFT ANALYSIS PAGE ────────────────────────────────
elif page == "Drift Analysis":
    st.title("🔍 Drift Analysis")

    drift_report = load_latest_drift_report()

    if drift_report:
        st.markdown(f"**Last check:** {drift_report.get('timestamp', 'N/A')}")
        st.markdown(f"**Method:** {drift_report.get('method', 'PSI')}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset Drift", "Yes" if drift_report.get("dataset_drift") else "No")
        col2.metric("Drift Score", f"{drift_report.get('drift_score', 0):.4f}")
        col3.metric(
            "Drifted Features",
            f"{drift_report.get('num_drifted_features', 0)} / {drift_report.get('total_features', 0)}",
        )

        st.markdown("---")

        # Performance comparison
        st.subheader("Model Performance on Production Data")
        perf = drift_report.get("performance_metrics", {})
        if perf:
            perf_df = pd.DataFrame(
                [{"Metric": k, "Production Value": f"{v:.4f}"} for k, v in perf.items()]
            )
            st.dataframe(perf_df, use_container_width=True, hide_index=True)

        # Link to Evidently HTML report
        st.markdown("---")
        html_reports = sorted(REPORTS_DIR.glob("drift_*.html"), reverse=True)
        if html_reports:
            st.subheader("📄 Full Evidently Reports")
            for rpt in html_reports[:5]:
                st.markdown(f"  • `{rpt.name}`")
            st.info("Open these HTML files in a browser for interactive drift analysis.")
    else:
        st.info("No drift reports available. Run a monitoring cycle first.")


# ─── MODEL REGISTRY PAGE ────────────────────────────────
elif page == "Model Registry":
    st.title("📦 Model Registry")

    metadata_files = sorted(ARTIFACTS_DIR.glob("metadata_*.json"), reverse=True)

    if metadata_files:
        for meta_path in metadata_files[:10]:
            with open(meta_path) as f:
                meta = json.load(f)

            version = meta.get("version", "unknown")
            trained_at = meta.get("trained_at", "N/A")[:19]
            metrics = meta.get("metrics", {})

            with st.expander(f"📦 Model {version} — trained {trained_at}", expanded=(meta_path == metadata_files[0])):
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                col2.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
                col3.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.4f}")

                st.json(meta)
    else:
        st.info("No models registered. Train a model first.")


# ─── INCIDENT REPORTS PAGE ───────────────────────────────
elif page == "Incident Reports":
    st.title("🚨 EU AI Act Incident Reports")
    st.markdown("Auto-generated under **Article 62** of the EU AI Act.")

    incidents = load_incident_reports()

    if incidents:
        for inc in incidents:
            report_id = inc.get("report_id", "Unknown")
            severity = inc.get("severity", "MEDIUM")
            timestamp = inc.get("timestamp", "N/A")[:19]

            severity_color = "🔴" if severity == "HIGH" else "🟡"

            with st.expander(
                f"{severity_color} {report_id} — {severity} — {timestamp}",
                expanded=(inc == incidents[0]),
            ):
                st.markdown(f"**System:** {inc.get('system_name', 'N/A')}")
                st.markdown(f"**Risk Category:** {inc.get('risk_category', 'N/A')}")
                st.markdown(f"**Detection Method:** {inc.get('detection_method', 'N/A')}")

                st.markdown("---")
                st.markdown("**Current Metrics:**")
                st.json(inc.get("current_metrics", {}))

                st.markdown("**Degradation:**")
                st.json(inc.get("degradation", {}))

                st.markdown("**Root Cause:**")
                st.markdown(inc.get("root_cause", "N/A"))

                st.markdown("**Corrective Measures:**")
                for m in inc.get("corrective_measures", []):
                    st.markdown(f"  • {m}")

                st.markdown("**Compliance Status:**")
                st.json(inc.get("compliance", {}))
    else:
        st.success("No incidents reported. System is operating normally.")


# ─── AUDIT TRAIL PAGE ───────────────────────────────────
elif page == "Audit Trail":
    st.title("📋 Audit Trail")
    st.markdown("Complete log of all agent actions for EU AI Act compliance.")

    # Load all audit logs
    all_entries = []
    for log_file in sorted(AUDIT_DIR.glob("audit_*.jsonl"), reverse=True):
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_entries.append(json.loads(line))

    if all_entries:
        # Reverse to show most recent first
        all_entries = all_entries[::-1]

        # Filters
        actions = list(set(e.get("action", "") for e in all_entries))
        selected_actions = st.multiselect(
            "Filter by action type:", actions, default=actions
        )

        filtered = [e for e in all_entries if e.get("action") in selected_actions]

        st.markdown(f"Showing **{len(filtered)}** of **{len(all_entries)}** entries.")

        for entry in filtered[:50]:
            ts = entry.get("timestamp", "")[:19]
            action = entry.get("action", "unknown")
            status = entry.get("status", "")
            details = json.dumps(entry.get("details", {}), indent=2)

            with st.expander(f"`{ts}` — **{action}** [{status}]"):
                st.code(details, language="json")
    else:
        st.info("No audit entries yet. Run a monitoring cycle first.")


# ─── FOOTER ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "ML Guardian — Agentic ML Model Lifecycle Guardian | "
    "EU AI Act Compliant | Built by Prakhar Prakarsh"
    "</div>",
    unsafe_allow_html=True,
)
