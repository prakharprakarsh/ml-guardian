"""
ML Guardian — Drift Simulation Script
Run this to demonstrate the full ML Guardian lifecycle:
1. Normal data → no drift detected
2. Drifted data → drift detected → retrain → A/B test → promote/rollback
"""
import pandas as pd
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import GuardianOrchestrator
from pipelines.data_generator import generate_credit_data
from config.settings import config


def run_simulation():
    """Run a complete ML Guardian simulation."""
    print("=" * 70)
    print("  ML Guardian — Full Lifecycle Simulation")
    print("  Demonstrating: Drift Detection → Retrain → A/B Test → Recovery")
    print("=" * 70)

    guardian = GuardianOrchestrator()
    guardian.initialize()

    # ─── SCENARIO 1: Normal Operations ────────────────────────
    print("\n" + "=" * 70)
    print("  SCENARIO 1: Normal Operations (No Drift)")
    print("=" * 70)

    normal_data = generate_credit_data(n_samples=3000, drift=False, seed=100)
    result1 = guardian.run_cycle(normal_data)

    print(f"\n  Status:       {result1['status']}")
    print(f"  Drift Score:  {result1['drift_score']:.4f}")
    print(f"  Drift Found:  {result1['drift_detected']}")
    print(f"  Champion:     {result1['champion_version']}")

    # ─── SCENARIO 2: Mild Drift ──────────────────────────────
    print("\n" + "=" * 70)
    print("  SCENARIO 2: Mild Economic Shift (Warning-Level Drift)")
    print("=" * 70)

    mild_drift = generate_credit_data(
        n_samples=3000, drift=True, drift_intensity=0.2, seed=200
    )
    result2 = guardian.run_cycle(mild_drift)

    print(f"\n  Status:       {result2['status']}")
    print(f"  Drift Score:  {result2['drift_score']:.4f}")
    print(f"  Drift Found:  {result2['drift_detected']}")

    # ─── SCENARIO 3: Severe Drift ────────────────────────────
    print("\n" + "=" * 70)
    print("  SCENARIO 3: Economic Downturn (Critical Drift)")
    print("  This triggers: Retrain → A/B Test → Promote/Rollback")
    print("=" * 70)

    severe_drift = generate_credit_data(
        n_samples=5000, drift=True, drift_intensity=0.5, seed=300
    )
    # Save for retrain pipeline
    severe_drift.to_csv(str(config.paths.data / "production_drifted.csv"), index=False)
    result3 = guardian.run_cycle(severe_drift)

    print(f"\n  Status:       {result3['status']}")
    print(f"  Drift Score:  {result3['drift_score']:.4f}")
    print(f"  Drift Found:  {result3['drift_detected']}")
    print(f"  Champion:     {result3['champion_version']}")
    print(f"  A/B Active:   {result3['ab_test_active']}")
    print(f"  Incidents:    {result3['total_incidents']}")
    print(f"  Retrains:     {result3['total_retrains']}")

    # ─── SUMMARY ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SIMULATION SUMMARY")
    print("=" * 70)
    print(f"  Total Drift Checks:    {result3['total_drift_checks']}")
    print(f"  Total Retrains:        {result3['total_retrains']}")
    print(f"  Total Rollbacks:       {result3['total_rollbacks']}")
    print(f"  Total Incidents:       {result3['total_incidents']}")
    print(f"  Final Status:          {result3['status']}")
    print(f"  Final Champion:        {result3['champion_version']}")

    print("\n  Recent Actions:")
    for action in result3.get("recent_actions", [])[-8:]:
        ts = action.get("timestamp", "")[:19]
        act = action.get("action", "")
        print(f"    [{ts}] {act}")

    print("\n" + "=" * 70)
    print("  Simulation complete!")
    print("  Check reports_output/ for incident reports and drift analysis.")
    print("  Check audit_logs/ for the full audit trail (JSONL).")
    print("  Run: streamlit run dashboard/app.py — to view the dashboard.")
    print("=" * 70)


if __name__ == "__main__":
    run_simulation()
