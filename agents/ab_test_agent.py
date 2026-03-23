"""
ML Guardian — A/B Test Agent
Manages shadow deployments and statistical comparison of model versions.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from core.state import GuardianState, AgentAction, SystemStatus
from config.settings import config
from pipelines.train_initial_model import FEATURE_COLS, TARGET_COL
from utils.logger import get_logger
from utils.slack_notifier import notifier

logger = get_logger("ab_test_agent", agent="ab_test")


class ABTestAgent:
    """
    Runs A/B tests between champion and challenger models.
    
    Responsibilities:
        - Deploy challenger model in shadow mode
        - Collect predictions from both models on same data
        - Run statistical significance tests
        - Promote challenger or keep champion based on results
    """

    def __init__(self):
        self.confidence_level = config.ab_test.confidence_level
        self.min_improvement = 0.02  # 2% minimum improvement to promote

    def execute(
        self,
        state: GuardianState,
        test_data: pd.DataFrame,
    ) -> GuardianState:
        """
        Run A/B test between champion and challenger.
        
        Args:
            state: Current guardian state.
            test_data: Production data with labels for evaluation.
            
        Returns:
            Updated state with A/B test results.
        """
        if not state.challenger_model:
            logger.info("No challenger model available for A/B test.")
            return state

        if not state.champion_model:
            # First model — auto-promote
            logger.info("No champion model. Promoting challenger directly.")
            state.champion_model = state.challenger_model
            state.champion_model.is_champion = True
            state.champion_model.promoted_at = datetime.utcnow().isoformat()
            state.challenger_model = None
            state.status = SystemStatus.HEALTHY
            return state

        logger.info(
            f"Starting A/B test: champion={state.champion_model.version} "
            f"vs challenger={state.challenger_model.version}"
        )
        state.ab_test_active = True
        state.log_action(AgentAction.AB_TEST_STARTED, {
            "champion": state.champion_model.version,
            "challenger": state.challenger_model.version,
        })

        try:
            # Load both models
            champion = joblib.load(state.champion_model.model_uri)
            challenger = joblib.load(state.challenger_model.model_uri)

            # Prepare test data
            X_test = test_data[FEATURE_COLS]
            y_true = test_data[TARGET_COL]

            # Get predictions from both
            champ_pred = champion.predict(X_test)
            champ_proba = champion.predict_proba(X_test)[:, 1]
            chall_pred = challenger.predict(X_test)
            chall_proba = challenger.predict_proba(X_test)[:, 1]

            # Calculate metrics
            champ_metrics = {
                "accuracy": accuracy_score(y_true, champ_pred),
                "f1_score": f1_score(y_true, champ_pred),
                "auc_roc": roc_auc_score(y_true, champ_proba),
            }
            chall_metrics = {
                "accuracy": accuracy_score(y_true, chall_pred),
                "f1_score": f1_score(y_true, chall_pred),
                "auc_roc": roc_auc_score(y_true, chall_proba),
            }

            # Statistical test (McNemar's test for paired predictions)
            significance = self._mcnemar_test(y_true, champ_pred, chall_pred)

            # Decision logic
            improvement = {
                k: chall_metrics[k] - champ_metrics[k]
                for k in champ_metrics
            }

            results = {
                "champion_metrics": {k: round(v, 4) for k, v in champ_metrics.items()},
                "challenger_metrics": {k: round(v, 4) for k, v in chall_metrics.items()},
                "improvement": {k: round(v, 4) for k, v in improvement.items()},
                "p_value": round(significance["p_value"], 6),
                "statistically_significant": significance["significant"],
                "test_samples": len(y_true),
            }

            # Decide: promote, keep, or rollback
            promote = (
                improvement["accuracy"] >= self.min_improvement
                and significance["significant"]
                and all(v >= -0.01 for v in improvement.values())
            )

            if promote:
                results["decision"] = "promote_challenger"
                state.champion_model = state.challenger_model
                state.champion_model.is_champion = True
                state.champion_model.promoted_at = datetime.utcnow().isoformat()
                state.challenger_model = None
                state.status = SystemStatus.HEALTHY
                logger.info(
                    f"Challenger PROMOTED: {state.champion_model.version}"
                )
                notifier.send(
                    title="A/B Test Complete — Challenger Promoted",
                    message=f"Version {state.champion_model.version} is now champion.",
                    severity="success",
                    fields=results["challenger_metrics"],
                    agent="A/B Test",
                )
            else:
                results["decision"] = "keep_champion"
                state.challenger_model = None
                state.status = SystemStatus.HEALTHY
                logger.info("Champion RETAINED — challenger did not improve enough.")
                notifier.send(
                    title="A/B Test Complete — Champion Retained",
                    message="Challenger did not show significant improvement.",
                    severity="info",
                    fields=results["improvement"],
                    agent="A/B Test",
                )

            state.ab_test_results = results
            state.ab_test_active = False
            state.log_action(AgentAction.AB_TEST_COMPLETED, results)

        except Exception as e:
            logger.error(f"A/B test failed: {e}")
            state.ab_test_active = False
            state.status = SystemStatus.WARNING

        return state

    def _mcnemar_test(
        self, y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
    ) -> dict:
        """
        Run McNemar's test for comparing two classifiers.
        Tests whether the two models have different error rates.
        """
        # Build contingency table
        correct_a = (pred_a == y_true)
        correct_b = (pred_b == y_true)

        # b = A correct, B wrong; c = A wrong, B correct
        b = np.sum(correct_a & ~correct_b)
        c = np.sum(~correct_a & correct_b)

        # McNemar's test with continuity correction
        if b + c == 0:
            return {"p_value": 1.0, "significant": False, "statistic": 0.0}

        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

        return {
            "p_value": p_value,
            "significant": p_value < (1 - self.confidence_level),
            "statistic": statistic,
        }
