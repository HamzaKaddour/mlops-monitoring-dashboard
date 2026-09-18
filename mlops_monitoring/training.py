"""Training utilities for the MLOps monitoring project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_tickets_30d"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "internet_service"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn_within_30_days"


@dataclass
class CandidateResult:
    name: str
    model: Pipeline
    metrics: dict[str, float]


def make_synthetic_churn_data(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Create deterministic, non-proprietary churn-like tabular data."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, size=n)
    monthly = np.clip(rng.normal(72, 24, size=n), 20, 150)
    tickets = np.clip(rng.poisson(1.6, size=n), 0, 12)
    contract = rng.choice(["month_to_month", "one_year", "two_year"], size=n, p=[0.55, 0.27, 0.18])
    payment = rng.choice(["card", "bank_transfer", "electronic_check"], size=n, p=[0.42, 0.34, 0.24])
    internet = rng.choice(["fiber", "dsl", "none"], size=n, p=[0.52, 0.36, 0.12])

    logit = (
        -1.15
        - 0.025 * tenure
        + 0.018 * (monthly - 70)
        + 0.34 * tickets
        + 0.95 * (contract == "month_to_month")
        - 0.55 * (contract == "two_year")
        + 0.38 * (payment == "electronic_check")
        + 0.28 * (internet == "fiber")
    )
    prob = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, np.clip(prob, 0.02, 0.95))

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly.round(2),
            "support_tickets_30d": tickets,
            "contract_type": contract,
            "payment_method": payment,
            "internet_service": internet,
            TARGET: target,
        }
    )


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def candidate_models(seed: int = 42) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=9,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.06,
            max_leaf_nodes=20,
            random_state=seed,
        ),
    }


def build_pipeline(estimator: object) -> Pipeline:
    return Pipeline([("preprocess", _preprocessor()), ("model", estimator)])


def classification_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, prob)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "log_loss": round(float(log_loss(y, prob)), 4),
    }
