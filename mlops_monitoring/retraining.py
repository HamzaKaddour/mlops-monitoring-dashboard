"""Rule-based retraining recommendation logic."""

from __future__ import annotations


def recommend_retraining(
    auc_delta_from_validation: float,
    overall_drift_score: float,
    high_drift_feature_count: int = 0,
) -> dict:
    reasons: list[str] = []

    if auc_delta_from_validation <= -0.05:
        reasons.append("production AUC dropped by at least 0.05 from validation")
    if overall_drift_score >= 0.25:
        reasons.append("overall feature drift exceeded PSI 0.25")
    if high_drift_feature_count >= 2:
        reasons.append("multiple features are in the drift alert range")

    recommended = bool(reasons)
    if recommended:
        status = "retrain"
    elif auc_delta_from_validation <= -0.03 or overall_drift_score >= 0.10:
        status = "watch"
    else:
        status = "healthy"

    return {
        "status": status,
        "retraining_recommended": recommended,
        "reasons": reasons,
        "policy": {
            "auc_drop_threshold": -0.05,
            "overall_psi_threshold": 0.25,
            "high_drift_feature_threshold": 2,
        },
    }
