from mlops_monitoring.retraining import recommend_retraining


def test_retraining_recommended_for_large_auc_drop():
    result = recommend_retraining(
        auc_delta_from_validation=-0.06,
        overall_drift_score=0.08,
        high_drift_feature_count=0,
    )
    assert result["retraining_recommended"] is True
    assert result["status"] == "retrain"


def test_retraining_watch_for_moderate_drift():
    result = recommend_retraining(
        auc_delta_from_validation=-0.02,
        overall_drift_score=0.15,
        high_drift_feature_count=0,
    )
    assert result["retraining_recommended"] is False
    assert result["status"] == "watch"


def test_retraining_healthy_when_signals_are_stable():
    result = recommend_retraining(
        auc_delta_from_validation=-0.01,
        overall_drift_score=0.05,
        high_drift_feature_count=0,
    )
    assert result["retraining_recommended"] is False
    assert result["status"] == "healthy"
