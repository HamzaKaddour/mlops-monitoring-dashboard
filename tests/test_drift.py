from mlops_monitoring.drift import (
    drift_status,
    population_stability_index_categorical,
    population_stability_index_numeric,
)


def test_numeric_psi_is_small_for_similar_distributions():
    reference = list(range(100))
    current = [x + 1 for x in range(100)]
    score = population_stability_index_numeric(reference, current, bins=10)
    assert score < 0.10


def test_numeric_psi_detects_large_shift():
    reference = list(range(100))
    current = [x + 100 for x in range(100)]
    score = population_stability_index_numeric(reference, current, bins=10)
    assert score >= 0.25


def test_categorical_psi_detects_distribution_change():
    reference = ["annual"] * 70 + ["monthly"] * 30
    current = ["annual"] * 30 + ["monthly"] * 70
    score = population_stability_index_categorical(reference, current)
    assert score >= 0.25


def test_status_thresholds():
    assert drift_status(0.05) == "stable"
    assert drift_status(0.15) == "watch"
    assert drift_status(0.30) == "alert"
