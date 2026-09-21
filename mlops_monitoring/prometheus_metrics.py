"""Prometheus instrumentation for live model inference."""

from prometheus_client import Counter, Gauge, Histogram

PREDICTION_REQUESTS = Counter(
    "mlops_prediction_requests_total",
    "Total number of prediction requests.",
)
POSITIVE_PREDICTIONS = Counter(
    "mlops_positive_predictions_total",
    "Total number of positive churn predictions.",
)
PREDICTION_PROBABILITY = Gauge(
    "mlops_last_prediction_probability",
    "Probability returned by the most recent prediction.",
)
INFERENCE_LATENCY = Histogram(
    "mlops_inference_latency_seconds",
    "Model inference latency in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
