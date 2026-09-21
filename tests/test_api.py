from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_returns_model_metadata():
    response = client.get("/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "model_name" in payload
    assert "summary" in payload


def test_drift_endpoint_returns_features():
    response = client.get("/drift")
    assert response.status_code == 200
    payload = response.json()
    assert "features" in payload
    assert isinstance(payload["features"], list)


def test_predict_endpoint_runs_saved_model():
    response = client.post(
        "/predict",
        json={
            "tenure_months": 8,
            "monthly_charges": 98.5,
            "support_tickets_30d": 4,
            "contract_type": "month_to_month",
            "payment_method": "electronic_check",
            "internet_service": "fiber",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in (0, 1)
    assert 0.0 <= payload["churn_probability"] <= 1.0


def test_retraining_status_endpoint():
    response = client.get("/retraining-status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"healthy", "watch", "retrain"}
    assert "retraining_recommended" in payload


def test_training_summary_endpoint():
    response = client.get("/training-summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_candidate"] in {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert "test_metrics" in payload
    assert payload["mlflow"]["tracking_uri"] == "sqlite:///mlflow.db"


def test_prometheus_metrics_endpoint():
    response = client.get("/prometheus-metrics")
    assert response.status_code == 200
    assert "mlops_prediction_requests_total" in response.text
    assert "mlops_inference_latency_seconds" in response.text


def test_evidently_summary_endpoint():
    response = client.get("/evidently-summary")
    assert response.status_code == 200
    payload = response.json()
    assert "metrics" in payload
