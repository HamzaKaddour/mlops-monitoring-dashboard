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
