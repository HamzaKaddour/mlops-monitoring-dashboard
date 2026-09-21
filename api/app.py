"""FastAPI service for model inference and monitoring artifacts."""

from __future__ import annotations

import json
from time import perf_counter
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from mlops_monitoring.inference import predict_one
from mlops_monitoring.prediction_store import log_prediction, recent_predictions
from mlops_monitoring.prometheus_metrics import (
    INFERENCE_LATENCY,
    POSITIVE_PREDICTIONS,
    PREDICTION_PROBABILITY,
    PREDICTION_REQUESTS,
)
from mlops_monitoring.retraining import recommend_retraining

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

app = FastAPI(
    title="MLOps Monitoring API",
    version="0.2.0",
    description="Model inference, prediction logging, drift monitoring, and retraining signals.",
)


class PredictionRequest(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=500)
    support_tickets_30d: int = Field(ge=0, le=50)
    contract_type: Literal["month_to_month", "one_year", "two_year"]
    payment_method: Literal["card", "bank_transfer", "electronic_check"]
    internet_service: Literal["fiber", "dsl", "none"]


def load_json(filename: str, directory: Path = DATA_DIR) -> dict[str, Any]:
    path = directory / filename
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Monitoring artifact not found: {filename}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    PREDICTION_REQUESTS.inc()
    started = perf_counter()
    try:
        result = predict_one(request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        INFERENCE_LATENCY.observe(perf_counter() - started)

    PREDICTION_PROBABILITY.set(result["churn_probability"])
    if result["prediction"] == 1:
        POSITIVE_PREDICTIONS.inc()

    log_prediction(
        features=request.model_dump(),
        prediction=result["prediction"],
        probability=result["churn_probability"],
        model_version=result["model_version"],
    )
    return result


@app.get("/prediction-events")
def prediction_events(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    return {"items": recent_predictions(limit=limit)}


@app.get("/training-summary")
def training_summary() -> dict[str, Any]:
    return load_json("training_summary.json", ARTIFACT_DIR)


@app.get("/prometheus-metrics")
def prometheus_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return load_json("model_metrics.json")


@app.get("/evidently-summary")
def evidently_summary() -> dict[str, Any]:
    return load_json("evidently_drift_summary.json")


@app.get("/drift")
def drift() -> dict[str, Any]:
    return load_json("drift_report.json")


@app.get("/predictions")
def predictions() -> dict[str, Any]:
    return load_json("prediction_logs.json")


@app.get("/model-card")
def model_card() -> dict[str, Any]:
    return load_json("model_card.json")


@app.get("/retraining-status")
def retraining_status() -> dict[str, Any]:
    metrics_payload = load_json("model_metrics.json")
    drift_payload = load_json("drift_report.json")

    auc_delta = float(metrics_payload.get("summary", {}).get("auc_delta_from_validation", 0.0))
    drift_score = float(drift_payload.get("drift_score", 0.0))
    high_drift_feature_count = sum(
        1
        for feature in drift_payload.get("features", [])
        if float(feature.get("drift_score", 0.0)) >= 0.25
    )
    return recommend_retraining(
        auc_delta_from_validation=auc_delta,
        overall_drift_score=drift_score,
        high_drift_feature_count=high_drift_feature_count,
    )
