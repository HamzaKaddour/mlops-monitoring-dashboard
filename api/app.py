"""FastAPI service exposing generated model-monitoring artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

app = FastAPI(
    title="MLOps Monitoring API",
    version="1.0.0",
    description="Read-only API for model performance, drift, prediction, and model-card artifacts.",
)


def load_json(filename: str) -> dict[str, Any]:
    path = DATA_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Monitoring artifact not found: {filename}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return load_json("model_metrics.json")


@app.get("/drift")
def drift() -> dict[str, Any]:
    return load_json("drift_report.json")


@app.get("/predictions")
def predictions() -> dict[str, Any]:
    return load_json("prediction_logs.json")


@app.get("/model-card")
def model_card() -> dict[str, Any]:
    return load_json("model_card.json")
