"""Model loading and inference helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
SUMMARY_PATH = ARTIFACT_DIR / "training_summary.json"


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact is missing. Run: python scripts/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def load_training_summary() -> dict[str, Any]:
    if not SUMMARY_PATH.exists():
        return {}
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def predict_one(features: dict[str, Any]) -> dict[str, Any]:
    model = load_model()
    frame = pd.DataFrame([features])
    probability = float(model.predict_proba(frame)[:, 1][0])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "churn_probability": round(probability, 6),
        "model_version": load_training_summary().get("model_version", "unknown"),
    }
