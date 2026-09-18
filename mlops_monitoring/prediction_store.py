"""SQLite persistence for live prediction traces."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "predictions.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model_version TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL NOT NULL,
            features_json TEXT NOT NULL
        )
        """
    )
    return conn


def log_prediction(
    features: dict[str, Any],
    prediction: int,
    probability: float,
    model_version: str,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO prediction_events
            (created_at, model_version, prediction, probability, features_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                model_version,
                int(prediction),
                float(probability),
                json.dumps(features, sort_keys=True),
            ),
        )


def recent_predictions(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, model_version, prediction, probability, features_json
            FROM prediction_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    return [
        {
            "id": row[0],
            "created_at": row[1],
            "model_version": row[2],
            "prediction": row[3],
            "probability": row[4],
            "features": json.loads(row[5]),
        }
        for row in rows
    ]
