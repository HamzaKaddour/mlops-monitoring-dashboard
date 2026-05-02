"""Generate lightweight MLOps monitoring artifacts.

The script simulates a binary classification monitoring workflow and writes JSON
artifacts consumed by the static dashboard. It uses only the Python standard
library so it can run quickly in GitHub Actions without dependency installation.
"""

from __future__ import annotations

import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RNG = random.Random(7)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def clip(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return max(low, min(high, value))


def simulate_window(n: int, quality: float, positive_rate: float) -> tuple[list[int], list[float]]:
    y_true = [1 if RNG.random() < positive_rate else 0 for _ in range(n)]
    probs = []
    for label in y_true:
        center = quality if label == 1 else 1 - quality
        probs.append(clip(center + RNG.gauss(0, 0.13)))
    return y_true, probs


def auc_score(y_true: list[int], probs: list[float]) -> float:
    pairs = sorted(zip(probs, y_true), key=lambda item: item[0])
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += rank
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def classification_metrics(y_true: list[int], probs: list[float]) -> dict:
    y_pred = [1 if p >= 0.5 else 0 for p in probs]
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true)
    loss = -sum(
        yt * math.log(clip(p)) + (1 - yt) * math.log(clip(1 - p))
        for yt, p in zip(y_true, probs)
    ) / len(y_true)

    return {
        "auc": round(auc_score(y_true, probs), 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "log_loss": round(loss, 3),
    }


def metrics_for_window(name: str, n: int, quality: float, positive_rate: float) -> dict:
    y_true, probs = simulate_window(n, quality, positive_rate)
    return {"window": name, **classification_metrics(y_true, probs)}


def build_model_metrics() -> dict:
    windows = [
        metrics_for_window("training", 5000, 0.86, 0.20),
        metrics_for_window("validation", 2200, 0.84, 0.21),
        metrics_for_window("production_7d", 2400, 0.80, 0.23),
        metrics_for_window("production_30d", 7000, 0.81, 0.22),
    ]
    validation = next(item for item in windows if item["window"] == "validation")
    prod = next(item for item in windows if item["window"] == "production_7d")
    auc_delta = round(prod["auc"] - validation["auc"], 3)

    return {
        "generated_at": f"{date.today().isoformat()}T00:00:00Z",
        "model_name": "Customer Churn Risk Classifier",
        "model_version": "v1.3.0",
        "problem_type": "binary_classification",
        "target": "churn_within_30_days",
        "summary": {
            "health_status": "watch" if abs(auc_delta) > 0.03 else "ok",
            "current_auc": prod["auc"],
            "current_f1": prod["f1"],
            "current_accuracy": prod["accuracy"],
            "auc_delta_from_validation": auc_delta,
            "predictions_last_7_days": 28420,
            "retraining_recommended": auc_delta < -0.05,
        },
        "windows": windows,
        "experiments": [
            {"run_id": "exp_lr_001", "model": "Logistic Regression", "auc": 0.842, "f1": 0.693, "latency_ms": 12},
            {"run_id": "exp_rf_014", "model": "Random Forest", "auc": 0.887, "f1": 0.756, "latency_ms": 42},
            {"run_id": "exp_xgb_021", "model": "Gradient Boosted Trees", "auc": validation["auc"], "f1": validation["f1"], "latency_ms": 31},
            {"run_id": "exp_nn_006", "model": "Small Neural Network", "auc": 0.899, "f1": 0.769, "latency_ms": 55},
        ],
    }


def build_drift_report() -> dict:
    features = [
        ("monthly_charges", "numeric", 0.271, "higher than baseline"),
        ("contract_type", "categorical", 0.238, "more month-to-month contracts"),
        ("support_tickets_30d", "numeric", 0.221, "higher than baseline"),
        ("tenure_months", "numeric", 0.176, "slightly lower than baseline"),
        ("payment_method", "categorical", 0.141, "minor category shift"),
        ("internet_service", "categorical", 0.094, "within baseline range"),
    ]
    feature_rows = []
    for name, typ, score, direction in features:
        status = "watch" if score >= 0.2 else "stable"
        feature_rows.append({"feature": name, "type": typ, "drift_score": score, "status": status, "direction": direction})
    drift_score = sum(row["drift_score"] for row in feature_rows) / len(feature_rows)
    return {
        "reference_window": "validation",
        "current_window": "production_7d",
        "overall_drift_status": "watch",
        "drift_score": round(drift_score, 3),
        "thresholds": {"low": 0.1, "medium": 0.2, "high": 0.35},
        "features": feature_rows,
        "alerts": [
            {"name": "AUC degradation", "severity": "watch", "message": "Production AUC is below validation AUC. Monitor for sustained decline."},
            {"name": "Feature drift", "severity": "watch", "message": "Three features exceed the medium drift threshold."},
            {"name": "Prediction volume", "severity": "ok", "message": "Prediction volume is within the expected range."},
            {"name": "Confidence distribution", "severity": "ok", "message": "Average confidence remains stable relative to the previous production window."},
        ],
    }


def build_prediction_logs() -> dict:
    start = date.today() - timedelta(days=13)
    daily = []
    for idx in range(14):
        day = start + timedelta(days=idx)
        daily.append({
            "date": day.isoformat(),
            "predictions": int(3900 + idx * 25 + RNG.gauss(0, 120)),
            "positive_rate": round(0.19 + idx * 0.003 + RNG.gauss(0, 0.008), 3),
            "avg_confidence": round(0.812 - idx * 0.0015 + RNG.gauss(0, 0.004), 3),
        })
    return {
        "window": "last_14_days",
        "daily": daily,
        "segments": [
            {"segment": "new_customers", "volume": 8120, "positive_rate": 0.281, "avg_confidence": 0.775},
            {"segment": "month_to_month", "volume": 17480, "positive_rate": 0.318, "avg_confidence": 0.764},
            {"segment": "annual_contract", "volume": 15230, "positive_rate": 0.118, "avg_confidence": 0.842},
            {"segment": "high_support_tickets", "volume": 4910, "positive_rate": 0.386, "avg_confidence": 0.751},
        ],
    }


def build_model_card() -> dict:
    return {
        "model_name": "Customer Churn Risk Classifier",
        "version": "v1.3.0",
        "owner": "ML Platform Team",
        "last_updated": date.today().isoformat(),
        "intended_use": "Prioritize customer retention outreach by estimating the probability that an active customer may churn within 30 days.",
        "model_type": "Gradient Boosted Trees",
        "training_data": "Historical customer account, billing, support, and contract records with labels derived from 30-day churn outcomes.",
        "input_features": ["tenure_months", "monthly_charges", "contract_type", "support_tickets_30d", "payment_method", "internet_service"],
        "limitations": [
            "Predictions should support retention prioritization, not automated denial of service.",
            "Performance may degrade when pricing, support operations, or customer acquisition channels change.",
            "Segments with low historical volume require additional review before operational decisions.",
        ],
        "monitoring_requirements": [
            "Track AUC, F1, precision, recall, and log loss by production window.",
            "Monitor feature drift for billing, contract, and support-ticket features.",
            "Review prediction distribution shifts weekly.",
            "Trigger retraining review if AUC drops by more than 0.05 or if overall drift score exceeds 0.25.",
        ],
        "deployment": {"environment": "batch scoring", "cadence": "daily", "latency_target_ms": 75, "current_p95_latency_ms": 48},
    }


def main() -> None:
    write_json(DATA_DIR / "model_metrics.json", build_model_metrics())
    write_json(DATA_DIR / "drift_report.json", build_drift_report())
    write_json(DATA_DIR / "prediction_logs.json", build_prediction_logs())
    write_json(DATA_DIR / "model_card.json", build_model_card())
    print(f"Wrote MLOps artifacts to {DATA_DIR}")


if __name__ == "__main__":
    main()
