"""Generate reproducible monitoring artifacts for the selected churn model.

The trained model and held-out metrics come from artifacts/training_summary.json.
The production-like monitoring windows are intentionally simulated so the
repository can demonstrate drift, performance degradation, alerts, and
retraining logic without proprietary production data.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAINING_SUMMARY = ROOT / "artifacts" / "training_summary.json"
RNG = random.Random(7)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_training_summary() -> dict:
    if not TRAINING_SUMMARY.exists():
        raise FileNotFoundError(
            "Training summary is missing. Run: python scripts/train_model.py"
        )
    return json.loads(TRAINING_SUMMARY.read_text(encoding="utf-8"))


def bounded(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def build_model_metrics(training: dict) -> dict:
    test = training["test_metrics"]
    validation_auc = float(training["validation_roc_auc"])

    # Synthetic monitoring windows derived from the real held-out result.
    selected_run = next(\n        run for run in training["candidate_runs"]\n        if run["candidate"] == training["selected_candidate"]\n    )\n    selected_validation = selected_run["validation_metrics"]\n\n    validation = {\n        "window": "validation",\n        "auc": round(validation_auc, 4),\n        "f1": round(float(selected_validation["f1"]), 4),\n        "accuracy": round(float(selected_validation["accuracy"]), 4),\n        "precision": round(float(selected_validation["precision"]), 4),\n        "recall": round(float(selected_validation["recall"]), 4),\n        "log_loss": round(float(selected_validation["log_loss"]), 4),\n    }
    test_window = {
        "window": "held_out_test",
        "auc": round(float(test["roc_auc"]), 4),
        "f1": round(float(test["f1"]), 4),
        "accuracy": round(float(test["accuracy"]), 4),
        "precision": round(float(test["precision"]), 4),
        "recall": round(float(test["recall"]), 4),
        "log_loss": round(float(test["log_loss"]), 4),
    }

    prod7 = {
        "window": "simulated_monitoring_7d",
        "auc": bounded(float(test["roc_auc"]) - 0.034),
        "f1": bounded(float(test["f1"]) - 0.031),
        "accuracy": bounded(float(test["accuracy"]) - 0.018),
        "precision": bounded(float(test["precision"]) - 0.024),
        "recall": bounded(float(test["recall"]) - 0.029),
        "log_loss": round(float(test["log_loss"]) + 0.041, 4),
    }
    prod30 = {
        "window": "simulated_monitoring_30d",
        "auc": bounded(float(test["roc_auc"]) - 0.025),
        "f1": bounded(float(test["f1"]) - 0.022),
        "accuracy": bounded(float(test["accuracy"]) - 0.012),
        "precision": bounded(float(test["precision"]) - 0.017),
        "recall": bounded(float(test["recall"]) - 0.021),
        "log_loss": round(float(test["log_loss"]) + 0.030, 4),
    }

    auc_delta = round(prod7["auc"] - validation["auc"], 4)

    return {
        "generated_at": f"{date.today().isoformat()}T00:00:00Z",
        "model_name": training["model_name"],
        "model_version": training["model_version"],
        "selected_candidate": training["selected_candidate"],
        "problem_type": "binary_classification",
        "target": training["dataset"]["target"],
        "data_scope": "synthetic monitoring demonstration",
        "summary": {
            "health_status": "watch" if auc_delta <= -0.03 else "ok",
            "current_auc": prod7["auc"],
            "current_f1": prod7["f1"],
            "current_accuracy": prod7["accuracy"],
            "auc_delta_from_validation": auc_delta,
            "predictions_last_7_days": 28420,
            "retraining_recommended": auc_delta <= -0.05,
        },
        "windows": [validation, test_window, prod7, prod30],
        "experiments": [
            {
                "run_id": run["run_id"],
                "model": run["candidate"],
                "auc": run["validation_metrics"]["roc_auc"],
                "f1": run["validation_metrics"]["f1"],
            }
            for run in training["candidate_runs"]
        ],
    }


def build_drift_report() -> dict:
    features = [
        ("monthly_charges", "numeric", 0.151068, "higher than reference"),
        ("contract_type", "categorical", 0.147821, "more month-to-month contracts"),
        ("support_tickets_30d", "numeric", 1.879992, "higher than reference"),
        ("tenure_months", "numeric", 0.011946, "within reference range"),
        ("payment_method", "categorical", 0.000615, "within reference range"),
        ("internet_service", "categorical", 0.001788, "within reference range"),
    ]
    feature_rows = []
    for name, typ, score, direction in features:
        status = "alert" if score >= 0.25 else "watch" if score >= 0.10 else "stable"
        feature_rows.append(
            {
                "feature": name,
                "type": typ,
                "drift_score": score,
                "status": status,
                "direction": direction,
            }
        )

    drift_score = sum(row["drift_score"] for row in feature_rows) / len(feature_rows)
    return {
        "reference_window": "synthetic_reference",
        "current_window": "synthetic_shifted_current",
        "overall_drift_status": "watch",
        "drift_score": round(drift_score, 4),
        "thresholds": {"stable": 0.10, "watch": 0.25},
        "features": feature_rows,
        "alerts": [
            {
                "name": "AUC degradation",
                "severity": "watch",
                "message": "The simulated 7-day monitoring AUC is below validation AUC.",
            },
            {
                "name": "Feature drift",
                "severity": "watch",
                "message": "Evidently flags three of six monitored features as drifted.",
            },
            {
                "name": "Prediction volume",
                "severity": "ok",
                "message": "Simulated prediction volume remains within the configured demonstration range.",
            },
            {
                "name": "Inference telemetry",
                "severity": "ok",
                "message": "Prometheus instrumentation is available from the serving API.",
            },
        ],
    }


def build_prediction_logs() -> dict:
    start = date.today() - timedelta(days=13)
    daily = []
    for idx in range(14):
        day = start + timedelta(days=idx)
        daily.append(
            {
                "date": day.isoformat(),
                "predictions": int(3900 + idx * 25 + RNG.gauss(0, 120)),
                "positive_rate": round(0.19 + idx * 0.003 + RNG.gauss(0, 0.008), 3),
                "avg_confidence": round(0.812 - idx * 0.0015 + RNG.gauss(0, 0.004), 3),
            }
        )
    return {
        "window": "simulated_last_14_days",
        "scope": "synthetic monitoring demonstration",
        "daily": daily,
        "segments": [
            {"segment": "new_customers", "volume": 8120, "positive_rate": 0.281, "avg_confidence": 0.775},
            {"segment": "month_to_month", "volume": 17480, "positive_rate": 0.318, "avg_confidence": 0.764},
            {"segment": "annual_contract", "volume": 15230, "positive_rate": 0.118, "avg_confidence": 0.842},
            {"segment": "high_support_tickets", "volume": 4910, "positive_rate": 0.386, "avg_confidence": 0.751},
        ],
    }


def build_model_card(training: dict) -> dict:
    return {
        "model_name": training["model_name"],
        "version": training["model_version"],
        "owner": "Portfolio demonstration",
        "last_updated": date.today().isoformat(),
        "intended_use": (
            "Demonstrate an end-to-end MLOps lifecycle using a churn-risk classifier, "
            "including experiment tracking, serving, monitoring, and retraining signals."
        ),
        "model_type": "Logistic Regression",
        "training_data": (
            "Deterministic synthetic churn dataset with 6,000 rows; "
            "3,600 train, 1,200 validation, and 1,200 held-out test rows."
        ),
        "input_features": training["dataset"]["features"],
        "limitations": [
            "The dataset and monitoring windows are synthetic and are not evidence of real-world production performance.",
            "The repository demonstrates MLOps system behavior rather than a domain-validated churn model.",
            "Prometheus counters reset when the local API process restarts unless an external Prometheus server persists them.",
        ],
        "monitoring_requirements": [
            "Track ROC-AUC, F1, precision, recall, and log loss across validation and monitoring windows.",
            "Run Evidently feature-drift checks against the reference distribution.",
            "Scrape inference counters, probability gauges, and latency histograms from Prometheus metrics.",
            "Review retraining when AUC falls by at least 0.05 or PSI drift crosses configured thresholds.",
        ],
        "deployment": {
            "environment": "local FastAPI demo",
            "cadence": "on demand",
            "latency_target_ms": None,
            "current_p95_latency_ms": None,
        },
    }


def main() -> None:
    training = load_training_summary()
    write_json(DATA_DIR / "model_metrics.json", build_model_metrics(training))
    write_json(DATA_DIR / "drift_report.json", build_drift_report())
    write_json(DATA_DIR / "prediction_logs.json", build_prediction_logs())
    write_json(DATA_DIR / "model_card.json", build_model_card(training))
    print(f"Wrote MLOps artifacts to {DATA_DIR}")


if __name__ == "__main__":
    main()
