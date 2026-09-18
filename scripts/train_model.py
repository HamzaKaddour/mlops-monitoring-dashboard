"""Train candidate churn models, track experiments with MLflow, and save the best model.

This script is intended to run on the workstation. It uses a deterministic
synthetic dataset so the full workflow is reproducible and free of proprietary
data.

Outputs:
- artifacts/model.joblib
- artifacts/training_summary.json
- local MLflow runs under ./mlruns

Run:
    python scripts/train_model.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from mlops_monitoring.training import (
    FEATURES,
    TARGET,
    build_pipeline,
    candidate_models,
    classification_metrics,
    make_synthetic_churn_data,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
MLRUNS_DIR = ROOT / "mlruns"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    frame = make_synthetic_churn_data()
    X = frame[FEATURES]
    y = frame[TARGET]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.40,
        random_state=42,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment("churn-model-selection")

    results = []
    best_name = None
    best_model = None
    best_auc = float("-inf")

    for name, estimator in candidate_models().items():
        model = build_pipeline(estimator)
        model.fit(X_train, y_train)

        val_metrics = classification_metrics(model, X_val, y_val)

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(
                {
                    "candidate": name,
                    "train_rows": len(X_train),
                    "validation_rows": len(X_val),
                    "test_rows": len(X_test),
                    "seed": 42,
                }
            )
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.sklearn.log_model(model, artifact_path="model")

            result = {
                "run_id": run.info.run_id,
                "candidate": name,
                "validation_metrics": val_metrics,
            }
            results.append(result)

        if val_metrics["roc_auc"] > best_auc:
            best_auc = val_metrics["roc_auc"]
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None

    test_metrics = classification_metrics(best_model, X_test, y_test)
    model_version = "v1.0.0"

    joblib.dump(best_model, ARTIFACT_DIR / "model.joblib")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "Customer Churn Risk Classifier",
        "model_version": model_version,
        "selected_candidate": best_name,
        "selection_metric": "validation_roc_auc",
        "validation_roc_auc": round(best_auc, 4),
        "test_metrics": test_metrics,
        "dataset": {
            "source": "deterministic synthetic churn scenario",
            "rows": len(frame),
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": len(X_test),
            "features": FEATURES,
            "target": TARGET,
        },
        "mlflow": {
            "tracking_uri": MLRUNS_DIR.resolve().as_uri(),
            "experiment_name": "churn-model-selection",
        },
        "candidate_runs": results,
    }
    (ARTIFACT_DIR / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSaved model: {ARTIFACT_DIR / 'model.joblib'}")
    print(f"Saved summary: {ARTIFACT_DIR / 'training_summary.json'}")
    print(f"MLflow runs: {MLRUNS_DIR}")


if __name__ == "__main__":
    main()
