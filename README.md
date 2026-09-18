# MLOps Monitoring Dashboard

A production-style machine learning monitoring project that tracks **model performance, data drift, prediction behavior, operational risk, and model-card metadata** through a reproducible Python pipeline, interactive dashboard, and FastAPI service.

The project now covers the full local lifecycle: **candidate training → MLflow experiment tracking → model selection → saved-model inference → prediction logging → drift/performance monitoring → retraining recommendation**.

## What this project demonstrates

- Reproducible scikit-learn candidate training and validation
- Local MLflow experiment tracking and model comparison
- Saved-model FastAPI inference with `POST /predict`
- SQLite prediction-event logging
- Rule-based retraining recommendation via `GET /retraining-status`
- Model performance monitoring across training, validation, and production windows
- Feature-level data-drift analysis
- Population Stability Index (PSI) utilities for numeric and categorical features
- Prediction-distribution and confidence monitoring
- Alert generation for production risk indicators
- Structured model-card metadata and deployment constraints
- FastAPI endpoints for programmatic access to monitoring artifacts
- Dockerized local deployment
- Automated tests with `pytest`
- GitHub Actions CI on pushes and pull requests
- Static monitoring dashboard suitable for GitHub Pages

## Architecture

```text
Reference / production data
          |
          v
Monitoring artifact generator
          |
          +--> model_metrics.json
          +--> drift_report.json
          +--> prediction_logs.json
          +--> model_card.json
          |
          +-------------------+
          |                   |
          v                   v
   FastAPI service      Static dashboard
   /metrics             index.html
   /drift               JavaScript + CSS
   /predictions
   /model-card
          |
          v
   Tests + GitHub Actions CI
```

## Repository structure

```text
api/
  app.py                         FastAPI monitoring service

mlops_monitoring/
  drift.py                       Reusable PSI drift metrics

scripts/
  generate_mlops_artifacts.py    Reproducible monitoring artifact generator

data/
  model_metrics.json             Performance windows and experiment summary
  drift_report.json              Drift status and feature alerts
  prediction_logs.json           Production prediction monitoring sample
  model_card.json                Structured model metadata

tests/
  test_drift.py                  Drift metric tests
  test_api.py                    API tests

.github/workflows/
  ci.yml                         Automated test pipeline

css/                             Dashboard styling
js/                              Dashboard behavior
index.html                       Static dashboard
Dockerfile                       Containerized API
requirements.txt
```

## Run locally

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model and record experiments

```bash
python scripts/train_model.py
```

This trains multiple candidate classifiers on a deterministic synthetic churn dataset, records each run in a local MLflow file store, selects the best candidate by validation ROC-AUC, and writes:

```text
artifacts/model.joblib
artifacts/training_summary.json
mlruns/
```

The binary model and local MLflow directory are intentionally ignored by Git. The JSON training summary can be committed as a reproducible workstation result.

To inspect the experiments locally:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Then open `http://127.0.0.1:5000`.

### 3. Generate monitoring artifacts

```bash
python scripts/generate_mlops_artifacts.py
```

### 4. Run the dashboard

```bash
python -m http.server 8080
```

Open:

```text
http://localhost:8080
```

## Run the monitoring API

```bash
uvicorn api.app:app --reload
```

Useful endpoints:

```text
GET /health
POST /predict
GET /prediction-events
GET /retraining-status
GET /metrics
GET /drift
GET /predictions
GET /model-card
```

Interactive OpenAPI documentation is available at:

```text
http://localhost:8000/docs
```

## Docker

Build and run the monitoring API:

```bash
docker build -t mlops-monitoring-dashboard .
docker run -p 8000:8000 mlops-monitoring-dashboard
```

Then open:

```text
http://localhost:8000/docs
```

## Drift monitoring

The `mlops_monitoring.drift` module implements **Population Stability Index (PSI)** for numeric and categorical distributions.

Monitoring states are mapped as:

```text
PSI < 0.10          stable
0.10 <= PSI < 0.25 watch
PSI >= 0.25         alert
```

The utilities are intentionally kept separate from the UI so they can be reused in batch jobs, APIs, scheduled monitoring workflows, or future integrations with production data stores.

## Model monitoring signals

The dashboard currently exposes:

- ROC-AUC
- F1 score
- accuracy
- precision
- recall
- log loss
- feature drift
- prediction volume
- positive prediction rate
- average confidence
- segment-level behavior
- alert severity
- retraining recommendation
- deployment latency metadata

## Testing and CI

Run tests locally with:

```bash
pytest -q
```

GitHub Actions automatically:

1. installs project dependencies,
2. regenerates monitoring artifacts,
3. executes the unit/API test suite.

This provides a basic CI safety net around monitoring logic and API behavior.

## Current scope

The repository uses a **deterministic synthetic binary-classification scenario** so the full monitoring workflow can be reproduced without proprietary data or external infrastructure.

The monitoring architecture and PSI implementation are real; the example production observations are simulated. This separation keeps the repository reproducible while demonstrating how the same monitoring layer can be connected to real inference logs and production datasets.

## Production roadmap

Natural extensions include:

- MLflow model registry promotion workflow
- Evidently monitoring reports
- Prometheus metrics and alerting
- OpenTelemetry tracing
- persistent prediction logging
- scheduled drift jobs
- model/version comparison
- automated retraining triggers
- cloud deployment on AWS
- authentication and role-based dashboard access

## Tech stack

`Python` · `FastAPI` · `NumPy` · `Pandas` · `scikit-learn` · `MLflow` · `SQLite` · `joblib` · `pytest` · `Docker` · `GitHub Actions` · `HTML/CSS/JavaScript` · `MLOps` · `Model Monitoring` · `Data Drift`
