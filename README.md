# MLOps Monitoring Dashboard

A production-style machine learning monitoring project that tracks **model performance, data drift, prediction behavior, operational risk, and model-card metadata** through a reproducible Python pipeline, interactive dashboard, and FastAPI service.

The project now covers the full local lifecycle: **candidate training → MLflow experiment tracking → model selection → saved-model inference → prediction logging → drift/performance monitoring → retraining recommendation**. The public dashboard also surfaces the committed workstation training summary so model-selection results and monitoring behavior can be inspected together.

## What this project demonstrates

- Reproducible scikit-learn candidate training and validation
- Local MLflow experiment tracking and model comparison
- Saved-model FastAPI inference with `POST /predict`
- SQLite prediction-event logging
- Rule-based retraining recommendation via `GET /retraining-status`
- Model performance monitoring across training, validation, and production windows
- Feature-level data-drift analysis
- Evidently DataDriftPreset report generation for reference vs. current datasets
- Prometheus-format inference counters, gauges, and latency histograms
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
   /predict             index.html
   /prediction-events   JavaScript + CSS
   /training-summary    workstation training result
   /metrics             monitoring artifacts
   /drift
   /retraining-status
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

This trains multiple candidate classifiers on a deterministic synthetic churn dataset, records each run in a local MLflow SQLite backend, selects the best candidate by validation ROC-AUC, and writes:

```text
artifacts/model.joblib
artifacts/training_summary.json
mlflow.db
```

The binary model, local MLflow SQLite database, and generated MLflow artifacts are intentionally ignored by Git. The JSON training summary can be committed as a reproducible workstation result.

To inspect the experiments locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
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
GET /training-summary
POST /predict
GET /prometheus-metrics
GET /evidently-summary
GET /observability-summary
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

## Native Ubuntu workflow

Docker is optional. The complete development workflow can run directly in a Python virtual environment:

```bash
source .venv/bin/activate
python scripts/train_model.py
python scripts/generate_mlops_artifacts.py
python scripts/generate_evidently_report.py
uvicorn api.app:app --reload --port 8000
```

In another terminal, serve the static dashboard:

```bash
python -m http.server 8080
```

The API is available at `http://127.0.0.1:8000`, the dashboard at `http://127.0.0.1:8080`, and MLflow can be launched separately with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Prometheus-format metrics are available without installing a Prometheus server:

```bash
curl http://127.0.0.1:8000/prometheus-metrics | grep mlops_
```

These counters are process-local and reset when the FastAPI process restarts unless an external Prometheus server persists them.

## Optional Docker workflow

Docker/Compose is included for reproducibility and for users who want a one-command API + Prometheus + Grafana stack. It is not required for the native workflow.

Build and run only the monitoring API:

```bash
docker build -t mlops-monitoring-dashboard .
docker run -p 8000:8000 mlops-monitoring-dashboard
```

Then open:

```text
http://localhost:8000/docs
```

## Published workstation result

The repository includes a sanitized `artifacts/training_summary.json` generated on the workstation. It records the selected candidate, validation ROC-AUC, held-out test metrics, dataset split sizes, and MLflow run IDs without publishing local filesystem paths or the binary model.

Current committed result:

- selected candidate: `logistic_regression`
- validation ROC-AUC: `0.7443`
- held-out test ROC-AUC: `0.7487`
- held-out test F1: `0.5329`
- rows: `6000` synthetic examples

The moderate model score is intentional: the repository is designed to demonstrate the MLOps lifecycle and monitoring behavior rather than manufacture an unrealistically easy benchmark.

## Drift monitoring

The `mlops_monitoring.drift` module implements **Population Stability Index (PSI)** for numeric and categorical distributions.

Monitoring states are mapped as:

```text
PSI < 0.10          stable
0.10 <= PSI < 0.25 watch
PSI >= 0.25         alert
```

The utilities are intentionally kept separate from the UI so they can be reused in batch jobs, APIs, scheduled monitoring workflows, or future integrations with production data stores.

## Evidently and Prometheus observability

Generate the Evidently drift report locally:

```bash
python scripts/generate_evidently_report.py
```

This writes:

```text
data/evidently_drift_report.html
data/evidently_drift_summary.json
```

The report compares deterministic reference data against a controlled shifted monitoring window using Evidently's `DataDriftPreset` with PSI.

With the FastAPI service running, inspect the structured report at:

```text
GET /evidently-summary
```

Prometheus instrumentation is emitted at:

```text
GET /prometheus-metrics
```

The current metrics include total prediction requests, total positive predictions, the most recent prediction probability, and an inference-latency histogram. The endpoint is compatible with a Prometheus scraper without requiring a Prometheus server for local development.

## Full local observability stack

A Docker Compose stack is included for the serving API, Prometheus, and Grafana:

```bash
docker compose up --build
```

Services:

```text
FastAPI      http://127.0.0.1:8000
Prometheus   http://127.0.0.1:9090
Grafana      http://127.0.0.1:3000
```

Grafana is provisioned automatically with Prometheus as its datasource and includes the **MLOps Inference Observability** dashboard. The dashboard shows prediction requests, positive predictions, last churn probability, and p95 inference latency.

The API container trains the deterministic model and generates its monitoring artifacts at image-build time, so `POST /predict` works without copying workstation-only model files into the repository.

The static frontend also reads `data/observability_summary.json` and shows the verified Evidently drift result alongside the Prometheus instrumentation.

## Model monitoring signals

The dashboard currently exposes the real workstation training result alongside explicitly simulated monitoring windows. It includes:

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

The training, model serving, SQLite prediction logging, Evidently analysis, Prometheus instrumentation, tests, and CI are implemented. The monitoring-window observations are deliberately simulated and labeled as such. This separation keeps the repository reproducible while demonstrating how the same monitoring layer can be connected to real inference logs and production datasets.

## Scope boundary

This repository is intentionally stopped at a complete local-first MLOps demonstration. Possible production extensions would include a managed model registry, persistent Prometheus/Grafana deployment, Alertmanager, scheduled monitoring jobs, authentication, and cloud infrastructure, but those are outside the current portfolio scope.

## Tech stack

`Python` · `FastAPI` · `NumPy` · `Pandas` · `scikit-learn` · `MLflow` · `Evidently` · `Prometheus` · `SQLite` · `joblib` · `pytest` · `Docker` · `GitHub Actions` · `HTML/CSS/JavaScript` · `MLOps` · `Model Monitoring` · `Data Drift`
