# MLOps Monitoring Dashboard

A production-style MLOps dashboard for monitoring model performance, data drift, prediction behavior, operational risk, and model-card metadata. The project is designed as a lightweight static application with reproducible Python-generated artifacts.

## Live application

After GitHub Pages is enabled with GitHub Actions as the source, the dashboard will be available at:

```text
https://hamzakaddour.github.io/mlops-monitoring-dashboard/
```

## Problem

Training a model is only one part of machine learning engineering. Once a model is deployed, teams need to monitor whether performance is stable, whether input data has shifted, whether prediction distributions changed, and whether retraining should be triggered.

This project focuses on the monitoring layer of an ML system: metrics, drift, prediction logs, alerts, and model-card reporting.

## Core capabilities

- Model performance tracking across training, validation, and production windows.
- Data drift analysis using feature-level drift scores.
- Prediction distribution monitoring.
- Alert summary for production risk indicators.
- Model card with intended use, limitations, metrics, and deployment notes.
- Lightweight artifact generation with Python.
- Static dashboard deployment with GitHub Pages.

## Architecture

```text
data/
  model_metrics.json       Performance metrics and experiment summary
  drift_report.json        Feature drift and alert status
  prediction_logs.json     Production prediction monitoring sample
  model_card.json          Structured model card metadata

scripts/
  generate_mlops_artifacts.py

css/
  styles.css

js/
  app.js

index.html
.github/workflows/pages.yml
```

## Local run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_mlops_artifacts.py
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

## Production extensions

A production version could connect to MLflow, Evidently, Prometheus, OpenTelemetry, a feature store, a model registry, and scheduled retraining workflows.

## Topics

```text
mlops, model-monitoring, data-drift, model-card, machine-learning, scikit-learn, github-actions, observability, data-science, python
```
