"""Generate an Evidently data-drift report from reproducible tabular data.

The reference dataset uses the same deterministic synthetic churn generator as
training. The current dataset applies controlled distribution shifts to several
features so the report has meaningful drift to inspect.

Outputs:
- data/evidently_drift_report.html
- data/evidently_drift_summary.json
- data/observability_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

from mlops_monitoring.training import FEATURES, make_synthetic_churn_data

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def make_monitoring_frames():
    reference = make_synthetic_churn_data(n=2500, seed=42)[FEATURES].copy()
    current = make_synthetic_churn_data(n=1800, seed=99)[FEATURES].copy()

    # Controlled production-like shifts.
    current["monthly_charges"] = (current["monthly_charges"] * 1.12).clip(upper=180)
    current["support_tickets_30d"] = (current["support_tickets_30d"] + 1).clip(upper=12)

    one_year_mask = current["contract_type"].eq("one_year")
    indices = current.index[one_year_mask][::2]
    current.loc[indices, "contract_type"] = "month_to_month"

    return reference, current


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reference, current = make_monitoring_frames()

    report = Report([DataDriftPreset(method="psi")], include_tests=True)
    snapshot = report.run(current, reference)

    html_path = DATA_DIR / "evidently_drift_report.html"
    json_path = DATA_DIR / "evidently_drift_summary.json"

    snapshot.save_html(str(html_path))
    payload = snapshot.dict()
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    feature_scores = {}
    drifted_count = 0
    drifted_share = 0.0
    for metric in payload.get("metrics", []):
        config = metric.get("config", {})
        metric_type = config.get("type", "")
        if metric_type.endswith("DriftedColumnsCount"):
            value = metric.get("value", {})
            drifted_count = int(value.get("count", 0))
            drifted_share = float(value.get("share", 0.0))
        elif metric_type.endswith("ValueDrift"):
            column = config.get("column")
            if column:
                feature_scores[column] = round(float(metric.get("value", 0.0)), 6)

    observability = {
        "evidently": {
            "method": "PSI",
            "threshold": 0.1,
            "total_features": len(FEATURES),
            "drifted_features": drifted_count,
            "drifted_share": drifted_share,
            "status": "drift_detected" if drifted_count else "stable",
            "feature_scores": feature_scores,
            "html_report": "data/evidently_drift_report.html",
        },
        "prometheus": {
            "endpoint": "/prometheus-metrics",
            "metrics": [
                "mlops_prediction_requests_total",
                "mlops_positive_predictions_total",
                "mlops_last_prediction_probability",
                "mlops_inference_latency_seconds",
            ],
        },
    }
    observability_path = DATA_DIR / "observability_summary.json"
    observability_path.write_text(
        json.dumps(observability, indent=2),
        encoding="utf-8",
    )

    print(f"Saved Evidently HTML report: {html_path}")
    print(f"Saved Evidently JSON summary: {json_path}")
    print(f"Saved observability summary: {observability_path}")


if __name__ == "__main__":
    main()
