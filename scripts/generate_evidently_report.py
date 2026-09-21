"""Generate an Evidently data-drift report from reproducible tabular data.

The reference dataset uses the same deterministic synthetic churn generator as
training. The current dataset applies controlled distribution shifts to several
features so the report has meaningful drift to inspect.

Outputs:
- data/evidently_drift_report.html
- data/evidently_drift_summary.json
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

    print(f"Saved Evidently HTML report: {html_path}")
    print(f"Saved Evidently JSON summary: {json_path}")


if __name__ == "__main__":
    main()
