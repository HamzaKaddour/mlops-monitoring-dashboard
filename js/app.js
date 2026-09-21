const state = { metrics: null, drift: null, predictions: null, modelCard: null, training: null, observability: null };

const baseLayout = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#111827", family: "Inter, system-ui, sans-serif" },
  margin: { t: 18, r: 18, b: 46, l: 52 },
  xaxis: { gridcolor: "rgba(148,163,184,0.22)", zerolinecolor: "rgba(148,163,184,0.24)", tickfont: { color: "#64748b" } },
  yaxis: { gridcolor: "rgba(148,163,184,0.22)", zerolinecolor: "rgba(148,163,184,0.24)", tickfont: { color: "#64748b" } },
  legend: { orientation: "h", y: -0.22, font: { color: "#334155" } }
};

async function loadData() {
  const [metricsRes, driftRes, predictionsRes, cardRes, trainingRes, observabilityRes] = await Promise.all([
    fetch("data/model_metrics.json"),
    fetch("data/drift_report.json"),
    fetch("data/prediction_logs.json"),
    fetch("data/model_card.json"),
    fetch("artifacts/training_summary.json"),
    fetch("data/observability_summary.json")
  ]);
  state.metrics = await metricsRes.json();
  state.drift = await driftRes.json();
  state.predictions = await predictionsRes.json();
  state.modelCard = await cardRes.json();
  state.training = await trainingRes.json();
  state.observability = await observabilityRes.json();

  renderSidebar();
  renderSummary();
  renderTraining();
  renderCharts();
  renderAlerts();
  renderObservability();
  renderModelCard();
}

function pct(value) { return `${Math.round(value * 100)}%`; }
function num(value) { return Number(value).toLocaleString(); }

function renderSidebar() {
  document.getElementById("sidebarModelName").textContent = state.metrics.model_name;
  document.getElementById("sidebarModelVersion").textContent = `${state.metrics.model_version} · ${state.metrics.problem_type.replaceAll("_", " ")}`;
}

function renderSummary() {
  const s = state.metrics.summary;
  const cards = [
    ["Health status", s.health_status.toUpperCase(), "Current production state"],
    ["Production AUC", s.current_auc.toFixed(3), `${s.auc_delta_from_validation.toFixed(3)} vs validation`],
    ["Production F1", s.current_f1.toFixed(3), "Current production window"],
    ["Predictions / 7d", num(s.predictions_last_7_days), "Observed scoring volume"]
  ];
  document.getElementById("summaryGrid").innerHTML = cards.map(([label, value, note]) => `
    <article class="kpi-card">
      <div class="kpi-label">${label}</div>
      <div class="kpi-value">${value}</div>
      <div class="kpi-note">${note}</div>
    </article>`).join("");
}


function prettyName(value) {
  return String(value).replaceAll("_", " ").replace(/\b\w/g, ch => ch.toUpperCase());
}

function renderTraining() {
  const t = state.training;
  if (!t) return;

  document.getElementById("trainingGeneratedAt").textContent =
    new Date(t.generated_at).toLocaleString();

  const test = t.test_metrics;
  const kpis = [
    ["Selected model", prettyName(t.selected_candidate), "Chosen by validation ROC-AUC"],
    ["Validation ROC-AUC", t.validation_roc_auc.toFixed(4), "Model-selection metric"],
    ["Test ROC-AUC", test.roc_auc.toFixed(4), "Held-out evaluation"],
    ["Test F1", test.f1.toFixed(4), "Thresholded classification"],
    ["Test accuracy", pct(test.accuracy), "Held-out test set"],
    ["Dataset", num(t.dataset.rows), "Synthetic, deterministic rows"]
  ];

  document.getElementById("trainingKpis").innerHTML = kpis.map(([label, value, note]) => `
    <article class="mini-kpi">
      <div class="kpi-label">${label}</div>
      <div class="mini-kpi-value">${value}</div>
      <div class="kpi-note">${note}</div>
    </article>
  `).join("");

  document.getElementById("candidateTable").innerHTML = t.candidate_runs.map(run => {
    const m = run.validation_metrics;
    const selected = run.candidate === t.selected_candidate;
    return `
      <tr class="${selected ? "selected-row" : ""}">
        <td><strong>${prettyName(run.candidate)}</strong>${selected ? '<span class="selected-badge">selected</span>' : ""}</td>
        <td>${m.roc_auc.toFixed(4)}</td>
        <td>${m.f1.toFixed(4)}</td>
        <td>${pct(m.accuracy)}</td>
        <td>${m.log_loss.toFixed(4)}</td>
      </tr>
    `;
  }).join("");

  const summary = state.metrics.summary;
  const drift = state.drift;
  const highDrift = drift.features.filter(f => Number(f.drift_score) >= 0.25).length;
  const retrain = summary.auc_delta_from_validation <= -0.05 ||
                  Number(drift.drift_score) >= 0.25 ||
                  highDrift >= 2;
  const status = retrain ? "RETRAIN" :
    (summary.auc_delta_from_validation <= -0.03 || Number(drift.drift_score) >= 0.10 ? "WATCH" : "HEALTHY");

  document.getElementById("retrainingPanel").innerHTML = `
    <span class="retraining-status ${status.toLowerCase()}">${status}</span>
    <div>
      <strong>Retraining recommendation</strong>
      <p>${retrain ? "Current monitoring signals cross the retraining policy." : "Continue monitoring; current signals do not cross the retraining threshold."}</p>
    </div>
  `;
}

function renderCharts() {
  const windows = state.metrics.windows.map(w => w.window.replaceAll("_", " "));
  Plotly.newPlot("performanceChart", [
    { x: windows, y: state.metrics.windows.map(w => w.auc), name: "AUC", type: "scatter", mode: "lines+markers", line: { width: 3, shape: "spline" }, marker: { size: 8 } },
    { x: windows, y: state.metrics.windows.map(w => w.f1), name: "F1", type: "scatter", mode: "lines+markers", line: { width: 3, shape: "spline" }, marker: { size: 8 } },
    { x: windows, y: state.metrics.windows.map(w => w.accuracy), name: "Accuracy", type: "scatter", mode: "lines+markers", line: { width: 3, shape: "spline" }, marker: { size: 8 } }
  ], { ...baseLayout, yaxis: { ...baseLayout.yaxis, range: [0.65, 0.95] } }, { responsive: true, displayModeBar: false });

  const sortedFeatures = [...state.drift.features].sort((a, b) => a.drift_score - b.drift_score);
  Plotly.newPlot("driftChart", [{
    x: sortedFeatures.map(f => f.drift_score),
    y: sortedFeatures.map(f => f.feature.replaceAll("_", " ")),
    type: "bar",
    orientation: "h",
    text: sortedFeatures.map(f => f.status),
    hovertext: sortedFeatures.map(f => f.direction)
  }], { ...baseLayout, xaxis: { ...baseLayout.xaxis, title: "Drift score" }, yaxis: { ...baseLayout.yaxis, automargin: true }, margin: { t: 12, r: 16, b: 42, l: 120 } }, { responsive: true, displayModeBar: false });

  const daily = state.predictions.daily;
  Plotly.newPlot("predictionChart", [
    { x: daily.map(d => d.date), y: daily.map(d => d.predictions), name: "Prediction volume", type: "bar", yaxis: "y" },
    { x: daily.map(d => d.date), y: daily.map(d => d.positive_rate), name: "Positive rate", type: "scatter", mode: "lines+markers", yaxis: "y2", line: { width: 3 } },
    { x: daily.map(d => d.date), y: daily.map(d => d.avg_confidence), name: "Avg confidence", type: "scatter", mode: "lines+markers", yaxis: "y3", line: { width: 3, dash: "dot" } }
  ], {
    ...baseLayout,
    yaxis: { title: "Volume", gridcolor: "rgba(148,163,184,0.22)" },
    yaxis2: { title: "Positive rate", overlaying: "y", side: "right", range: [0, 0.4], gridcolor: "rgba(0,0,0,0)", tickfont: { color: "#64748b" } },
    yaxis3: { visible: false, overlaying: "y", range: [0.7, 0.9] }
  }, { responsive: true, displayModeBar: false });
}

function renderAlerts() {
  document.getElementById("alertList").innerHTML = state.drift.alerts.map(alert => {
    const cls = alert.severity === "ok" ? "alert-ok" : alert.severity === "critical" ? "alert-critical" : "alert-watch";
    return `<article class="alert-item"><div><strong>${alert.name}</strong><span>${alert.message}</span></div><span class="alert-badge ${cls}">${alert.severity.toUpperCase()}</span></article>`;
  }).join("");
}


function renderObservability() {
  const obs = state.observability;
  if (!obs) return;

  const ev = obs.evidently;
  const statusEl = document.getElementById("evidentlyStatus");
  statusEl.textContent = ev.status === "drift_detected" ? "DRIFT DETECTED" : "STABLE";
  statusEl.className = "status-chip " + (ev.status === "drift_detected" ? "watch" : "stable");

  document.getElementById("evidentlyKpis").innerHTML = [
    ["Drifted features", ev.drifted_features, `of ${ev.total_features}`],
    ["Drift share", pct(ev.drifted_share), `PSI threshold ${ev.threshold}`],
    ["Method", ev.method, "Evidently DataDriftPreset"]
  ].map(([label, value, note]) => `
    <div class="obs-kpi">
      <span>${label}</span>
      <strong>${value}</strong>
      <small>${note}</small>
    </div>
  `).join("");

  const sorted = Object.entries(ev.feature_scores).sort((a, b) => b[1] - a[1]);
  document.getElementById("evidentlyFeatureList").innerHTML = sorted.map(([name, score]) => {
    const drifted = score >= ev.threshold;
    return `
      <div class="feature-row">
        <span>${prettyName(name)}</span>
        <div class="feature-score-wrap">
          <div class="feature-bar"><i style="width:${Math.min(100, score / Math.max(ev.threshold, 0.001) * 35)}%"></i></div>
          <strong class="${drifted ? "score-alert" : ""}">${score.toFixed(3)}</strong>
        </div>
      </div>
    `;
  }).join("");

  const prom = obs.prometheus;
  const example = prom.verified_example || {};
  const rows = [
    ["Prediction requests", example.prediction_requests_total ?? "—", "mlops_prediction_requests_total"],
    ["Positive predictions", example.positive_predictions_total ?? "—", "mlops_positive_predictions_total"],
    ["Last probability", example.last_prediction_probability != null ? example.last_prediction_probability.toFixed(6) : "—", "mlops_last_prediction_probability"],
    ["Observed latency", example.inference_latency_seconds != null ? example.inference_latency_seconds.toFixed(3) + " s" : "—", "mlops_inference_latency_seconds"]
  ];
  document.getElementById("prometheusMetrics").innerHTML = rows.map(([label, value, metric]) => `
    <div class="metric-row">
      <div><strong>${label}</strong><code>${metric}</code></div>
      <span>${value}</span>
    </div>
  `).join("");
}

function renderModelCard() {
  const card = state.modelCard;
  const items = [
    ["Model", `${card.model_name} (${card.version})`],
    ["Type", card.model_type],
    ["Intended use", card.intended_use],
    ["Training data", card.training_data],
    ["Inputs", card.input_features.join(", ")],
    ["Deployment", `${card.deployment.environment}, ${card.deployment.cadence}; p95 latency ${card.deployment.current_p95_latency_ms} ms`],
    ["Limitations", card.limitations.join(" ")],
    ["Monitoring", card.monitoring_requirements.join(" ")]
  ];
  document.getElementById("modelCard").innerHTML = items.map(([title, text]) => `
    <article class="model-card-item"><h3>${title}</h3><p>${text}</p></article>
  `).join("");
}

loadData().catch(error => {
  console.error(error);
  document.body.insertAdjacentHTML("afterbegin", `<div style="padding:1rem;background:#7f1d1d;color:white">Failed to load dashboard data. Use a local server instead of opening index.html directly.</div>`);
});
