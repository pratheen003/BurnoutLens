/**
 * BurnoutLens Single Page Application Logic
 * Pure Vanilla JavaScript - No external dependencies or runtime CDN calls.
 */

// Application In-Memory State (Never stored in localStorage or persistent storage)
const state = {
  currentRoute: "home",
  schema: null,
  manifest: null,
  lastPredictionResult: null,
  analyticsImportance: null,
  analyticsClusters: null,
  modelsComparison: null,
  activeContributionGroup: "all",
};

// Canonical Feature to Snake Case Mapping
const FEATURE_MAP = {
  "Gender": "gender",
  "Age": "age",
  "Occupation": "occupation",
  "Sleep Duration": "sleep_duration",
  "Quality of Sleep": "quality_of_sleep",
  "Physical Activity Level": "physical_activity_level",
  "BMI Category": "bmi_category",
  "Heart Rate": "heart_rate",
  "Daily Steps": "daily_steps",
  "Sleep Disorder": "sleep_disorder",
  "Systolic BP": "systolic_bp",
  "Diastolic BP": "diastolic_bp",
};

const FEATURE_LABELS = {
  "gender": "Gender",
  "age": "Age (years)",
  "occupation": "Occupation",
  "sleep_duration": "Sleep Duration (hours/night)",
  "quality_of_sleep": "Quality of Sleep (scale 1–10)",
  "physical_activity_level": "Physical Activity (minutes/day)",
  "bmi_category": "BMI Category",
  "heart_rate": "Resting Heart Rate (bpm)",
  "daily_steps": "Daily Steps (steps/day)",
  "sleep_disorder": "Sleep Disorder",
  "systolic_bp": "Systolic Blood Pressure (mmHg)",
  "diastolic_bp": "Diastolic Blood Pressure (mmHg)",
};

// Verified Dataset Examples (Real rows from cleaned dataset: Low=Row 32, Medium=Row 0, High=Row 1)
const DATASET_EXAMPLES = {
  low: {
    gender: "Female",
    age: 31,
    occupation: "Nurse",
    sleep_duration: 7.9,
    quality_of_sleep: 8,
    physical_activity_level: 75,
    bmi_category: "Normal",
    heart_rate: 69,
    daily_steps: 6800,
    sleep_disorder: "None",
    systolic_bp: 117,
    diastolic_bp: 76,
  },
  medium: {
    gender: "Male",
    age: 27,
    occupation: "Software Engineer",
    sleep_duration: 6.1,
    quality_of_sleep: 6,
    physical_activity_level: 42,
    bmi_category: "Overweight",
    heart_rate: 77,
    daily_steps: 4200,
    sleep_disorder: "None",
    systolic_bp: 126,
    diastolic_bp: 83,
  },
  high: {
    gender: "Male",
    age: 28,
    occupation: "Doctor",
    sleep_duration: 6.2,
    quality_of_sleep: 6,
    physical_activity_level: 60,
    bmi_category: "Normal",
    heart_rate: 75,
    daily_steps: 10000,
    sleep_disorder: "None",
    systolic_bp: 125,
    diastolic_bp: 80,
  },
};

// SVG Namespace Helper
const SVG_NS = "http://www.w3.org/2000/svg";

function createSvgElement(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    el.setAttribute(k, v);
  }
  return el;
}

/* ==========================================================================
   Routing System
   ========================================================================== */

function initRouter() {
  window.addEventListener("hashchange", handleRouteChange);
  handleRouteChange();
}

function handleRouteChange() {
  const hash = window.location.hash.slice(1) || "/home";
  const route = hash.replace(/^\//, "") || "home";
  navigateTo(route, false);
}

function navigateTo(route, updateHash = true) {
  const validRoutes = ["home", "assess", "result", "analytics", "models", "about"];
  const targetRoute = validRoutes.includes(route) ? route : "home";

  state.currentRoute = targetRoute;

  // Toggle Section Views
  validRoutes.forEach((r) => {
    const sec = document.getElementById(`view-${r}`);
    if (sec) {
      sec.style.display = r === targetRoute ? "block" : "none";
    }
  });

  // Update Nav Link Active States
  document.querySelectorAll(".nav-link").forEach((link) => {
    if (link.getAttribute("data-route") === targetRoute) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });

  if (updateHash) {
    window.location.hash = `#/${targetRoute}`;
  }

  window.scrollTo({ top: 0, behavior: "smooth" });

  // Route-Specific Initializers
  if (targetRoute === "assess") {
    ensureSchemaLoaded();
  } else if (targetRoute === "result") {
    checkResultView();
  } else if (targetRoute === "analytics") {
    loadAnalyticsView();
  } else if (targetRoute === "models") {
    loadModelsView();
  }
}

/* ==========================================================================
   Theme Management
   ========================================================================== */

function initTheme() {
  const toggleBtn = document.getElementById("theme-toggle-btn");
  const darkIcon = document.getElementById("theme-icon-dark");
  const lightIcon = document.getElementById("theme-icon-light");

  const savedTheme = localStorage.getItem("burnoutlens_theme") || "auto";

  function applyTheme(theme) {
    if (theme === "dark" || (theme === "auto" && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
      document.documentElement.setAttribute("data-theme", "dark");
      darkIcon.style.display = "none";
      lightIcon.style.display = "block";
    } else {
      document.documentElement.setAttribute("data-theme", "light");
      darkIcon.style.display = "block";
      lightIcon.style.display = "none";
    }
  }

  applyTheme(savedTheme);

  toggleBtn.addEventListener("click", () => {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const nextTheme = isDark ? "light" : "dark";
    localStorage.setItem("burnoutlens_theme", nextTheme);
    applyTheme(nextTheme);
  });
}

/* ==========================================================================
   API Service Layer
   ========================================================================== */

async function checkApiHealth() {
  const statusIndicator = document.getElementById("system-status");
  const dot = statusIndicator.querySelector(".status-dot");
  const label = statusIndicator.querySelector(".status-label");
  const errorContainer = document.getElementById("global-error-container");

  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.status === "ok") {
      dot.classList.remove("status-error");
      label.textContent = "API Ready";
      errorContainer.style.display = "none";
      return true;
    }
  } catch (err) {
    dot.classList.add("status-error");
    label.textContent = "API Offline";
    errorContainer.style.display = "block";
    return false;
  }
}

async function fetchMetadata() {
  const res = await fetch("/meta");
  if (!res.ok) throw new Error(`Failed to load metadata: ${res.statusText}`);
  return await res.json();
}

async function fetchAnalyticsImportance() {
  const res = await fetch("/analytics/importance");
  if (!res.ok) throw new Error(`Failed to load importance: ${res.statusText}`);
  return await res.json();
}

async function fetchAnalyticsClusters() {
  const res = await fetch("/analytics/clusters");
  if (!res.ok) throw new Error(`Failed to load clusters: ${res.statusText}`);
  return await res.json();
}

async function fetchModelsComparison() {
  const res = await fetch("/models/comparison");
  if (!res.ok) throw new Error(`Failed to load model comparison: ${res.statusText}`);
  return await res.json();
}

async function submitAssessment(payload) {
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : `Server error (HTTP ${res.status})`;
    const error = new Error(detail);
    error.status = res.status;
    error.data = data;
    throw error;
  }
  return data;
}

/* ==========================================================================
   View: Assess (Dynamic Form Construction)
   ========================================================================== */

async function ensureSchemaLoaded() {
  if (state.schema) return;

  const container = document.getElementById("form-fields-container");
  container.replaceChildren();

  const loadingText = document.createElement("div");
  loadingText.className = "loading-placeholder";
  loadingText.textContent = "Calibrating dynamic input schema from /meta...";
  container.appendChild(loadingText);

  try {
    const meta = await fetchMetadata();
    state.schema = meta.feature_schema;
    state.manifest = meta.manifest_summary;
    renderDynamicForm(meta.feature_schema, meta.manifest_summary.feature_order);
  } catch (err) {
    container.replaceChildren();
    const errBox = document.createElement("div");
    errBox.className = "alert-box alert-error";
    errBox.textContent = `Unable to calibrate form from API metadata: ${err.message}. Please check connection.`;
    container.appendChild(errBox);
  }
}

function renderDynamicForm(schema, featureOrder) {
  const container = document.getElementById("form-fields-container");
  container.replaceChildren();

  const order = featureOrder || Object.keys(schema);

  order.forEach((featName) => {
    const spec = schema[featName];
    if (!spec) return;

    const fieldKey = FEATURE_MAP[featName] || featName.toLowerCase().replace(/\s+/g, "_");
    const labelText = FEATURE_LABELS[fieldKey] || featName;

    const group = document.createElement("div");
    group.className = "form-group";

    const label = document.createElement("label");
    label.className = "form-label";
    label.setAttribute("for", `input-${fieldKey}`);
    label.textContent = labelText;
    group.appendChild(label);

    if (spec.type === "categorical") {
      const select = document.createElement("select");
      select.className = "form-select";
      select.id = `input-${fieldKey}`;
      select.name = fieldKey;
      select.required = true;

      const defaultOpt = document.createElement("option");
      defaultOpt.value = "";
      defaultOpt.textContent = `Select ${labelText}...`;
      select.appendChild(defaultOpt);

      (spec.allowed_categories || []).forEach((cat) => {
        const opt = document.createElement("option");
        opt.value = cat;
        opt.textContent = cat;
        select.appendChild(opt);
      });

      group.appendChild(select);
    } else {
      const input = document.createElement("input");
      input.className = "form-input";
      input.type = "number";
      input.id = `input-${fieldKey}`;
      input.name = fieldKey;
      input.required = true;

      if (spec.min !== undefined) input.min = String(spec.min);
      if (spec.max !== undefined) input.max = String(spec.max);

      // Determine appropriate step
      if (spec.dtype === "float64" || spec.min % 1 !== 0 || spec.max % 1 !== 0) {
        input.step = "0.1";
      } else {
        input.step = "1";
      }

      group.appendChild(input);

      const hint = document.createElement("div");
      hint.className = "form-hint";
      const medianText = spec.median !== undefined ? ` • Dataset median: ${spec.median}` : "";
      hint.textContent = `Allowed range: ${spec.min} to ${spec.max}${medianText}`;
      group.appendChild(hint);
    }

    const errorDiv = document.createElement("div");
    errorDiv.className = "form-error-msg";
    errorDiv.id = `error-${fieldKey}`;
    errorDiv.style.display = "none";
    group.appendChild(errorDiv);

    container.appendChild(group);
  });
}

function populateFormValues(data) {
  const form = document.getElementById("assessment-form");
  hideFormError();

  for (const [key, val] of Object.entries(data)) {
    const el = form.elements[key];
    if (el) {
      el.value = String(val);
      el.dispatchEvent(new Event("change"));
    }
  }
}

function showFormError(msg) {
  const box = document.getElementById("form-error-callout");
  const txt = document.getElementById("form-error-text");
  txt.textContent = msg;
  box.style.display = "block";
  box.scrollIntoView({ behavior: "smooth", block: "center" });
}

function hideFormError() {
  const box = document.getElementById("form-error-callout");
  box.style.display = "none";
}

/* ==========================================================================
   Form Handling & Submission
   ========================================================================== */

function initAssessmentForm() {
  const form = document.getElementById("assessment-form");
  const submitBtn = document.getElementById("btn-submit-assessment");
  const spinner = document.getElementById("submit-spinner");

  // Load Example Buttons
  document.getElementById("btn-example-low").addEventListener("click", () => populateFormValues(DATASET_EXAMPLES.low));
  document.getElementById("btn-example-med").addEventListener("click", () => populateFormValues(DATASET_EXAMPLES.medium));
  document.getElementById("btn-example-high").addEventListener("click", () => populateFormValues(DATASET_EXAMPLES.high));

  // Reset Button
  document.getElementById("btn-reset-assessment").addEventListener("click", () => {
    hideFormError();
    document.querySelectorAll(".form-error-msg").forEach((e) => (e.style.display = "none"));
  });

  // Submit Handler
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideFormError();

    if (!state.schema) {
      showFormError("Schema is not yet loaded. Please wait a moment.");
      return;
    }

    const formData = new FormData(form);
    const payload = {};
    let clientValidationError = null;

    // Validate client-side against dynamic schema
    for (const [featName, spec] of Object.entries(state.schema)) {
      const fieldKey = FEATURE_MAP[featName] || featName.toLowerCase().replace(/\s+/g, "_");
      const val = formData.get(fieldKey);

      if (val === null || val === "") {
        clientValidationError = `Field "${FEATURE_LABELS[fieldKey] || featName}" is required.`;
        break;
      }

      if (spec.type === "numeric") {
        const num = parseFloat(val);
        if (isNaN(num)) {
          clientValidationError = `Field "${FEATURE_LABELS[fieldKey] || featName}" must be a valid number.`;
          break;
        }
        if (num < spec.min || num > spec.max) {
          clientValidationError = `Field "${FEATURE_LABELS[fieldKey] || featName}" (${num}) is outside allowed range [${spec.min}, ${spec.max}].`;
          break;
        }
        payload[fieldKey] = num;
      } else {
        const strVal = String(val).trim();
        if (!spec.allowed_categories.includes(strVal)) {
          clientValidationError = `Field "${FEATURE_LABELS[fieldKey] || featName}" has invalid option "${strVal}".`;
          break;
        }
        payload[fieldKey] = strVal;
      }
    }

    if (clientValidationError) {
      showFormError(clientValidationError);
      return;
    }

    // Submit Request
    try {
      submitBtn.disabled = true;
      spinner.style.display = "inline-block";

      const result = await submitAssessment(payload);
      state.lastPredictionResult = result;

      // Render Result & Navigate
      renderResult(result);
      navigateTo("result");
    } catch (err) {
      showFormError(err.message || "Failed to process prediction request.");
    } finally {
      submitBtn.disabled = false;
      spinner.style.display = "none";
    }
  });
}

/* ==========================================================================
   View: Result Rendering
   ========================================================================== */

function checkResultView() {
  const noResultBox = document.getElementById("no-result-warning");
  const content = document.getElementById("result-content");

  if (!state.lastPredictionResult) {
    noResultBox.style.display = "block";
    content.style.display = "none";
  } else {
    noResultBox.style.display = "none";
    content.style.display = "block";
  }
}

function renderResult(data) {
  checkResultView();

  const pred = data.prediction;
  const contribs = data.contributions;
  const cluster = data.cluster;
  const recs = data.recommendations;

  // 1. Risk Level Badge
  const badge = document.getElementById("res-risk-badge");
  badge.textContent = pred.predicted_class;
  badge.className = "risk-badge";
  if (pred.predicted_class === "Low") {
    badge.classList.add("risk-low");
  } else if (pred.predicted_class === "Medium") {
    badge.classList.add("risk-medium");
  } else {
    badge.classList.add("risk-high");
  }

  // 2. Uncalibrated Model Score Bars
  const barsContainer = document.getElementById("score-bars-container");
  barsContainer.replaceChildren();

  const classes = ["Low", "Medium", "High"];
  classes.forEach((c) => {
    const score = pred.model_probability[c] !== undefined ? pred.model_probability[c] : 0;
    const pct = Math.round(score * 100);

    const row = document.createElement("div");
    row.className = "score-bar-row";

    const header = document.createElement("div");
    header.className = "score-bar-header";

    const nameSpan = document.createElement("span");
    nameSpan.textContent = `${c} Risk Score`;
    const valSpan = document.createElement("span");
    valSpan.textContent = `${score.toFixed(4)} (${pct}%)`;

    header.appendChild(nameSpan);
    header.appendChild(valSpan);
    row.appendChild(header);

    const track = document.createElement("div");
    track.className = "score-bar-track";

    const fill = document.createElement("div");
    fill.className = `score-bar-fill risk-${c.toLowerCase()}`;
    fill.style.width = `${pct}%`;
    track.appendChild(fill);

    row.appendChild(track);
    barsContainer.appendChild(row);
  });

  // Probability note tooltip
  const tooltip = document.getElementById("probability-tooltip-text");
  if (pred.probability_note) {
    tooltip.textContent = pred.probability_note;
  }

  // 3. Render SHAP Contributions Chart
  renderContributionsChart(contribs);

  // 4. Render Behavioral Cluster & PCA Plot
  renderClusterSection(cluster);

  // 5. Render Recommendations
  renderRecommendations(recs);

  // 6. Disclaimer
  const disclaimerEl = document.getElementById("res-disclaimer-text");
  disclaimerEl.textContent = data.disclaimer || "Lifestyle-based burnout risk estimate for educational purposes. Not a medical diagnosis.";
}

/* ==========================================================================
   Diverging SHAP Feature Contributions Chart (Inline SVG)
   ========================================================================== */

function renderContributionsChart(contribs) {
  const container = document.getElementById("contributions-chart-container");
  container.replaceChildren();

  const features = contribs.features || [];
  const filtered = state.activeContributionGroup === "all"
    ? features
    : features.filter((f) => f.group === state.activeContributionGroup);

  if (filtered.length === 0) {
    const empty = document.createElement("p");
    empty.className = "card-intro";
    empty.textContent = "No features found for this group.";
    container.appendChild(empty);
    return;
  }

  // Dimensions
  const rowHeight = 36;
  const margin = { top: 20, right: 90, bottom: 40, left: 180 };
  const width = Math.max(container.clientWidth || 600, 580);
  const height = margin.top + margin.bottom + filtered.length * rowHeight;
  const chartWidth = width - margin.left - margin.right;

  // Max absolute contribution for symmetric scaling
  const maxAbs = Math.max(0.8, ...filtered.map((f) => Math.abs(f.signed_contribution)));
  const zeroX = margin.left + chartWidth / 2;

  const svg = createSvgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width: "100%",
    height: String(height),
    class: "contributions-svg",
  });

  // Zero reference axis
  const zeroLine = createSvgElement("line", {
    x1: String(zeroX),
    y1: String(margin.top),
    x2: String(zeroX),
    y2: String(height - margin.bottom),
    stroke: "var(--border-strong)",
    "stroke-width": "1.5",
    "stroke-dasharray": "3,3",
  });
  svg.appendChild(zeroLine);

  // Axis Labels
  const leftLabel = createSvgElement("text", {
    x: String(margin.left + 10),
    y: String(height - 12),
    "font-size": "11",
    fill: "var(--text-muted)",
    "text-anchor": "start",
    "font-weight": "600",
  });
  leftLabel.textContent = `← Pushes Away from ${contribs.predicted_class}`;
  svg.appendChild(leftLabel);

  const rightLabel = createSvgElement("text", {
    x: String(width - margin.right - 10),
    y: String(height - 12),
    "font-size": "11",
    fill: "var(--text-muted)",
    "text-anchor": "end",
    "font-weight": "600",
  });
  rightLabel.textContent = `Pushes Toward ${contribs.predicted_class} →`;
  svg.appendChild(rightLabel);

  filtered.forEach((f, idx) => {
    const y = margin.top + idx * rowHeight;
    const barY = y + 7;
    const barHeight = 20;

    // Feature Name + User Value Label
    const nameText = createSvgElement("text", {
      x: String(margin.left - 12),
      y: String(barY + 14),
      "text-anchor": "end",
      "font-size": "12",
      fill: "var(--text-primary)",
      "font-weight": "500",
    });
    nameText.textContent = `${f.feature} (${f.user_value})`;
    svg.appendChild(nameText);

    // Bar Calculation
    const barWidth = (Math.abs(f.signed_contribution) / maxAbs) * (chartWidth / 2);
    const barX = f.signed_contribution >= 0 ? zeroX : zeroX - barWidth;

    // Colors: Positive pushes toward class; negative pushes away
    const barColor = f.signed_contribution >= 0 ? "var(--primary)" : "var(--border-strong)";

    const rect = createSvgElement("rect", {
      x: String(barX),
      y: String(barY),
      width: String(barWidth),
      height: String(barHeight),
      fill: barColor,
      rx: "3",
      opacity: "0.88",
    });
    svg.appendChild(rect);

    // Value Text
    const valTextX = f.signed_contribution >= 0 ? zeroX + barWidth + 8 : zeroX - barWidth - 8;
    const valAnchor = f.signed_contribution >= 0 ? "start" : "end";

    const valText = createSvgElement("text", {
      x: String(valTextX),
      y: String(barY + 14),
      "text-anchor": valAnchor,
      "font-size": "11",
      "font-weight": "700",
      fill: f.signed_contribution >= 0 ? "var(--primary)" : "var(--text-secondary)",
    });
    valText.textContent = `${f.signed_contribution > 0 ? "+" : ""}${f.signed_contribution.toFixed(4)}`;
    svg.appendChild(valText);
  });

  container.appendChild(svg);
}

// Group Filter Tab Listeners
function initContributionTabs() {
  document.querySelectorAll(".group-tabs .tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".group-tabs .tab-btn").forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      state.activeContributionGroup = btn.getAttribute("data-group");
      if (state.lastPredictionResult) {
        renderContributionsChart(state.lastPredictionResult.contributions);
      }
    });
  });
}

/* ==========================================================================
   Behavioral Group & 2D PCA Scatter Plot (Inline SVG)
   ========================================================================== */

async function renderClusterSection(userCluster) {
  // Cluster info cards
  document.getElementById("res-cluster-id").textContent = `Cluster ${userCluster.cluster_id}`;
  document.getElementById("res-cluster-label").textContent = userCluster.profile_label || `Cluster ${userCluster.cluster_id}`;

  const statsContainer = document.getElementById("res-cluster-stats");
  statsContainer.replaceChildren();

  const summary = userCluster.profile_summary || {};
  const statMetrics = [
    { label: "Mean Sleep", val: summary.mean_sleep_duration ? `${summary.mean_sleep_duration} hrs` : "N/A" },
    { label: "Mean Quality", val: summary.mean_quality_of_sleep ? `${summary.mean_quality_of_sleep}/10` : "N/A" },
    { label: "Daily Steps", val: summary.mean_daily_steps ? `${Math.round(summary.mean_daily_steps)}` : "N/A" },
    { label: "Resting HR", val: summary.mean_heart_rate ? `${Math.round(summary.mean_heart_rate)} bpm` : "N/A" },
  ];

  statMetrics.forEach((m) => {
    const item = document.createElement("div");
    item.className = "stat-item";

    const lbl = document.createElement("div");
    lbl.className = "stat-label";
    lbl.textContent = m.label;

    const v = document.createElement("div");
    v.className = "stat-value";
    v.textContent = m.val;

    item.appendChild(lbl);
    item.appendChild(v);
    statsContainer.appendChild(item);
  });

  // Load Background PCA Points if needed
  if (!state.analyticsClusters) {
    try {
      state.analyticsClusters = await fetchAnalyticsClusters();
    } catch (err) {
      console.error("Unable to load background PCA points:", err);
    }
  }

  const bgPoints = state.analyticsClusters ? state.analyticsClusters.pca_points_variant_b : [];
  renderPcaScatterPlot(bgPoints, userCluster.user_pca_coordinates);
}

function renderPcaScatterPlot(points, userCoord) {
  const container = document.getElementById("pca-scatter-container");
  container.replaceChildren();

  const width = Math.max(container.clientWidth || 360, 320);
  const height = 260;
  const padding = 35;

  // Compute Domain Bounds
  const allX = points.map((p) => p.pc1);
  const allY = points.map((p) => p.pc2);

  if (userCoord) {
    allX.push(userCoord.pc1);
    allY.push(userCoord.pc2);
  }

  const minX = Math.min(-3.0, ...allX);
  const maxX = Math.max(3.0, ...allX);
  const minY = Math.min(-2.5, ...allY);
  const maxY = Math.max(2.5, ...allY);

  const scaleX = (val) => padding + ((val - minX) / (maxX - minX)) * (width - 2 * padding);
  const scaleY = (val) => height - padding - ((val - minY) / (maxY - minY)) * (height - 2 * padding);

  const svg = createSvgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width: "100%",
    height: String(height),
  });

  // Cluster Color Palette (5 clusters)
  const clusterColors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"];

  // Axes
  const axisX = createSvgElement("line", {
    x1: String(padding),
    y1: String(scaleY(0)),
    x2: String(width - padding),
    y2: String(scaleY(0)),
    stroke: "var(--border-color)",
    "stroke-width": "1",
  });
  const axisY = createSvgElement("line", {
    x1: String(scaleX(0)),
    y1: String(padding),
    x2: String(scaleX(0)),
    y2: String(height - padding),
    stroke: "var(--border-color)",
    "stroke-width": "1",
  });
  svg.appendChild(axisX);
  svg.appendChild(axisY);

  // Background 132 Unique Profile Points
  points.forEach((p) => {
    const cx = scaleX(p.pc1);
    const cy = scaleY(p.pc2);
    const color = clusterColors[p.cluster_id % clusterColors.length];

    const dot = createSvgElement("circle", {
      cx: String(cx),
      cy: String(cy),
      r: "4.5",
      fill: color,
      opacity: "0.65",
      stroke: "var(--bg-surface)",
      "stroke-width": "1",
    });
    svg.appendChild(dot);
  });

  // User Highlight Point
  if (userCoord) {
    const ux = scaleX(userCoord.pc1);
    const uy = scaleY(userCoord.pc2);

    // Pulsing outer halo
    const halo = createSvgElement("circle", {
      cx: String(ux),
      cy: String(uy),
      r: "12",
      fill: "none",
      stroke: "var(--text-primary)",
      "stroke-width": "2",
      "stroke-dasharray": "2,2",
    });
    svg.appendChild(halo);

    const userPoint = createSvgElement("circle", {
      cx: String(ux),
      cy: String(uy),
      r: "6",
      fill: "var(--primary)",
      stroke: "var(--text-inverse)",
      "stroke-width": "2",
    });
    svg.appendChild(userPoint);

    const userLabel = createSvgElement("text", {
      x: String(ux + 9),
      y: String(uy - 9),
      "font-size": "11",
      "font-weight": "800",
      fill: "var(--text-primary)",
    });
    userLabel.textContent = "YOU";
    svg.appendChild(userLabel);
  }

  container.appendChild(svg);
}

/* ==========================================================================
   Wellness Information Rendering
   ========================================================================== */

function renderRecommendations(recs) {
  const container = document.getElementById("wellness-items-container");
  container.replaceChildren();

  const badge = document.getElementById("wellness-category-badge");
  badge.textContent = recs.field_category || "General wellness information, not medical advice";

  (recs.items || []).forEach((text) => {
    const item = document.createElement("div");
    item.className = "wellness-item";

    const icon = createSvgElement("svg", {
      viewBox: "0 0 24 24",
      width: "20",
      height: "20",
      fill: "none",
      stroke: "currentColor",
      "stroke-width": "2",
      class: "wellness-icon",
    });
    const check = createSvgElement("polyline", { points: "20 6 9 17 4 12" });
    icon.appendChild(check);
    item.appendChild(icon);

    const p = document.createElement("p");
    p.textContent = text;
    item.appendChild(p);

    container.appendChild(item);
  });
}

/* ==========================================================================
   View 4: Analytics
   ========================================================================== */

async function loadAnalyticsView() {
  const chartContainer = document.getElementById("analytics-permutation-chart");
  const clustersGrid = document.getElementById("analytics-clusters-grid");

  try {
    if (!state.analyticsImportance) {
      state.analyticsImportance = await fetchAnalyticsImportance();
    }
    if (!state.analyticsClusters) {
      state.analyticsClusters = await fetchAnalyticsClusters();
    }

    renderPermutationImportanceChart(state.analyticsImportance.permutation_importance);
    renderClusterProfilesCards(state.analyticsClusters.variant_b_profiles);
  } catch (err) {
    chartContainer.replaceChildren();
    const errEl = document.createElement("div");
    errEl.className = "alert-box alert-error";
    errEl.textContent = `Failed to load analytics: ${err.message}`;
    chartContainer.appendChild(errEl);
  }
}

function renderPermutationImportanceChart(permData) {
  const container = document.getElementById("analytics-permutation-chart");
  container.replaceChildren();

  const ranking = permData.ranking || [];
  const rowHeight = 34;
  const margin = { top: 20, right: 140, bottom: 40, left: 170 };
  const width = Math.max(container.clientWidth || 600, 560);
  const height = margin.top + margin.bottom + ranking.length * rowHeight;
  const chartWidth = width - margin.left - margin.right;

  const maxVal = Math.max(0.2, ...ranking.map((r) => r.mean_f1_drop + r.std_f1_drop));
  const scale = (val) => margin.left + Math.max(0, (val / maxVal) * chartWidth);

  const svg = createSvgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width: "100%",
    height: String(height),
  });

  ranking.forEach((item, idx) => {
    const y = margin.top + idx * rowHeight;
    const barY = y + 6;
    const barHeight = 20;

    // Feature Name Label
    const textName = createSvgElement("text", {
      x: String(margin.left - 10),
      y: String(barY + 14),
      "text-anchor": "end",
      "font-size": "12",
      "font-weight": "500",
      fill: "var(--text-primary)",
    });
    textName.textContent = item.feature;
    svg.appendChild(textName);

    // Bar
    const barW = Math.max(0, (item.mean_f1_drop / maxVal) * chartWidth);
    const rect = createSvgElement("rect", {
      x: String(margin.left),
      y: String(barY),
      width: String(barW),
      height: String(barHeight),
      fill: "var(--primary)",
      rx: "3",
      opacity: "0.85",
    });
    svg.appendChild(rect);

    // Error Bar (Mean +/- Std)
    const errX1 = scale(Math.max(0, item.mean_f1_drop - item.std_f1_drop));
    const errX2 = scale(item.mean_f1_drop + item.std_f1_drop);
    const errY = barY + barHeight / 2;

    const errLine = createSvgElement("line", {
      x1: String(errX1),
      y1: String(errY),
      x2: String(errX2),
      y2: String(errY),
      stroke: "var(--text-primary)",
      "stroke-width": "1.5",
    });
    svg.appendChild(errLine);

    // Data Rule Badge: clearly above zero if mean > 2*std
    const isClear = item.mean_f1_drop > 2 * item.std_f1_drop;
    const badgeText = createSvgElement("text", {
      x: String(errX2 + 10),
      y: String(barY + 14),
      "font-size": "10",
      "font-weight": isClear ? "700" : "500",
      fill: isClear ? "var(--risk-low)" : "var(--text-muted)",
    });
    badgeText.textContent = isClear ? "CLEAR (mean > 2σ)" : "not distinguishable";
    svg.appendChild(badgeText);
  });

  container.appendChild(svg);
}

function renderClusterProfilesCards(profiles) {
  const container = document.getElementById("analytics-clusters-grid");
  container.replaceChildren();

  const clusterColors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"];

  for (const [cid, prof] of Object.entries(profiles)) {
    const card = document.createElement("article");
    card.className = "cluster-profile-card";

    const header = document.createElement("div");
    header.className = "cluster-card-header";

    const badge = document.createElement("span");
    badge.className = "cluster-card-badge";
    badge.style.backgroundColor = `${clusterColors[parseInt(cid) % clusterColors.length]}22`;
    badge.style.color = clusterColors[parseInt(cid) % clusterColors.length];
    badge.textContent = `Cluster ${cid}`;

    const size = document.createElement("span");
    size.className = "cluster-card-size";
    size.textContent = `N=${prof.size} (${prof.share_pct.toFixed(1)}%)`;

    header.appendChild(badge);
    header.appendChild(size);
    card.appendChild(header);

    const title = document.createElement("h3");
    title.className = "cluster-card-title";
    title.textContent = prof.human_descriptive_label;
    card.appendChild(title);

    const stats = prof.key_profile_values || {};
    const list = document.createElement("ul");
    list.style.fontSize = "0.8125rem";
    list.style.color = "var(--text-secondary)";
    list.style.paddingLeft = "1.2rem";
    list.style.lineHeight = "1.5";

    const addBullet = (txt) => {
      const li = document.createElement("li");
      li.textContent = txt;
      list.appendChild(li);
    };

    if (stats.mean_sleep_duration) addBullet(`Sleep: ${stats.mean_sleep_duration}h (Quality: ${stats.mean_quality_of_sleep}/10)`);
    if (stats.mean_daily_steps) addBullet(`Activity: ${stats.mean_daily_steps} steps, ${stats.mean_physical_activity}m`);
    if (stats.mean_heart_rate) addBullet(`Resting HR: ${stats.mean_heart_rate} bpm`);
    if (stats.dominant_gender) addBullet(`Dominant sex: ${stats.dominant_gender} (${stats.gender_share_pct}%)`);

    card.appendChild(list);
    container.appendChild(card);
  }
}

/* ==========================================================================
   View 5: Models Comparison
   ========================================================================== */

async function loadModelsView() {
  const chartContainer = document.getElementById("models-comparison-chart");
  const tableBody = document.getElementById("models-table-body");

  try {
    if (!state.modelsComparison) {
      state.modelsComparison = await fetchModelsComparison();
    }

    renderModelsComparison(state.modelsComparison.models);
  } catch (err) {
    chartContainer.replaceChildren();
    const errEl = document.createElement("div");
    errEl.className = "alert-box alert-error";
    errEl.textContent = `Failed to load model comparison: ${err.message}`;
    chartContainer.appendChild(errEl);
  }
}

function renderModelsComparison(modelsData) {
  const chartContainer = document.getElementById("models-comparison-chart");
  const tableBody = document.getElementById("models-table-body");
  chartContainer.replaceChildren();
  tableBody.replaceChildren();

  // Model comparison metrics
  const rows = [];
  for (const [mName, records] of Object.entries(modelsData)) {
    if (mName.includes("Majority-Class")) continue;

    const protoC = records.find((r) => r.protocol.includes("Protocol C (Grouped 5-Fold CV 5-Seeds Repeated)"));
    const protoD = records.find((r) => r.protocol.includes("Protocol D (Random Stratified 5-Fold CV 5-Seeds Repeated)"));

    if (protoC && protoD) {
      const diffF1 = (protoD.macro_f1 - protoC.macro_f1) * 100;
      rows.push({
        name: mName,
        accC: protoC.accuracy,
        stdAccC: protoC.std_accuracy,
        f1C: protoC.macro_f1,
        stdF1C: protoC.std_macro_f1,
        accD: protoD.accuracy,
        stdAccD: protoD.std_accuracy,
        f1D: protoD.macro_f1,
        stdF1D: protoD.std_macro_f1,
        diffF1: diffF1,
      });

      // Table row
      const tr = document.createElement("tr");

      const tdName = document.createElement("td");
      tdName.innerHTML = `<strong>${mName}</strong>${mName === "Logistic Regression" ? " <span class='badge-subtle'>Selected</span>" : ""}`;
      tr.appendChild(tdName);

      const tdAccC = document.createElement("td");
      tdAccC.textContent = `${(protoC.accuracy * 100).toFixed(2)}% ± ${(protoC.std_accuracy * 100).toFixed(2)}`;
      tr.appendChild(tdAccC);

      const tdF1C = document.createElement("td");
      tdF1C.textContent = `${(protoC.macro_f1 * 100).toFixed(2)}% ± ${(protoC.std_macro_f1 * 100).toFixed(2)}`;
      tr.appendChild(tdF1C);

      const tdAccD = document.createElement("td");
      tdAccD.textContent = `${(protoD.accuracy * 100).toFixed(2)}% ± ${(protoD.std_accuracy * 100).toFixed(2)}`;
      tr.appendChild(tdAccD);

      const tdF1D = document.createElement("td");
      tdF1D.textContent = `${(protoD.macro_f1 * 100).toFixed(2)}% ± ${(protoD.std_macro_f1 * 100).toFixed(2)}`;
      tr.appendChild(tdF1D);

      const tdDiff = document.createElement("td");
      tdDiff.textContent = `+${diffF1.toFixed(2)}%`;
      tdDiff.style.fontWeight = "700";
      tdDiff.style.color = diffF1 > 2.0 ? "var(--risk-med)" : "var(--primary)";
      tr.appendChild(tdDiff);

      tableBody.appendChild(tr);
    }
  }

  // Render Grouped Bar Chart (Protocol C vs Protocol D Macro-F1)
  const margin = { top: 25, right: 30, bottom: 50, left: 160 };
  const width = Math.max(chartContainer.clientWidth || 600, 560);
  const height = 280;
  const chartHeight = height - margin.top - margin.bottom;
  const chartWidth = width - margin.left - margin.right;

  const svg = createSvgElement("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width: "100%",
    height: String(height),
  });

  const groupHeight = chartHeight / rows.length;
  const barHeight = 14;

  rows.forEach((r, idx) => {
    const groupY = margin.top + idx * groupHeight;

    // Label
    const text = createSvgElement("text", {
      x: String(margin.left - 10),
      y: String(groupY + groupHeight / 2 + 4),
      "text-anchor": "end",
      "font-size": "12",
      "font-weight": r.name === "Logistic Regression" ? "700" : "500",
      fill: "var(--text-primary)",
    });
    text.textContent = r.name;
    svg.appendChild(text);

    // Protocol C Bar (Blue)
    const lenC = ((r.f1C - 0.88) / (1.0 - 0.88)) * chartWidth;
    const barC = createSvgElement("rect", {
      x: String(margin.left),
      y: String(groupY + 4),
      width: String(Math.max(0, lenC)),
      height: String(barHeight),
      fill: "var(--primary)",
      rx: "2",
      opacity: "0.9",
    });
    svg.appendChild(barC);

    // Protocol D Bar (Orange/Amber)
    const lenD = ((r.f1D - 0.88) / (1.0 - 0.88)) * chartWidth;
    const barD = createSvgElement("rect", {
      x: String(margin.left),
      y: String(groupY + 20),
      width: String(Math.max(0, lenD)),
      height: String(barHeight),
      fill: "var(--risk-med)",
      rx: "2",
      opacity: "0.9",
    });
    svg.appendChild(barD);
  });

  // Legend at bottom
  const legC = createSvgElement("rect", {
    x: String(margin.left),
    y: String(height - 18),
    width: "14",
    height: "14",
    fill: "var(--primary)",
    rx: "2",
  });
  const legCLbl = createSvgElement("text", {
    x: String(margin.left + 20),
    y: String(height - 7),
    "font-size": "11",
    fill: "var(--text-primary)",
  });
  legCLbl.textContent = "Protocol C (Grouped CV - Honest)";

  const legD = createSvgElement("rect", {
    x: String(margin.left + 230),
    y: String(height - 18),
    width: "14",
    height: "14",
    fill: "var(--risk-med)",
    rx: "2",
  });
  const legDLbl = createSvgElement("text", {
    x: String(margin.left + 250),
    y: String(height - 7),
    "font-size": "11",
    fill: "var(--text-primary)",
  });
  legDLbl.textContent = "Protocol D (Random CV - Duplicate Inflated)";

  svg.appendChild(legC);
  svg.appendChild(legCLbl);
  svg.appendChild(legD);
  svg.appendChild(legDLbl);

  chartContainer.appendChild(svg);
}

/* ==========================================================================
   Application Initialization
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initRouter();
  initAssessmentForm();
  initContributionTabs();

  // Print Button Handler
  const printBtn = document.getElementById("btn-print-result");
  if (printBtn) {
    printBtn.addEventListener("click", () => window.print());
  }

  // Retry Connection Button
  const retryBtn = document.getElementById("btn-retry-connection");
  if (retryBtn) {
    retryBtn.addEventListener("click", () => checkApiHealth());
  }

  // Initial Health Check
  checkApiHealth();
});
