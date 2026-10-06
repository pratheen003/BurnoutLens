# BurnoutLens: Lifestyle-Based Burnout Risk Analytics

> [!WARNING]
> **MANDATORY EDUCATIONAL DISCLAIMER**: Lifestyle-based burnout risk estimate for educational purposes. Not a medical diagnosis. BurnoutLens evaluates statistical correlations with self-reported stress levels and does not provide clinical diagnosis, psychiatric assessment, or medical treatment plans.

---

## 1. What Is BurnoutLens?

**BurnoutLens** is an open, transparent machine learning application that estimates relative burnout risk based on 12 measurable lifestyle, sleep, and cardiovascular metrics. 

Unlike opaque black-box classifiers, BurnoutLens provides:
- **Exact Linear Additivity**: Mathematical SHAP feature contributions computed on the model's log-odds decision scale relative to a clean background mean of 132 unique survey respondents.
- **Leakage-Free Validation**: Strict Grouped 5-Fold Cross-Validation that isolates identical participant profiles across folds, demonstrating that random CV overstated historical accuracy by 1–3 percentage points.
- **Behavioral Clustering**: Variant B unsupervised K-Means ($k=5$) and 2D PCA projections focused purely on behavioral lifestyle factors (sleep duration, sleep quality, physical activity, daily steps, heart rate), decoupled from demographic modifiers.
- **Zero-Dependency Frontend**: Responsive, offline-capable Single Page Application built with pure Vanilla HTML, CSS, and JavaScript with custom SVG data visualizations and zero external runtime network requests.

---

## 2. Project Architecture & Directory Structure

```
BurnoutLens/
├── docs/                     # Technical specifications and API documentation
│   ├── API.md               # REST API endpoints, request/response contracts
│   └── screenshots/         # UI verification and interface captures
├── frontend/                # Offline Single Page Application (mounted at "/")
│   ├── css/styles.css       # Vanilla CSS design system (light/dark responsive)
│   ├── js/app.js            # Router, dynamic form generation, SVG visualizers
│   └── index.html           # Accessible semantic markup
├── models/                  # Serialized production model artifacts
│   ├── behavior_clusters.joblib   # StandardScaler, KMeans(k=5), PCA(2D)
│   ├── burnout_pipeline.joblib    # Preprocessor + LogisticRegression pipeline
│   ├── explain_background.json   # 27-feature background mean across 132 unique rows
│   ├── model_manifest.json        # Dependencies, CSV sha256, CV benchmarks
│   └── pca_points_variant_b.json  # 2D PCA coordinates for 132 unique profiles
├── reports/                 # Methodological audit reports and reproduction benchmarks
│   ├── cluster_profiles.json      # Complete profiling of Variant A & B clusters
│   ├── feature_importance.json    # SHAP rankings, permutation drops, worked examples
│   ├── feature_schema.json        # Calibrated ranges and categories for 12 features
│   └── supervised_results.csv     # Benchmarks across Protocols A, B, C, D
├── src/burnoutlens/         # Core application package
│   ├── analytics.py         # K-Means clustering, PCA, demographic purity metrics
│   ├── api.py               # FastAPI backend with CORS, endpoints, static SPA mount
│   ├── config.py            # Feature definitions and forbidden leakage lists
│   ├── data.py              # Data loader with immutability guarantees
│   ├── evaluation.py        # Grouped and repeated cross-validation protocols
│   ├── explain.py           # Linear SHAP explainer and additivity calculations
│   ├── features.py          # Cleaning, target mapping, duplicate group extraction
│   ├── leakage.py           # Runtime assertions preventing target and score leakage
│   ├── preprocessing.py     # ColumnTransformer specification
│   ├── schema.py            # Feature typing and validation logic
│   └── service.py           # Unified prediction, attribution, and clustering service
├── tests/                   # Automated test suite (53 passing tests)
│   ├── test_analytics.py    # Clustering and PCA determinism
│   ├── test_api.py          # FastAPI endpoint integration and 422 validations
│   ├── test_explain.py      # SHAP additivity and feature mapping
│   ├── test_frontend.py     # Static mount, asset integrity, offline URL checks
│   ├── test_preprocessing.py# Clean data transforms and leakage assertions
│   ├── test_service.py      # End-to-end inference and mean-vector formula equivalence
│   └── test_supervised.py   # Protocol C/D isolation and pipeline hygiene
├── BUILD_LOG.md             # Chronological audit log of all engineering phases
└── requirements.txt         # Pinned runtime dependencies
```

---

## 3. Getting Started & Running Locally

### Prerequisites
- Python 3.10+ (tested on Python 3.14.6)
- PowerShell, Bash, or standard command shell

### Dataset Acquisition (Optional for Inference, Required for Full Training/Data Tests)
1. Download the **"Sleep Health and Lifestyle Dataset"** from Kaggle:
   - [Kaggle: Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
   - *Note*: License terms have not been independently verified; the raw CSV is excluded from git tracking (`.gitignore`) and is never committed to the repository.
2. Place the downloaded CSV file in the repository at:
   ```text
   data/raw/Sleep_health_and_lifestyle_dataset.csv
   ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pratheen003/BurnoutLens.git
   cd BurnoutLens
   ```

2. Create and activate a Python virtual environment:
   ```powershell
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install pinned dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```powershell
   uvicorn burnoutlens.api:app --app-dir src --port 8000
   ```

5. Open your browser:
   - **Frontend Application**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
   - **Interactive OpenAPI Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 4. Running the Test Suite

Tests can be run either **with** or **without** the raw Kaggle dataset CSV:

### Running Without the CSV (Inference, Service & API Tests)
If the raw dataset CSV is absent, tests that require raw data will automatically skip with a clear message, while all service, model artifact, API endpoint, and frontend tests will execute and pass:

```powershell
$env:PYTHONPATH = "src"
pytest -v
```
*Expected Output*: **21 passed, 33 skipped**.

### Running With the CSV (Full Suite)
When `data/raw/Sleep_health_and_lifestyle_dataset.csv` is present, the complete test suite executes across all preprocessing, model validation, and analysis modules:

```powershell
$env:PYTHONPATH = "src"
pytest -v
```
*Expected Output*: **54 passed**.

### Test Coverage Breakdown:
- **Phase 2 Preprocessing & Leakage Tests** (`tests/test_preprocessing.py`): 10 tests (skipped without CSV)
- **Phase 3 Supervised Evaluation Tests** (`tests/test_supervised.py`): 5 tests (skipped without CSV)
- **Phase 4 Behavioral Clustering Tests** (`tests/test_analytics.py`): 7 tests (skipped without CSV)
- **Phase 5 Explainable AI Tests** (`tests/test_explain.py`): 5 tests (skipped without CSV)
- **Phase 6 Prediction Service Tests** (`tests/test_service.py`): 9 tests (4 pass without CSV, 5 skip without CSV)
- **Phase 7 FastAPI Backend Tests** (`tests/test_api.py`): 12 tests (all pass without CSV)
- **Phase 8 Frontend & Dataset Integrity Tests** (`tests/test_frontend.py`): 6 tests (5 pass without CSV, 1 skips without CSV)

---

## 5. Development Phases Summary

| Phase | Milestone | Primary Deliverable |
|:---:|:---|:---|
| **Phase 1** | Baseline Recovery | Reproduced original notebook baseline, identified 242 duplicate rows and leakage flaws. |
| **Phase 2** | Leakage-Free Preprocessing | Built reusable `burnoutlens` package, `ColumnTransformer`, `dup_group` hash identifier. |
| **Phase 3** | Honest Supervised Evaluation | Benchmarked 5 model architectures across 4 CV protocols. Selected Logistic Regression. |
| **Phase 4** | Behavioral Analytics & PCA | Evaluated K-Means across $k=2..8$. Developed Variant B (5 behavioral features, $k=5$). |
| **Phase 5** | Explainable AI & SHAP | Linear SHAP attributions with 132-profile background. Verified with permutation test ($\rho=0.9371$). |
| **Phase 6** | Artifact Serialization | Serialized production pipeline, background mean vector, clustering models, and manifest. |
| **Phase 7** | REST API Service | FastAPI backend providing prediction, local SHAP attributions, cluster assignments, and metadata. |
| **Phase 8** | Custom Frontend & Integration | Single Page Application with hash routing, dynamic form calibration, and SVG visualizers. |

---

## 6. Analytical Limitations & Known Artifacts

1. **Observational Nurse Artifact**: In this survey dataset, participants logging $\ge 10,000$ steps and $\ge 90$ minutes of physical activity are 94% nurses reporting High burnout risk. Consequently, the model associates higher steps with higher predicted risk. This is a sampling artifact of occupational representation, not health advice.
2. **Deterministic Stress Target**: Ground truth `Burnout Risk` is deterministically derived from self-reported `Stress Level` (Low: 1–4, Medium: 5–6, High: 7–10). The model predicts self-reported psychological stress.
3. **Uncalibrated Model Scores**: Probabilities generated by the multiclass logistic regression model represent relative log-odds in a small sample ($N=374$) and are not calibrated epidemiological risk estimates.
4. **Context vs. Actionable Indicators**: Features are classified into *Lifestyle* (actionable), *Health Indicators*, and *Context* (non-actionable demographic parameters: Gender, Age, Occupation).
5. **Data License Notice**: The raw Kaggle Sleep Health and Lifestyle dataset is subject to its original licensing terms. The raw CSV is not committed or redistributed in this repository.
