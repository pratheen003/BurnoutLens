# BurnoutLens REST API Specification

This document details the FastAPI HTTP service for the BurnoutLens project, providing real-time burnout risk inference, local SHAP feature attributions, behavioral cluster mapping, and model analytics.

---

## 1. Running the Server Locally

To start the API development server with auto-reload:

```powershell
uvicorn burnoutlens.api:app --app-dir src --reload
```

The service runs by default at `http://127.0.0.1:8000`. Interactive OpenAPI documentation is accessible at `http://127.0.0.1:8000/docs`.

---

## 2. API Endpoints Overview

| Method | Path | Summary | Description |
|:---|:---|:---|:---|
| `GET` | `/health` | Health Check | Verifies service availability and operational readiness. |
| `GET` | `/meta` | Model Metadata & Limitations | Returns schema bounds, manifest summary, disclaimer, and analytical limitations. |
| `POST` | `/predict` | Predict & Explain | Evaluates risk class, uncalibrated probabilities, SHAP contributions, cluster assignment, and wellness guidance. |
| `GET` | `/analytics/clusters` | Behavioral Clustering | Returns Variant B cluster profiles and 2D PCA projection coordinates for 132 unique profiles. |
| `GET` | `/analytics/importance` | Global Importance | Returns global SHAP values, held-out permutation drops, and tree cross-check results. |
| `GET` | `/models/comparison` | Model Comparison | Returns Protocol C and Protocol D cross-validation benchmark results across model families. |

---

## 3. Detailed Endpoint Specifications

### 3.1 GET `/health`
- **Request**: None
- **Response**: `200 OK`
```json
{
  "status": "ok",
  "service": "BurnoutLens API"
}
```

---

### 3.2 GET `/meta`
- **Request**: None
- **Response**: `200 OK`
- **Payload**:
  - `feature_schema`: Permissible ranges, categories, and medians derived from Phase 2 preprocessing.
  - `manifest_summary`: Sanitized model manifest without local directory paths.
  - `disclaimer`: Explicit educational disclaimer.
  - `limitations`: List of known analytical limitations (dataset size, proxy target, observational nurse artifact).

---

### 3.3 POST `/predict`
- **Request Content-Type**: `application/json`
- **Request Keys (snake_case)**:
  - `gender` (*string*): `"Female"` or `"Male"`
  - `age` (*number*): 27.0 to 59.0
  - `occupation` (*string*): e.g., `"Accountant"`, `"Doctor"`, `"Engineer"`, `"Lawyer"`, `"Nurse"`, `"Software Engineer"`, etc.
  - `sleep_duration` (*number*): 5.8 to 8.5 (hours)
  - `quality_of_sleep` (*number*): 4.0 to 9.0 (scale 1–10)
  - `physical_activity_level` (*number*): 30.0 to 90.0 (minutes/day)
  - `bmi_category` (*string*): `"Normal"`, `"Overweight"`, `"Obese"`
  - `heart_rate` (*number*): 65.0 to 86.0 (bpm)
  - `daily_steps` (*number*): 3000.0 to 10000.0 (steps/day)
  - `sleep_disorder` (*string*): `"None"`, `"Insomnia"`, `"Sleep Apnea"`
  - `systolic_bp` (*number*): 115.0 to 142.0 (mmHg)
  - `diastolic_bp` (*number*): 75.0 to 95.0 (mmHg)

#### Real Live Example (Low-Risk Sample from Live Smoke Test)

**Request**:
```json
{
  "gender": "Female",
  "age": 44,
  "occupation": "Accountant",
  "sleep_duration": 7.9,
  "quality_of_sleep": 8,
  "physical_activity_level": 75,
  "bmi_category": "Normal",
  "heart_rate": 69,
  "daily_steps": 7500,
  "sleep_disorder": "None",
  "systolic_bp": 117,
  "diastolic_bp": 76
}
```

**Response (`200 OK`)**:
```json
{
  "prediction": {
    "predicted_class": "Low",
    "model_probability": {
      "Low": 0.9234,
      "Medium": 0.0765,
      "High": 0.0001
    },
    "probability_note": "Model probabilities are uncalibrated estimates derived from a logistic regression model trained on a small dataset (N=374). They represent relative model scores rather than true clinical risk probabilities."
  },
  "contributions": {
    "predicted_class": "Low",
    "base_value": -0.3873,
    "features": [
      {
        "feature": "Quality of Sleep",
        "user_value": 8.0,
        "signed_contribution": 1.7284,
        "direction": "pushes toward Low",
        "group": "lifestyle"
      },
      {
        "feature": "Systolic BP",
        "user_value": 117.0,
        "signed_contribution": 1.0745,
        "direction": "pushes toward Low",
        "group": "health_indicator"
      },
      {
        "feature": "Gender",
        "user_value": "Female",
        "signed_contribution": 0.9633,
        "direction": "pushes toward Low",
        "group": "context"
      },
      {
        "feature": "Daily Steps",
        "user_value": 7500.0,
        "signed_contribution": -0.7574,
        "direction": "pushes away from Low",
        "group": "lifestyle"
      },
      {
        "feature": "Sleep Duration",
        "user_value": 7.9,
        "signed_contribution": 0.7226,
        "direction": "pushes toward Low",
        "group": "lifestyle"
      },
      {
        "feature": "Heart Rate",
        "user_value": 69.0,
        "signed_contribution": 0.6482,
        "direction": "pushes toward Low",
        "group": "lifestyle"
      },
      {
        "feature": "Occupation",
        "user_value": "Accountant",
        "signed_contribution": -0.4906,
        "direction": "pushes away from Low",
        "group": "context"
      },
      {
        "feature": "BMI Category",
        "user_value": "Normal",
        "signed_contribution": -0.3952,
        "direction": "pushes away from Low",
        "group": "health_indicator"
      },
      {
        "feature": "Diastolic BP",
        "user_value": 76.0,
        "signed_contribution": -0.3018,
        "direction": "pushes away from Low",
        "group": "health_indicator"
      },
      {
        "feature": "Physical Activity Level",
        "user_value": 75.0,
        "signed_contribution": 0.2196,
        "direction": "pushes toward Low",
        "group": "lifestyle"
      },
      {
        "feature": "Age",
        "user_value": 44.0,
        "signed_contribution": -0.1178,
        "direction": "pushes away from Low",
        "group": "context"
      },
      {
        "feature": "Sleep Disorder",
        "user_value": "None",
        "signed_contribution": -0.0674,
        "direction": "pushes away from Low",
        "group": "health_indicator"
      }
    ]
  },
  "cluster": {
    "cluster_id": 0,
    "profile_label": "Variant B Cluster 0 (moderate sleep, moderate activity) [Human Label]",
    "profile_summary": {
      "mean_sleep_duration": 7.61,
      "mean_quality_of_sleep": 8.03,
      "mean_physical_activity": 71.66,
      "mean_heart_rate": 68.75,
      "mean_daily_steps": 7479.78,
      "dominant_gender": "Male",
      "gender_share_pct": 59.0,
      "dominant_bmi": "Normal",
      "dominant_sleep_disorder": "None"
    },
    "user_pca_coordinates": {
      "pc1": 1.234817,
      "pc2": 0.641267
    }
  },
  "recommendations": {
    "field_category": "general wellness information, not medical advice",
    "items": [
      "If you feel persistently overwhelmed, consider talking to a trusted person or a qualified professional."
    ]
  },
  "disclaimer": "Lifestyle-based burnout risk estimate for educational purposes. Not a medical diagnosis."
}
```

#### Validation Error Example (`422 Unprocessable Content`)

When an unseen occupation or out-of-range numeric feature is provided:
```json
{
  "detail": "Field 'occupation' has unseen category 'Astronaut'. Allowed categories: ['Accountant', 'Doctor', 'Engineer', 'Lawyer', 'Manager', 'Nurse', 'Sales Representative', 'Salesperson', 'Scientist', 'Software Engineer', 'Teacher']."
}
```

---

### 3.4 GET `/analytics/clusters`
- **Request**: None
- **Response**: `200 OK`
- Returns:
  - `variant_b_profiles`: The 5 cluster profiles from Phase 4.1.
  - `pca_points_variant_b`: PC1 and PC2 coordinates, cluster assignments, and ground-truth risk for the 132 unique survey records (no raw feature values).

---

### 3.5 GET `/analytics/importance`
- **Request**: None
- **Response**: `200 OK`
- Returns global SHAP feature importance, permutation importance with fold standard deviations, and Spearman rank correlations.

---

### 3.6 GET `/models/comparison`
- **Request**: None
- **Response**: `200 OK`
- Returns cross-validation benchmark results across models for Protocol C (Grouped CV) and Protocol D (Random Stratified CV), with explicit note distinguishing them from Protocol A.

---

## 4. Architectural Notes & Safety Guardrails

1. **Relative Path Resolution**: Artifacts are loaded via `Path(__file__).resolve().parent`, avoiding directory traversal errors regardless of the current working directory.
2. **Leakage Prevention**: All inputs are checked to ensure no forbidden columns (`Stress Level`, `Burnout Risk`, `Burnout Index`, `Lifestyle Score`) can enter the service.
3. **Additive SHAP via Mean-Vector**: Feature contributions on the decision function scale are computed as:
   $$\phi_j = w_j (x_j - \bar{x}_j)$$
   where $\bar{x}$ is the mean of the 132 unique profile rows. Sum of contributions plus intercept exactly equals `decision_function` ($< 10^{-9}$ numerical error).
4. **Static Rule Recommendations**: Recommendations strictly reflect standard sleep duration (< 7h) and sleep quality (<= 6) wellness guidance. No model attributions or heart rate/activity/step metrics are ever used to generate behavioral rules.
