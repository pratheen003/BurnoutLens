"""Integration tests for BurnoutLens FastAPI endpoints (src/burnoutlens/api.py)."""

import pytest
from fastapi.testclient import TestClient

from burnoutlens.api import app

client = TestClient(app)

VALID_SAMPLE_PAYLOAD = {
    "gender": "Female",
    "age": 31,
    "occupation": "Nurse",
    "sleep_duration": 7.9,
    "quality_of_sleep": 8,
    "physical_activity_level": 75,
    "bmi_category": "Normal",
    "heart_rate": 69,
    "daily_steps": 6800,
    "sleep_disorder": "None",
    "systolic_bp": 117,
    "diastolic_bp": 76,
}


def test_health_endpoint():
    """GET /health must return 200 and ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "BurnoutLens API" in data["service"]


def test_predict_endpoint_valid():
    """POST /predict with valid payload must return 200 with all required sections."""
    response = client.post("/predict", json=VALID_SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    # Required top-level keys
    assert "prediction" in data
    assert "contributions" in data
    assert "cluster" in data
    assert "recommendations" in data
    assert "disclaimer" in data

    # Verify prediction section
    pred = data["prediction"]
    assert pred["predicted_class"] in ["Low", "Medium", "High"]
    assert "model_probability" in pred
    assert set(pred["model_probability"].keys()) == {"Low", "Medium", "High"}
    assert "uncalibrated" in pred["probability_note"].lower()

    # Verify contributions section
    contribs = data["contributions"]
    assert "base_value" in contribs
    assert len(contribs["features"]) == 12
    for item in contribs["features"]:
        assert "feature" in item
        assert "signed_contribution" in item
        assert "direction" in item
        assert item["group"] in ["lifestyle", "health_indicator", "context"]

    # Verify cluster section
    cluster = data["cluster"]
    assert "cluster_id" in cluster
    assert cluster["cluster_id"] in [0, 1, 2, 3, 4]
    assert "profile_label" in cluster
    assert "user_pca_coordinates" in cluster
    assert "pc1" in cluster["user_pca_coordinates"]
    assert "pc2" in cluster["user_pca_coordinates"]

    # Verify recommendations section
    recs = data["recommendations"]
    assert "general wellness information, not medical advice" in recs["field_category"]
    assert any("trusted person or a qualified professional" in item for item in recs["items"])

    # Verify disclaimer
    assert "Lifestyle-based burnout risk estimate for educational purposes. Not a medical diagnosis." == data["disclaimer"]


def test_predict_unseen_occupation_returns_422():
    """POST /predict with unseen occupation must return HTTP 422."""
    bad_payload = VALID_SAMPLE_PAYLOAD.copy()
    bad_payload["occupation"] = "Astronaut"

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "unseen category" in str(data["detail"]).lower() or "occupation" in str(data["detail"]).lower()


def test_predict_out_of_range_age_returns_422():
    """POST /predict with age 200 must return HTTP 422."""
    bad_payload = VALID_SAMPLE_PAYLOAD.copy()
    bad_payload["age"] = 200

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "allowed range" in str(data["detail"]).lower() or "age" in str(data["detail"]).lower()


def test_predict_missing_field_returns_422():
    """POST /predict with missing field must return HTTP 422."""
    bad_payload = VALID_SAMPLE_PAYLOAD.copy()
    del bad_payload["sleep_duration"]

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "field required" in str(data["detail"]).lower() or "sleep_duration" in str(data["detail"]).lower()


def test_predict_forbidden_column_returns_422():
    """POST /predict with forbidden column (e.g. stress_level) must return HTTP 422."""
    bad_payload = VALID_SAMPLE_PAYLOAD.copy()
    bad_payload["stress_level"] = 8

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_response_contains_no_forbidden_keys():
    """Verify that response payload never contains forbidden leakage columns."""
    response = client.post("/predict", json=VALID_SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()

    forbidden = {"Stress Level", "stress_level", "Burnout Risk", "burnout_risk", "Burnout Index", "Lifestyle Score"}
    def check_dict(d):
        if isinstance(d, dict):
            for k, v in d.items():
                assert k not in forbidden, f"Forbidden key '{k}' found in response"
                check_dict(v)
        elif isinstance(d, list):
            for item in d:
                check_dict(item)

    check_dict(data)


def test_recommendations_never_mention_steps_activity_hr_bmi_bp():
    """Ensure recommendations never mention steps, physical activity, heart rate, BMI, or BP."""
    response = client.post("/predict", json=VALID_SAMPLE_PAYLOAD)
    assert response.status_code == 200
    items = response.json()["recommendations"]["items"]

    forbidden_terms = [
        "step",
        "physical activity",
        "exercise",
        "heart rate",
        "bpm",
        "blood pressure",
        "systolic",
        "diastolic",
        "bmi",
    ]
    for text in items:
        lower_t = text.lower()
        for term in forbidden_terms:
            assert term not in lower_t, f"Forbidden advisory term '{term}' found in text: {text}"


def test_metadata_endpoint():
    """GET /meta must return 200 with schema, manifest summary, disclaimer, and limitations."""
    response = client.get("/meta")
    assert response.status_code == 200
    data = response.json()

    assert "feature_schema" in data
    assert len(data["feature_schema"]) == 12
    assert "manifest_summary" in data
    assert "selected_model" in data["manifest_summary"]
    assert "disclaimer" in data
    assert "limitations" in data
    assert len(data["limitations"]) >= 4


def test_analytics_clusters_endpoint():
    """GET /analytics/clusters must return 200 with variant_b profiles and 132 PCA points."""
    response = client.get("/analytics/clusters")
    assert response.status_code == 200
    data = response.json()

    assert "variant_b_profiles" in data
    assert set(data["variant_b_profiles"].keys()) == {"0", "1", "2", "3", "4"}
    assert "pca_points_variant_b" in data
    assert len(data["pca_points_variant_b"]) == 132
    assert "disclaimer" in data


def test_analytics_importance_endpoint():
    """GET /analytics/importance must return 200 with global SHAP and permutation importance."""
    response = client.get("/analytics/importance")
    assert response.status_code == 200
    data = response.json()

    assert "global_shap_importance" in data
    assert "permutation_importance" in data
    assert "spearman_correlation_shap_vs_permutation" in data


def test_models_comparison_endpoint():
    """GET /models/comparison must return 200 with Protocol C and D data and comparability note."""
    response = client.get("/models/comparison")
    assert response.status_code == 200
    data = response.json()

    assert "models" in data
    assert "Logistic Regression" in data["models"]
    assert "note" in data
    assert "Protocol A is a single split and not comparable" in data["note"]
    assert "disclaimer" in data
