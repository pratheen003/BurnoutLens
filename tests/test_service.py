"""Tests for BurnoutLens prediction service (src/burnoutlens/service.py)."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import shap

from burnoutlens.analytics import BEHAVIORAL_FEATURES
from burnoutlens.config import INPUT_FEATURES
from burnoutlens.data import load_raw
from burnoutlens.explain import (
    CLASS_NAMES,
    explain_row,
    fit_explainability_pipeline,
    load_feature_schema,
)
from burnoutlens.features import clean_data
from burnoutlens.service import (
    BurnoutService,
    ServiceValidationError,
    get_service,
)


@pytest.fixture(scope="module")
def df_clean():
    return clean_data(load_raw())


@pytest.fixture(scope="module")
def service():
    return get_service()


def test_artifacts_load_from_different_working_directory(tmp_path, monkeypatch):
    """Ensure artifacts load cleanly via package-relative paths even when CWD is changed."""
    monkeypatch.chdir(tmp_path)
    assert Path.cwd() == tmp_path

    # Instantiate new service instance from tmp_path
    svc = BurnoutService()
    assert svc.pipeline is not None
    assert len(svc.bg_mean) == 27
    assert svc.behavior_kmeans is not None
    assert len(svc.pca_points_variant_b) == 132
    assert len(svc.feature_schema) == 12


def test_predict_equals_pipeline_direct_for_10_rows(df_clean, service):
    """Ensure service.predict matches direct pipeline predict & predict_proba for 10 real rows."""
    test_rows = df_clean.head(10)
    for idx, row in test_rows.iterrows():
        row_dict = row[INPUT_FEATURES].to_dict()
        clean_row = service.validate_and_standardize_input(row_dict)

        res = service.predict(clean_row)

        df_row = pd.DataFrame([clean_row])[INPUT_FEATURES]
        direct_probs = service.pipeline.predict_proba(df_row)[0]
        direct_pred_idx = int(np.argmax(direct_probs))
        direct_pred_class = CLASS_NAMES[direct_pred_idx]

        assert res["predicted_class"] == direct_pred_class
        for i, c_name in enumerate(CLASS_NAMES):
            assert abs(res["model_probability"][c_name] - direct_probs[i]) < 1e-4


def test_contributions_base_plus_sum_equals_decision_function(df_clean, service):
    """Ensure base_value + sum(contributions) == decision_function within 1e-9 on raw scale."""
    for idx in [32, 0, 1]:  # Low, Medium, High representative rows
        row_dict = df_clean.loc[idx, INPUT_FEATURES].to_dict()
        clean_row = service.validate_and_standardize_input(row_dict)

        pred_res = service.predict(clean_row)
        pred_class = pred_res["predicted_class"]
        c_idx = CLASS_NAMES.index(pred_class)

        df_row = pd.DataFrame([clean_row])[INPUT_FEATURES]
        dec_fn = service.pipeline.decision_function(df_row)[0, c_idx]

        # Calculate exact unrounded components
        x_trans = service.pipeline.named_steps["preprocessor"].transform(df_row)[0]
        lr = service.pipeline.named_steps["classifier"]
        w = lr.coef_[c_idx]
        b = lr.intercept_[c_idx]

        diff = x_trans - service.bg_mean
        shap_trans = w * diff
        base_val = float(b + np.dot(w, service.bg_mean))
        sum_contribs = float(np.sum(shap_trans))

        # Additivity verification: base + sum(contributions) == decision_function
        assert abs((base_val + sum_contribs) - dec_fn) < 1e-9


def test_contributions_equal_phase_5_worked_examples(df_clean, service):
    """Ensure contributions match Phase 5 explain_row within 1e-6 on the 3 worked examples."""
    pipe, explainer, _, trans_names, _ = fit_explainability_pipeline(df_clean, random_state=42)
    schema = load_feature_schema()

    # The 3 representative indices from Phase 5
    for idx in [32, 0, 1]:
        row_dict = df_clean.loc[idx, INPUT_FEATURES].to_dict()
        clean_row = service.validate_and_standardize_input(row_dict)

        ex = explain_row(pipe, explainer, row_dict, schema=schema)
        contribs_res = service.contributions(clean_row, predicted_class=ex["predicted_class"])

        assert abs(contribs_res["base_value"] - ex["base_value"]) < 1e-6
        service_features_map = {item["feature"]: item["signed_contribution"] for item in contribs_res["features"]}

        for top_item in ex["top_features"]:
            feat = top_item["feature"]
            expected_contrib = top_item["signed_contribution"]
            actual_contrib = service_features_map[feat]
            assert abs(actual_contrib - expected_contrib) < 1e-6


def test_mean_vector_formula_equals_shap_linear_explainer(df_clean, service):
    """Ensure the mean-vector formula matches shap.LinearExplainer within 1e-8."""
    pipe, explainer, _, _, _ = fit_explainability_pipeline(df_clean, random_state=42)

    for idx in [32, 0, 1]:
        row_dict = df_clean.loc[idx, INPUT_FEATURES].to_dict()
        clean_row = service.validate_and_standardize_input(row_dict)

        pred_class = service.predict(clean_row)["predicted_class"]
        c_idx = CLASS_NAMES.index(pred_class)

        df_row = pd.DataFrame([clean_row])[INPUT_FEATURES]
        x_trans = service.pipeline.named_steps["preprocessor"].transform(df_row)[0]

        # Mean-vector calculation
        lr = service.pipeline.named_steps["classifier"]
        w = lr.coef_[c_idx]
        diff = x_trans - service.bg_mean
        shap_formula = w * diff

        # Direct shap.LinearExplainer calculation
        shap_explanation = explainer(x_trans.reshape(1, -1))
        shap_explainer_vals = shap_explanation.values[0, :, c_idx]

        assert np.allclose(shap_formula, shap_explainer_vals, atol=1e-8)


def test_cluster_assignment_and_dataset_cluster_sizes(df_clean, service):
    """Ensure service.cluster matches kmeans.predict and reproduces sizes [178, 34, 22, 108, 32]."""
    # 1. Single row check
    sample_row = df_clean.iloc[0][INPUT_FEATURES].to_dict()
    clean_row = service.validate_and_standardize_input(sample_row)
    c_res = service.cluster(clean_row)

    df_behav = pd.DataFrame([{f: clean_row[f] for f in BEHAVIORAL_FEATURES}])[BEHAVIORAL_FEATURES]
    scaled_behav = service.behavior_scaler.transform(df_behav)
    direct_cid = int(service.behavior_kmeans.predict(scaled_behav)[0])

    assert c_res["cluster_id"] == direct_cid
    assert "profile_label" in c_res
    assert "user_pca_coordinates" in c_res
    assert "pc1" in c_res["user_pca_coordinates"]
    assert "pc2" in c_res["user_pca_coordinates"]

    # 2. Reproduction of cluster sizes on all 374 rows
    all_behav_scaled = service.behavior_scaler.transform(df_clean[BEHAVIORAL_FEATURES])
    all_labels = service.behavior_kmeans.predict(all_behav_scaled)
    cluster_counts = [int(s) for s in np.bincount(all_labels, minlength=5)]

    assert cluster_counts == [178, 34, 22, 108, 32]


def test_recommendations_content_and_omissions(service):
    """Ensure recommendations are static wellness guidelines and never advise on steps/activity/HR/BP/BMI."""
    # Row with short sleep and low sleep quality
    row_bad_sleep = {
        "gender": "Female",
        "age": 30,
        "occupation": "Nurse",
        "sleep_duration": 5.8,
        "quality_of_sleep": 5,
        "physical_activity_level": 80,
        "bmi_category": "Normal",
        "heart_rate": 78,
        "daily_steps": 9000,
        "sleep_disorder": "None",
        "systolic_bp": 130,
        "diastolic_bp": 85,
    }
    clean_row = service.validate_and_standardize_input(row_bad_sleep)
    rec_res = service.recommendations(clean_row)
    items = rec_res["items"]

    assert any("7-9 hours" in t for t in items)
    assert any("sleep hygiene" in t.lower() for t in items)
    assert any("trusted person or a qualified professional" in t for t in items)

    # Strictly verify NO mentions of steps, physical activity, heart rate, BMI, or BP
    forbidden_terms = [
        "step",
        "steps",
        "physical activity",
        "exercise",
        "heart rate",
        "bpm",
        "blood pressure",
        "systolic",
        "diastolic",
        "bmi",
        "weight",
    ]
    for text in items:
        lower_t = text.lower()
        for term in forbidden_terms:
            assert term not in lower_t, f"Forbidden advisory term '{term}' found in recommendation text: '{text}'"


def test_input_validation_errors(service):
    """Verify validation errors for unseen categories, out-of-range numerics, missing fields, and forbidden keys."""
    valid_row = {
        "gender": "Female",
        "age": 35,
        "occupation": "Engineer",
        "sleep_duration": 7.5,
        "quality_of_sleep": 8,
        "physical_activity_level": 60,
        "bmi_category": "Normal",
        "heart_rate": 70,
        "daily_steps": 7000,
        "sleep_disorder": "None",
        "systolic_bp": 120,
        "diastolic_bp": 80,
    }

    # 1. Unseen occupation
    bad_occ = valid_row.copy()
    bad_occ["occupation"] = "Astronaut"
    with pytest.raises(ServiceValidationError, match="unseen category"):
        service.validate_and_standardize_input(bad_occ)

    # 2. Out-of-range age
    bad_age = valid_row.copy()
    bad_age["age"] = 200
    with pytest.raises(ServiceValidationError, match="out of allowed range"):
        service.validate_and_standardize_input(bad_age)

    # 3. Missing field
    missing = valid_row.copy()
    del missing["gender"]
    with pytest.raises(ServiceValidationError, match="Missing required field"):
        service.validate_and_standardize_input(missing)

    # 4. Forbidden column (e.g. Stress Level)
    forbidden = valid_row.copy()
    forbidden["stress_level"] = 8
    with pytest.raises(ServiceValidationError, match="Forbidden column"):
        service.validate_and_standardize_input(forbidden)


def test_full_process_request_contains_disclaimer_and_no_forbidden(service):
    """Ensure end-to-end process_request includes disclaimer and contains no forbidden columns."""
    valid_row = {
        "gender": "Male",
        "age": 40,
        "occupation": "Doctor",
        "sleep_duration": 7.0,
        "quality_of_sleep": 7,
        "physical_activity_level": 65,
        "bmi_category": "Normal",
        "heart_rate": 72,
        "daily_steps": 7500,
        "sleep_disorder": "None",
        "systolic_bp": 125,
        "diastolic_bp": 82,
    }

    result = service.process_request(valid_row)

    assert "disclaimer" in result
    assert "educational purposes" in result["disclaimer"].lower()
    assert "prediction" in result
    assert "contributions" in result
    assert "cluster" in result
    assert "recommendations" in result

    # Verify no forbidden keys anywhere in the output
    forbidden_keys = {"Stress Level", "stress_level", "Burnout Risk", "burnout_risk", "Burnout Index", "Lifestyle Score"}
    def check_keys(d):
        if isinstance(d, dict):
            for k, v in d.items():
                assert k not in forbidden_keys, f"Forbidden key '{k}' found in response"
                check_keys(v)
        elif isinstance(d, list):
            for elem in d:
                check_keys(elem)

    check_keys(result)
