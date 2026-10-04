"""Unit tests for Phase 5 explainability (SHAP, feature mapping, local explanations)."""

import numpy as np
import pytest

from burnoutlens.config import FORBIDDEN_COLUMNS, INPUT_FEATURES
from burnoutlens.data import load_raw
from burnoutlens.explain import (
    aggregate_shap_to_original,
    explain_row,
    fit_explainability_pipeline,
    load_feature_schema,
    map_transformed_to_original,
)
from burnoutlens.features import clean_data


@pytest.fixture
def clean_df():
    """Fixture providing clean DataFrame."""
    return clean_data(load_raw())


@pytest.fixture
def fitted_components(clean_df):
    """Fixture providing fitted pipeline and explainer."""
    pipe, explainer, X_trans, trans_names, orig_map = fit_explainability_pipeline(clean_df, random_state=42)
    return pipe, explainer, X_trans, trans_names, orig_map


def test_every_transformed_column_maps_to_exactly_one_original_feature(fitted_components):
    """Every transformed column must map to exactly one original feature in INPUT_FEATURES."""
    _, _, _, trans_names, orig_map = fitted_components

    assert len(trans_names) == 27
    assert len(orig_map) == 27

    # Check mapping validity
    for orig in orig_map:
        assert orig in INPUT_FEATURES
        assert orig not in FORBIDDEN_COLUMNS

    # Ensure all 12 original features are covered in the mapping
    assert set(orig_map) == set(INPUT_FEATURES)

    # Calling map_transformed_to_original directly must reproduce identical results
    reproduced_map = map_transformed_to_original(trans_names)
    assert reproduced_map == orig_map


def test_shap_additivity(fitted_components):
    """SHAP additivity: base_value + sum(shap_values) == decision_function within 1e-6.

    Scale used: decision_function (log-odds scale for multiclass Logistic Regression).
    """
    pipe, explainer, X_trans, _, _ = fitted_components
    classifier = pipe.named_steps["classifier"]

    # Compute decision function on sample rows (first 5 samples)
    sample_X = X_trans[:5]
    decision_out = classifier.decision_function(sample_X)  # shape (5, 3)

    explanation = explainer(sample_X)
    shap_vals = explanation.values  # shape (5, 27, 3)
    base_vals = explanation.base_values  # shape (5, 3)

    # Verify additivity for each class
    for c in range(3):
        reconstructed = base_vals[:, c] + shap_vals[:, :, c].sum(axis=1)
        np.testing.assert_allclose(
            reconstructed,
            decision_out[:, c],
            atol=1e-6,
            rtol=1e-6,
            err_msg=f"SHAP additivity violated on decision_function scale for class {c}",
        )


def test_explain_row_output_contains_only_input_features_no_forbidden(clean_df, fitted_components):
    """explain_row output must only reference INPUT_FEATURES and never contain FORBIDDEN_COLUMNS."""
    pipe, explainer, _, _, _ = fitted_components
    schema = load_feature_schema()

    sample_row = clean_df.iloc[0][INPUT_FEATURES].to_dict()
    res = explain_row(pipe, explainer, sample_row, schema=schema)

    assert "predicted_class" in res
    assert res["predicted_class"] in ["Low", "Medium", "High"]
    assert "class_probabilities" in res
    assert "base_value" in res
    assert "top_features" in res
    assert len(res["top_features"]) == 5

    for item in res["top_features"]:
        feat = item["feature"]
        assert feat in INPUT_FEATURES
        assert feat not in FORBIDDEN_COLUMNS
        assert "user_value" in item
        assert "signed_contribution" in item
        assert "abs_contribution" in item
        assert "direction" in item


def test_explain_row_raises_on_unseen_category_and_out_of_range_numeric(clean_df, fitted_components):
    """explain_row must raise a clear ValueError on unseen category and out-of-range numeric."""
    pipe, explainer, _, _, _ = fitted_components
    schema = load_feature_schema()

    valid_row = clean_df.iloc[0][INPUT_FEATURES].to_dict()

    # Test 1: Unseen category in Occupation
    bad_cat_row = valid_row.copy()
    bad_cat_row["Occupation"] = "Astronaut"
    with pytest.raises(ValueError, match="unseen category"):
        explain_row(pipe, explainer, bad_cat_row, schema=schema)

    # Test 2: Out of range numeric in Sleep Duration (max is 8.5)
    bad_num_row = valid_row.copy()
    bad_num_row["Sleep Duration"] = 12.0
    with pytest.raises(ValueError, match="outside allowed range"):
        explain_row(pipe, explainer, bad_num_row, schema=schema)

    # Test 3: Out of range numeric below min (min Age is 27)
    bad_age_row = valid_row.copy()
    bad_age_row["Age"] = 15
    with pytest.raises(ValueError, match="outside allowed range"):
        explain_row(pipe, explainer, bad_age_row, schema=schema)


def test_explainability_determinism(clean_df):
    """Rerunning fit_explainability_pipeline and explain_row with same seed gives identical outputs."""
    pipe1, explainer1, _, _, _ = fit_explainability_pipeline(clean_df, random_state=42)
    pipe2, explainer2, _, _, _ = fit_explainability_pipeline(clean_df, random_state=42)

    sample_row = clean_df.iloc[10][INPUT_FEATURES].to_dict()
    res1 = explain_row(pipe1, explainer1, sample_row)
    res2 = explain_row(pipe2, explainer2, sample_row)

    assert res1["predicted_class"] == res2["predicted_class"]
    assert res1["base_value"] == res2["base_value"]
    assert res1["class_probabilities"] == res2["class_probabilities"]
    assert len(res1["top_features"]) == len(res2["top_features"])

    for f1, f2 in zip(res1["top_features"], res2["top_features"]):
        assert f1["feature"] == f2["feature"]
        assert f1["signed_contribution"] == f2["signed_contribution"]
        assert f1["direction"] == f2["direction"]
