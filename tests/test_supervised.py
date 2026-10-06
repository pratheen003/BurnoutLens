"""Tests for Phase 3 supervised modeling and evaluation pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from burnoutlens.config import INPUT_FEATURES, NUMERIC_FEATURES, RAW_CSV_PATH
from burnoutlens.data import load_raw
from burnoutlens.features import add_duplicate_group_id, clean_data, make_target
from burnoutlens.leakage import assert_no_leakage
from burnoutlens.modeling import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    get_models,
    make_pipeline,
)

pytestmark = pytest.mark.skipif(
    not RAW_CSV_PATH.exists(),
    reason="Raw dataset CSV not found at: data/raw/Sleep_health_and_lifestyle_dataset.csv",
)


@pytest.fixture
def dataset_fixtures():
    raw_df = load_raw()
    cleaned = clean_data(raw_df)
    target = make_target(cleaned)
    df_groups = add_duplicate_group_id(cleaned)
    X = df_groups[INPUT_FEATURES].copy()
    y = target.map(LABEL_TO_INT)
    groups = df_groups["dup_group"]
    return X, y, groups


def test_y_explicit_mapping(dataset_fixtures):
    """Test that y uses explicit mapping Low=0, Medium=1, High=2."""
    _, y, _ = dataset_fixtures

    assert LABEL_TO_INT == {"Low": 0, "Medium": 1, "High": 2}
    assert INT_TO_LABEL == {0: "Low", 1: "Medium", 2: "High"}

    unique_vals = set(y.unique())
    assert unique_vals == {0, 1, 2}

    counts = y.value_counts().to_dict()
    assert counts[0] == 141  # Low
    assert counts[1] == 113  # Medium
    assert counts[2] == 120  # High


def test_assert_no_leakage_on_training_features(dataset_fixtures):
    """Test assert_no_leakage passes on X used for training."""
    X, _, _ = dataset_fixtures
    assert_no_leakage(X)
    assert list(X.columns) == INPUT_FEATURES


def test_no_group_overlap_protocol_b_and_c(dataset_fixtures):
    """Test no group overlap between train and test in protocol B and C splits."""
    X, y, groups = dataset_fixtures

    # Protocol B: Grouped holdout across seeds 0 to 4
    for seed in range(5):
        sgkf_b = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, test_idx = next(sgkf_b.split(X, y, groups))
        g_train = set(groups.iloc[train_idx])
        g_test = set(groups.iloc[test_idx])
        overlap = g_train.intersection(g_test)
        assert len(overlap) == 0, f"Protocol B seed {seed} has group overlap: {overlap}"

    # Protocol C: Grouped 5-Fold CV (Seed 42)
    sgkf_c = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(sgkf_c.split(X, y, groups)):
        g_train = set(groups.iloc[train_idx])
        g_val = set(groups.iloc[val_idx])
        overlap = g_train.intersection(g_val)
        assert len(overlap) == 0, f"Protocol C fold {fold} has group overlap: {overlap}"


def test_pipeline_preprocessor_fit_on_train_only(dataset_fixtures):
    """Test pipeline preprocessor is fit on training rows only."""
    X, y, _ = dataset_fixtures

    # Use first 100 rows as training subset
    train_subset_X = X.iloc[:100].copy()
    train_subset_y = y.iloc[:100].copy()

    model = get_models()["Logistic Regression"]
    pipeline = make_pipeline(model)

    pipeline.fit(train_subset_X, train_subset_y)

    # Extract fitted scaler means from pipeline
    fitted_means = pipeline.named_steps["prep"].named_transformers_["num"].mean_

    # Compute expected mean on train subset only
    expected_train_means = train_subset_X[NUMERIC_FEATURES].mean().to_numpy()

    # Compute full dataset mean
    full_dataset_means = X[NUMERIC_FEATURES].mean().to_numpy()

    # Must equal train subset mean
    np.testing.assert_allclose(fitted_means, expected_train_means, rtol=1e-5)

    # Must NOT equal full dataset mean (verifying scaler didn't leak full data)
    assert not np.allclose(fitted_means, full_dataset_means, rtol=1e-3)


def test_protocol_d_split_and_pipeline(dataset_fixtures):
    """Test that Protocol D split does not require groups and fits preprocessing on train folds only."""
    from sklearn.model_selection import StratifiedKFold
    from burnoutlens.evaluation import run_protocol_d

    X, y, groups = dataset_fixtures

    # 1. Verify run_protocol_d executes successfully without groups
    lr_model = {"Logistic Regression": get_models()["Logistic Regression"]}
    res_d_no_groups = run_protocol_d(lr_model, X, y, groups=None, seeds=[0])
    assert "Logistic Regression" in res_d_no_groups["models"]
    assert res_d_no_groups["avg_twin_rows_per_fold"] == 0.0

    # 2. Verify StratifiedKFold split does not require groups and fits preprocessor on train only
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        pipe = make_pipeline(get_models()["Logistic Regression"])
        pipe.fit(X_tr, y_tr)

        fitted_means = pipe.named_steps["prep"].named_transformers_["num"].mean_
        expected_means = X_tr[NUMERIC_FEATURES].mean().to_numpy()
        full_means = X[NUMERIC_FEATURES].mean().to_numpy()

        np.testing.assert_allclose(fitted_means, expected_means, rtol=1e-5)
        assert not np.allclose(fitted_means, full_means, rtol=1e-3)

        preds = pipe.predict(X_val)
        assert len(preds) == len(val_idx)

