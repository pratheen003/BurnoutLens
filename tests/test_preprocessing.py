"""Tests for BurnoutLens Phase 2 preprocessing and feature engineering."""

import hashlib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

from burnoutlens.config import (
    FORBIDDEN_COLUMNS,
    INPUT_FEATURES,
    RAW_CSV_PATH,
)
from burnoutlens.data import load_raw
from burnoutlens.features import (
    add_duplicate_group_id,
    clean_data,
    compute_lifestyle_score,
    make_target,
)
from burnoutlens.leakage import assert_no_leakage
from burnoutlens.preprocessing import build_preprocessor

pytestmark = pytest.mark.skipif(
    not RAW_CSV_PATH.exists(),
    reason="Raw dataset CSV not found at: data/raw/Sleep_health_and_lifestyle_dataset.csv",
)


@pytest.fixture
def raw_df():
    """Load raw dataset fixture."""
    return load_raw()


@pytest.fixture
def cleaned_df(raw_df):
    """Clean dataset fixture."""
    return clean_data(raw_df)


def test_clean_data_shape_and_columns(raw_df, cleaned_df):
    """Test clean_data keeps 374 rows, no NaN, columns equal INPUT_FEATURES + Stress Level."""
    assert len(cleaned_df) == 374
    assert cleaned_df.isna().sum().sum() == 0

    expected_cols = INPUT_FEATURES + ["Stress Level"]
    assert list(cleaned_df.columns) == expected_cols


def test_target_counts(cleaned_df):
    """Test target counts Low 141 / Medium 113 / High 120."""
    target = make_target(cleaned_df)
    counts = target.value_counts().to_dict()

    assert counts.get("Low") == 141
    assert counts.get("Medium") == 113
    assert counts.get("High") == 120
    assert len(target) == 374


def test_bmi_normalization(cleaned_df):
    """Test Normal Weight no longer exists; BMI categories are Normal / Overweight / Obese."""
    bmi_cats = set(cleaned_df["BMI Category"].unique())
    assert "Normal Weight" not in bmi_cats
    assert bmi_cats == {"Normal", "Overweight", "Obese"}

    # Exact expected counts after merge
    counts = cleaned_df["BMI Category"].value_counts().to_dict()
    assert counts["Normal"] == 216
    assert counts["Overweight"] == 148
    assert counts["Obese"] == 10


def test_sleep_disorder_values(cleaned_df):
    """Test Sleep Disorder values are exactly {'None', 'Sleep Apnea', 'Insomnia'}."""
    sd_vals = set(cleaned_df["Sleep Disorder"].unique())
    assert sd_vals == {"None", "Sleep Apnea", "Insomnia"}

    counts = cleaned_df["Sleep Disorder"].value_counts().to_dict()
    assert counts["None"] == 219
    assert counts["Sleep Apnea"] == 78
    assert counts["Insomnia"] == 77


def test_blood_pressure_split_and_drop(cleaned_df):
    """Test Systolic/Diastolic are ints and the raw Blood Pressure column is gone."""
    assert "Blood Pressure" not in cleaned_df.columns
    assert "Systolic BP" in cleaned_df.columns
    assert "Diastolic BP" in cleaned_df.columns

    assert np.issubdtype(cleaned_df["Systolic BP"].dtype, np.integer)
    assert np.issubdtype(cleaned_df["Diastolic BP"].dtype, np.integer)


def test_assert_no_leakage(cleaned_df):
    """Test assert_no_leakage passes on final X and raises when forbidden cols added."""
    X = cleaned_df[INPUT_FEATURES].copy()

    # Must pass without exception
    assert_no_leakage(X)

    # Must raise ValueError for each forbidden column
    for forbidden in ["Burnout Risk", "Stress Level", "Burnout Index", "Lifestyle Score"]:
        leaked_df = X.copy()
        leaked_df[forbidden] = 1
        with pytest.raises(ValueError, match="Data leakage detected"):
            assert_no_leakage(leaked_df)

    # Test all forbidden columns defined in config
    for col in FORBIDDEN_COLUMNS:
        leaked_df = X.copy()
        leaked_df[col] = 1
        with pytest.raises(ValueError, match="Data leakage detected"):
            assert_no_leakage(leaked_df)


def test_immutability_and_raw_integrity(raw_df):
    """Test clean_data does not mutate input and raw CSV sha256 is unchanged."""
    raw_copy_before = raw_df.copy(deep=True)
    _ = clean_data(raw_df)

    # Verify input DataFrame was not mutated
    pd.testing.assert_frame_equal(raw_df, raw_copy_before)

    # Verify raw CSV on disk remains bit-for-bit identical
    expected_hash = "1EFE7B6F781FF88078D08D81FE136CFF98B1B0C32560F35F8650CA5984B77841"
    with open(RAW_CSV_PATH, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest().upper()
    assert actual_hash == expected_hash


def test_preprocessor_unfitted_and_transform(cleaned_df):
    """Test preprocessor is unfitted, transforms test split, and handles unseen category."""
    preprocessor = build_preprocessor()

    X = cleaned_df[INPUT_FEATURES]
    y = make_target(cleaned_df)

    # Must be unfitted initially
    with pytest.raises(NotFittedError):
        preprocessor.transform(X)

    # Split train/test
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit on train split only
    preprocessor.fit(X_train)

    # Transform test split
    X_test_trans = preprocessor.transform(X_test)
    assert X_test_trans.shape[0] == len(X_test)
    assert X_test_trans.shape[1] == len(preprocessor.get_feature_names_out())

    # Transform row with UNSEEN Occupation without crashing (handle_unknown='ignore')
    unseen_row = X_test.iloc[[0]].copy()
    unseen_row.loc[unseen_row.index[0], "Occupation"] = "Quantum Astrophysicist"
    unseen_trans = preprocessor.transform(unseen_row)
    assert unseen_trans.shape[0] == 1
    assert unseen_trans.shape[1] == X_test_trans.shape[1]


def test_duplicate_group_id(cleaned_df):
    """Test add_duplicate_group_id leaves 374 rows and yields expected number of groups."""
    df_with_groups = add_duplicate_group_id(cleaned_df)

    assert len(df_with_groups) == 374
    assert "dup_group" in df_with_groups.columns

    # Verify number of distinct duplicate groups
    num_groups = df_with_groups["dup_group"].nunique()
    assert num_groups == 132


def test_compute_lifestyle_score_hand_checkable(raw_df):
    """Test compute_lifestyle_score matches notebook values on hand-checkable rows."""
    scores = compute_lifestyle_score(raw_df)

    # Verified reference values from notebook Cell 37:
    expected_hand_checks = {
        0: 61.07,
        1: 81.67,
        2: 81.67,
        3: 37.56,
        4: 37.56,
        5: 37.56,
        6: 52.22,
        7: 88.22,
    }

    for idx, expected_score in expected_hand_checks.items():
        assert pytest.approx(scores.iloc[idx], abs=1e-2) == expected_score
