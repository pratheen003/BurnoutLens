"""Unit tests for unsupervised behavioral analytics (K-Means, PCA, stability)."""

import numpy as np
import pytest

from burnoutlens.analytics import (
    evaluate_kmeans_k_range,
    fit_pca,
    prepare_clustering_data,
)
from burnoutlens.config import FORBIDDEN_COLUMNS, INPUT_FEATURES, RAW_CSV_PATH
from burnoutlens.data import load_raw
from burnoutlens.features import clean_data

pytestmark = pytest.mark.skipif(
    not RAW_CSV_PATH.exists(),
    reason="Raw dataset CSV not found at: data/raw/Sleep_health_and_lifestyle_dataset.csv",
)


@pytest.fixture
def clean_df():
    """Fixture providing clean DataFrame."""
    return clean_data(load_raw())


def test_clustering_input_no_forbidden_columns(clean_df):
    """Clustering input matrix must not contain any forbidden target/score columns."""
    X_input = clean_df[INPUT_FEATURES].copy()
    for col in FORBIDDEN_COLUMNS:
        assert col not in X_input.columns, f"Forbidden column '{col}' detected in clustering input!"

    X_trans, feature_names, _ = prepare_clustering_data(clean_df)
    assert X_trans.shape == (374, 27)
    for col in FORBIDDEN_COLUMNS:
        for fname in feature_names:
            assert col.lower() not in fname.lower(), f"Forbidden target '{col}' found in transformed feature '{fname}'!"


def test_cluster_sizes_and_labels(clean_df):
    """Cluster sizes must sum to 374 and labels must be strictly within range(k)."""
    X_trans, _, _ = prepare_clustering_data(clean_df)
    eval_res = evaluate_kmeans_k_range(X_trans, k_min=2, k_max=8, random_state=42, n_init=10)
    chosen_k = eval_res["chosen_k"]
    chosen_labels = eval_res["results_by_k"][chosen_k]["labels"]
    cluster_sizes = eval_res["results_by_k"][chosen_k]["cluster_sizes"]

    assert len(chosen_labels) == 374
    assert sum(cluster_sizes) == 374
    unique_labels = set(np.unique(chosen_labels))
    assert unique_labels == set(range(chosen_k))


def test_pca_variance_properties(clean_df):
    """PCA explained_variance_ratio_ must sum to <= 1.0 and cumulative variance must be monotonic."""
    X_trans, feature_names, _ = prepare_clustering_data(clean_df)
    pca_res = fit_pca(X_trans, feature_names, n_components=10, random_state=42)

    ev_ratios = pca_res["explained_variance_ratio"]
    cum_ratios = pca_res["cumulative_variance_ratio"]

    assert len(ev_ratios) == 10
    assert sum(ev_ratios) <= 1.0 + 1e-6
    for ev in ev_ratios:
        assert 0.0 < ev <= 1.0

    # Test monotonicity of cumulative explained variance
    for i in range(len(cum_ratios) - 1):
        assert cum_ratios[i] <= cum_ratios[i + 1]


def test_clustering_reproducibility(clean_df):
    """Rerunning K-Means with the same seed must produce identical cluster labels."""
    X_trans, _, _ = prepare_clustering_data(clean_df)

    run1 = evaluate_kmeans_k_range(X_trans, k_min=3, k_max=3, random_state=42, n_init=10)
    labels1 = run1["results_by_k"][3]["labels"]

    run2 = evaluate_kmeans_k_range(X_trans, k_min=3, k_max=3, random_state=42, n_init=10)
    labels2 = run2["results_by_k"][3]["labels"]

    np.testing.assert_array_equal(labels1, labels2)


def test_variant_b_input_columns_and_no_forbidden(clean_df):
    """Variant B input must have exactly the 5 behavioral columns and no forbidden columns."""
    from burnoutlens.analytics import BEHAVIORAL_FEATURES, prepare_behavioral_clustering_data

    assert len(BEHAVIORAL_FEATURES) == 5
    expected_cols = {
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Daily Steps",
        "Heart Rate",
    }
    assert set(BEHAVIORAL_FEATURES) == expected_cols

    for col in FORBIDDEN_COLUMNS:
        assert col not in BEHAVIORAL_FEATURES

    X_b, feature_names, _ = prepare_behavioral_clustering_data(clean_df)
    assert X_b.shape == (374, 5)
    assert feature_names == list(BEHAVIORAL_FEATURES)
    for col in FORBIDDEN_COLUMNS:
        assert col not in feature_names


def test_variant_b_cluster_sizes_and_labels(clean_df):
    """Variant B cluster sizes must sum to 374 and labels must be strictly within range(k)."""
    from burnoutlens.analytics import BEHAVIORAL_FEATURES, prepare_behavioral_clustering_data
    from burnoutlens.config import INPUT_FEATURES

    X_b_full, _, scaler_b = prepare_behavioral_clustering_data(clean_df)
    df_unique = clean_df.drop_duplicates(subset=INPUT_FEATURES).copy()
    X_b_unique = scaler_b.transform(df_unique[BEHAVIORAL_FEATURES])

    eval_b = evaluate_kmeans_k_range(
        X_b_full, X_unique_trans=X_b_unique, k_min=2, k_max=8, random_state=42, n_init=10, selection_criterion="unique"
    )
    chosen_k = eval_b["chosen_k"]
    assert chosen_k == 5

    labels = eval_b["results_by_k"][chosen_k]["labels"]
    sizes = eval_b["results_by_k"][chosen_k]["cluster_sizes"]
    assert len(labels) == 374
    assert sum(sizes) == 374
    assert set(np.unique(labels)) == set(range(chosen_k))


def test_variant_b_determinism(clean_df):
    """Rerunning Variant B K-Means with the same seed must produce identical labels."""
    from burnoutlens.analytics import prepare_behavioral_clustering_data

    X_b_full, _, _ = prepare_behavioral_clustering_data(clean_df)

    run1 = evaluate_kmeans_k_range(X_b_full, k_min=5, k_max=5, random_state=42, n_init=10)
    labels1 = run1["results_by_k"][5]["labels"]

    run2 = evaluate_kmeans_k_range(X_b_full, k_min=5, k_max=5, random_state=42, n_init=10)
    labels2 = run2["results_by_k"][5]["labels"]

    np.testing.assert_array_equal(labels1, labels2)

