"""Unit tests for unsupervised behavioral analytics (K-Means, PCA, stability)."""

import numpy as np
import pytest

from burnoutlens.analytics import (
    evaluate_kmeans_k_range,
    fit_pca,
    prepare_clustering_data,
)
from burnoutlens.config import FORBIDDEN_COLUMNS, INPUT_FEATURES
from burnoutlens.data import load_raw
from burnoutlens.features import clean_data


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
