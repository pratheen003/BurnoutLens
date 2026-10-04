"""Unsupervised behavioral analytics: K-Means clustering, PCA, and profiling."""

from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from burnoutlens.config import INPUT_FEATURES, NUMERIC_FEATURES
from burnoutlens.features import compute_lifestyle_score, make_target
from burnoutlens.leakage import assert_no_leakage
from burnoutlens.preprocessing import build_preprocessor


from sklearn.preprocessing import StandardScaler

BEHAVIORAL_FEATURES = [
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Daily Steps",
    "Heart Rate",
]


def prepare_clustering_data(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], Any]:
    """Prepare clean, leakage-free clustering matrix from input dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    Tuple[np.ndarray, List[str], Any]
        Transformed dense feature matrix (374, 27), feature names, and fitted preprocessor.
    """
    X = df[INPUT_FEATURES].copy()
    assert_no_leakage(X)

    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    return X_trans, feature_names, preprocessor


def prepare_behavioral_clustering_data(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """Prepare behavioral-only feature matrix (Variant B: 5 lifestyle/physiological features).

    Features: Sleep Duration, Quality of Sleep, Physical Activity Level, Daily Steps, Heart Rate.
    Demographics, Age, BMI, Sleep Disorder, and BP are excluded.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.

    Returns
    -------
    Tuple[np.ndarray, List[str], StandardScaler]
        Transformed dense feature matrix (374, 5), feature names, and fitted StandardScaler.
    """
    X = df[BEHAVIORAL_FEATURES].copy()
    assert_no_leakage(X)

    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X)
    return X_trans, list(BEHAVIORAL_FEATURES), scaler


def evaluate_kmeans_k_range(
    X_trans: np.ndarray,
    X_unique_trans: Optional[np.ndarray] = None,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    n_init: int = 10,
    selection_criterion: str = "full",
) -> Dict[str, Any]:
    """Evaluate K-Means across a range of k values using inertia and silhouette score.

    Computes metrics on the full dataset (374 rows) and optionally on the unique-profile rows (132 rows).

    Pre-specified selection rule:
    - Choose k with the highest silhouette score (on full dataset if selection_criterion="full",
      or on unique rows if selection_criterion="unique").
    - If k=3 is within 0.02 of the best, choose k=3 for continuity with the original project.

    Parameters
    ----------
    X_trans : np.ndarray
        Standard-scaled and one-hot encoded feature matrix.
    X_unique_trans : Optional[np.ndarray]
        Standard-scaled feature matrix on the 132 unique-profile rows.
    k_min : int
        Minimum k (default 2).
    k_max : int
        Maximum k (default 8).
    random_state : int
        Random seed for KMeans.
    n_init : int
        Number of centroid initializations.
    selection_criterion : str
        "full" or "unique".

    Returns
    -------
    Dict[str, Any]
        Dictionary containing per-k metrics, chosen k, selection rationale, and models.
    """
    results_by_k = {}
    best_k = k_min
    best_silhouette = -1.0

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_trans)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X_trans, labels))
        sizes = [int(s) for s in np.bincount(labels)]

        entry = {
            "k": k,
            "inertia": inertia,
            "silhouette": sil,
            "cluster_sizes": sizes,
            "labels": labels,
            "model": km,
        }

        if X_unique_trans is not None:
            km_u = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
            labels_u = km_u.fit_predict(X_unique_trans)
            inertia_u = float(km_u.inertia_)
            sil_u = float(silhouette_score(X_unique_trans, labels_u))
            sizes_u = [int(s) for s in np.bincount(labels_u)]
            entry["unique_inertia"] = inertia_u
            entry["unique_silhouette"] = sil_u
            entry["unique_cluster_sizes"] = sizes_u
            entry["unique_labels"] = labels_u
            entry["unique_model"] = km_u

        results_by_k[k] = entry

        eval_sil = entry["unique_silhouette"] if (selection_criterion == "unique" and X_unique_trans is not None) else sil
        if eval_sil > best_silhouette:
            best_silhouette = eval_sil
            best_k = k

    # Apply pre-specified continuity rule
    crit_label = "132-unique silhouette" if (selection_criterion == "unique" and X_unique_trans is not None) else "full silhouette"
    if 3 in results_by_k:
        k3_entry = results_by_k[3]
        k3_sil = k3_entry["unique_silhouette"] if (selection_criterion == "unique" and X_unique_trans is not None) else k3_entry["silhouette"]
        is_k3_within_threshold = (best_silhouette - k3_sil) <= 0.02
        if is_k3_within_threshold:
            chosen_k = 3
            selection_reason = (
                f"k=3 selected under continuity rule: {crit_label} ({k3_sil:.4f}) is within 0.02 "
                f"of best silhouette ({best_silhouette:.4f} at k={best_k})."
            )
        else:
            chosen_k = best_k
            selection_reason = (
                f"k={best_k} selected with highest {crit_label} ({best_silhouette:.4f}). "
                f"k=3 {crit_label} ({k3_sil:.4f}) is not within 0.02 of the best (diff: {best_silhouette - k3_sil:.4f} > 0.02)."
            )
    else:
        is_k3_within_threshold = False
        chosen_k = best_k
        selection_reason = f"k={best_k} selected with highest {crit_label} ({best_silhouette:.4f}) (k=3 not evaluated in range)."

    return {
        "results_by_k": results_by_k,
        "best_k_by_silhouette": best_k,
        "best_silhouette": best_silhouette,
        "chosen_k": chosen_k,
        "is_k3_within_threshold": is_k3_within_threshold,
        "selection_reason": selection_reason,
        "selection_criterion": selection_criterion,
    }


def compute_demographic_purity(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    chosen_k: int,
) -> Dict[str, Any]:
    """Compute share of clusters that are >=90% dominated by a single gender or occupation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing Gender and Occupation columns.
    cluster_labels : np.ndarray
        Cluster assignments (n_samples,).
    chosen_k : int
        Number of clusters.

    Returns
    -------
    Dict[str, Any]
        Details per cluster and summary purity proportion.
    """
    df_temp = df.copy()
    df_temp["Cluster"] = cluster_labels
    cluster_purity = {}
    pure_clusters_count = 0

    for cid in range(chosen_k):
        sub = df_temp[df_temp["Cluster"] == cid]
        size = len(sub)
        top_gender = str(sub["Gender"].mode()[0])
        gender_share = float(round((sub["Gender"] == top_gender).mean() * 100, 1))

        top_occ = str(sub["Occupation"].mode()[0])
        occ_share = float(round((sub["Occupation"] == top_occ).mean() * 100, 1))

        is_pure = (gender_share >= 90.0) or (occ_share >= 90.0)
        if is_pure:
            pure_clusters_count += 1

        cluster_purity[cid] = {
            "size": size,
            "top_gender": top_gender,
            "gender_share_pct": gender_share,
            "top_occupation": top_occ,
            "occ_share_pct": occ_share,
            "is_pure_90": is_pure,
        }

    purity_rate = float(round(pure_clusters_count / chosen_k * 100, 1))
    return {
        "pure_clusters_count": pure_clusters_count,
        "total_clusters": chosen_k,
        "purity_share_pct": purity_rate,
        "clusters": cluster_purity,
    }


def evaluate_cluster_stability(
    X_trans: np.ndarray,
    df: pd.DataFrame,
    chosen_k: int,
    preprocessor: Any,
    random_state: int = 42,
    n_seeds: int = 10,
) -> Dict[str, Any]:
    """Evaluate sensitivity and stability of chosen clustering solution.

    (a) Sensitivity to duplicate rows: fit on 132 unique profiles and compute ARI with full-data labels.
    (b) Seed stability: fit across 10 random seeds and compute pairwise ARI distribution.

    Parameters
    ----------
    X_trans : np.ndarray
        Full transformed matrix.
    df : pd.DataFrame
        Cleaned dataframe with original rows.
    chosen_k : int
        Number of clusters.
    preprocessor : Any
        Fitted ColumnTransformer.
    random_state : int
        Reference random seed.
    n_seeds : int
        Number of seeds to test (0 to n_seeds-1).

    Returns
    -------
    Dict[str, Any]
        Stability metrics, ARIs, and uniqueness comparisons.
    """
    # 1. Full data clustering at reference seed
    km_full = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=10)
    labels_full = km_full.fit_predict(X_trans)

    # 2. Unique-profile clustering (132 rows)
    df_unique = df.drop_duplicates(subset=INPUT_FEATURES).copy()
    idx_unique = df_unique.index
    X_unique_trans = preprocessor.transform(df_unique[INPUT_FEATURES])

    km_unique = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=10)
    labels_unique = km_unique.fit_predict(X_unique_trans)

    # Compare full labels restricted to the 132 unique rows vs unique clustering labels
    labels_full_on_unique = labels_full[idx_unique]
    ari_unique = float(adjusted_rand_score(labels_full_on_unique, labels_unique))

    unique_sizes_full_mapped = [int(s) for s in np.bincount(labels_full_on_unique, minlength=chosen_k)]
    unique_sizes_direct = [int(s) for s in np.bincount(labels_unique, minlength=chosen_k)]

    # 3. Seed stability across 10 random seeds
    seed_labels = []
    seeds = list(range(n_seeds))
    for s in seeds:
        km_s = KMeans(n_clusters=chosen_k, random_state=s, n_init=10)
        seed_labels.append(km_s.fit_predict(X_trans))

    pairwise_aris = [
        float(adjusted_rand_score(l1, l2)) for l1, l2 in combinations(seed_labels, 2)
    ]
    mean_pairwise_ari = float(np.mean(pairwise_aris))
    std_pairwise_ari = float(np.std(pairwise_aris))
    min_pairwise_ari = float(np.min(pairwise_aris))
    max_pairwise_ari = float(np.max(pairwise_aris))

    is_stable = mean_pairwise_ari >= 0.70

    return {
        "chosen_k": chosen_k,
        "n_unique_rows": len(df_unique),
        "ari_unique_vs_full": ari_unique,
        "unique_sizes_full_mapped": unique_sizes_full_mapped,
        "unique_sizes_direct": unique_sizes_direct,
        "seeds_tested": seeds,
        "mean_pairwise_ari": mean_pairwise_ari,
        "std_pairwise_ari": std_pairwise_ari,
        "min_pairwise_ari": min_pairwise_ari,
        "max_pairwise_ari": max_pairwise_ari,
        "is_stable": is_stable,
    }


def fit_pca(
    X_trans: np.ndarray,
    feature_names: List[str],
    n_components: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit Principal Component Analysis on the transformed matrix.

    Parameters
    ----------
    X_trans : np.ndarray
        Transformed dense feature matrix.
    feature_names : List[str]
        List of 27 feature names.
    n_components : int
        Number of components to evaluate (default 10).
    random_state : int
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Explained variance ratios, cumulative variance, loadings, and 2-D coordinates.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X_trans)

    exp_var_ratio = [float(r) for r in pca.explained_variance_ratio_]
    cum_var_ratio = [float(c) for c in np.cumsum(exp_var_ratio)]
    two_pc_variance = float(np.sum(pca.explained_variance_ratio_[:2]))

    # Loadings for PC1 and PC2
    loadings_pc1 = pd.Series(pca.components_[0], index=feature_names)
    loadings_pc2 = pd.Series(pca.components_[1], index=feature_names)

    pc1_top_pos = [(feat, float(val)) for feat, val in loadings_pc1.sort_values(ascending=False).head(5).items()]
    pc1_top_neg = [(feat, float(val)) for feat, val in loadings_pc1.sort_values(ascending=True).head(5).items()]

    pc2_top_pos = [(feat, float(val)) for feat, val in loadings_pc2.sort_values(ascending=False).head(5).items()]
    pc2_top_neg = [(feat, float(val)) for feat, val in loadings_pc2.sort_values(ascending=True).head(5).items()]

    return {
        "pca_model": pca,
        "coords_2d": coords[:, :2],
        "explained_variance_ratio": exp_var_ratio,
        "cumulative_variance_ratio": cum_var_ratio,
        "two_pc_variance": two_pc_variance,
        "pc1_loadings": {
            "top_positive": pc1_top_pos,
            "top_negative": pc1_top_neg,
        },
        "pc2_loadings": {
            "top_positive": pc2_top_pos,
            "top_negative": pc2_top_neg,
        },
    }


def compute_cluster_profiles(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    chosen_k: int,
) -> Dict[str, Any]:
    """Compute detailed, neutral descriptive cluster profiles.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with original raw/clean feature scales.
    cluster_labels : np.ndarray
        Cluster assignments (374,).
    chosen_k : int
        Number of clusters.

    Returns
    -------
    Dict[str, Any]
        Profile dictionary per cluster, crosstab, and chi-square significance.
    """
    df_prof = df.copy()
    df_prof["Cluster"] = cluster_labels
    if "Burnout Risk" not in df_prof.columns:
        df_prof["Burnout Risk"] = make_target(df_prof)
    if "Lifestyle Score" not in df_prof.columns:
        df_prof["Lifestyle Score"] = compute_lifestyle_score(df_prof)

    total_rows = len(df_prof)
    profiles = {}

    # Neutral descriptive labels derived strictly from measured values (human labels, not diagnoses)
    # Neutral labels will be auto-generated or templated from dominant numeric & categorical traits
    for cid in range(chosen_k):
        c_sub = df_prof[df_prof["Cluster"] == cid]
        size = len(c_sub)
        pct = float(size / total_rows * 100)

        # Numeric features: mean and median
        num_stats = {}
        for col in NUMERIC_FEATURES:
            num_stats[col] = {
                "mean": float(round(c_sub[col].mean(), 2)),
                "median": float(round(c_sub[col].median(), 2)),
                "std": float(round(c_sub[col].std(), 2)),
            }

        # Categorical features: mode and share
        gender_mode = str(c_sub["Gender"].mode()[0])
        gender_share = float(round((c_sub["Gender"] == gender_mode).mean() * 100, 1))

        bmi_mode = str(c_sub["BMI Category"].mode()[0])
        bmi_share = float(round((c_sub["BMI Category"] == bmi_mode).mean() * 100, 1))

        sleep_dis_mode = str(c_sub["Sleep Disorder"].mode()[0])
        sleep_dis_share = float(round((c_sub["Sleep Disorder"] == sleep_dis_mode).mean() * 100, 1))

        # Top 3 occupations
        top_occ = [
            {"occupation": str(occ), "count": int(cnt), "share_pct": float(round(cnt / size * 100, 1))}
            for occ, cnt in c_sub["Occupation"].value_counts().head(3).items()
        ]

        # Post-hoc only metrics
        stress_mean = float(round(c_sub["Stress Level"].mean(), 2))
        lifestyle_score_mean = float(round(c_sub["Lifestyle Score"].mean(), 2))

        risk_dist = {}
        for risk in ["Low", "Medium", "High"]:
            cnt = int((c_sub["Burnout Risk"] == risk).sum())
            risk_dist[risk] = {
                "count": cnt,
                "share": float(round(cnt / size * 100, 1)),
            }

        # Formulate neutral descriptive label based on sleep, activity, BP, and BMI
        sleep_str = "longer sleep" if num_stats["Sleep Duration"]["mean"] >= 7.5 else ("shorter sleep" if num_stats["Sleep Duration"]["mean"] <= 6.5 else "moderate sleep")
        act_str = "high activity" if num_stats["Physical Activity Level"]["mean"] >= 70 else ("low activity" if num_stats["Physical Activity Level"]["mean"] <= 45 else "moderate activity")
        bp_str = "elevated BP" if num_stats["Systolic BP"]["mean"] >= 135 else "normal BP"
        human_label = f"{sleep_str}, {act_str}, {bp_str}"

        profiles[cid] = {
            "cluster_id": cid,
            "size": size,
            "share_pct": pct,
            "human_descriptive_label": human_label,
            "numeric_stats": num_stats,
            "categorical_modes": {
                "gender": {"mode": gender_mode, "share_pct": gender_share},
                "bmi_category": {"mode": bmi_mode, "share_pct": bmi_share},
                "sleep_disorder": {"mode": sleep_dis_mode, "share_pct": sleep_dis_share},
                "top_occupations": top_occ,
            },
            "post_hoc_metrics": {
                "mean_stress_level": stress_mean,
                "mean_lifestyle_score": lifestyle_score_mean,
                "burnout_risk_distribution": risk_dist,
            },
        }

    # Crosstab with Burnout Risk
    ct = pd.crosstab(df_prof["Cluster"], df_prof["Burnout Risk"])
    chi2_val, p_val, dof, _ = chi2_contingency(ct)

    crosstab_dict = {}
    for cid in range(chosen_k):
        crosstab_dict[cid] = {
            risk: int(ct.loc[cid, risk]) if risk in ct.columns and cid in ct.index else 0
            for risk in ["Low", "Medium", "High"]
        }

    return {
        "chosen_k": chosen_k,
        "profiles": profiles,
        "crosstab": crosstab_dict,
        "chi2_test": {
            "chi2": float(round(chi2_val, 2)),
            "p_value": float(p_val),
            "degrees_of_freedom": int(dof),
            "significant_at_05": bool(p_val < 0.05),
        },
    }
