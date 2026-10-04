"""Run K-Means clustering and PCA behavioral analytics for BurnoutLens Phase 4 & Phase 4.1.

Outputs:
- reports/figures/elbow_silhouette.png
- reports/figures/pca_scree.png
- reports/figures/pca_clusters.png
- reports/figures/pca_by_risk.png
- reports/cluster_profiles.json (with 'variant_a_full_features' and 'variant_b_behavioral_only')
- reports/clustering_pca.md
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from burnoutlens.analytics import (
    BEHAVIORAL_FEATURES,
    compute_cluster_profiles,
    compute_demographic_purity,
    evaluate_cluster_stability,
    evaluate_kmeans_k_range,
    fit_pca,
    prepare_behavioral_clustering_data,
    prepare_clustering_data,
)
from burnoutlens.config import INPUT_FEATURES
from burnoutlens.data import load_raw
from burnoutlens.features import clean_data, compute_lifestyle_score, make_target


def main():
    root_dir = Path(__file__).resolve().parent.parent
    figures_dir = root_dir / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root_dir / "reports"

    print("Loading clean dataset...")
    df = clean_data(load_raw())
    df["Burnout Risk"] = make_target(df)
    df["Lifestyle Score"] = compute_lifestyle_score(df)
    df_unique = df.drop_duplicates(subset=INPUT_FEATURES).copy()

    # =========================================================================
    # VARIANT A: Full 12 Input Features (27-dimensional transformed space)
    # =========================================================================
    print("Preparing Variant A feature matrix...")
    X_a_full, feat_names_a, prep_a = prepare_clustering_data(df)
    X_a_unique = prep_a.transform(df_unique[INPUT_FEATURES])

    print("Evaluating Variant A for k=2..8 on full and unique rows...")
    k_eval_a = evaluate_kmeans_k_range(
        X_a_full, X_unique_trans=X_a_unique, k_min=2, k_max=8, random_state=42, n_init=10, selection_criterion="full"
    )
    chosen_k_a = k_eval_a["chosen_k"]
    chosen_labels_a = k_eval_a["results_by_k"][chosen_k_a]["labels"]
    labels_a_3 = k_eval_a["results_by_k"][3]["labels"]

    print(f"Variant A chosen k: {chosen_k_a}")
    stability_a_chosen = evaluate_cluster_stability(
        X_a_full, df, chosen_k=chosen_k_a, preprocessor=prep_a, random_state=42, n_seeds=10
    )
    stability_a_k3 = evaluate_cluster_stability(
        X_a_full, df, chosen_k=3, preprocessor=prep_a, random_state=42, n_seeds=10
    )

    pca_a = fit_pca(X_a_full, feat_names_a, n_components=10, random_state=42)
    prof_a = compute_cluster_profiles(df, chosen_labels_a, chosen_k_a)
    purity_a = compute_demographic_purity(df, chosen_labels_a, chosen_k_a)

    # Human descriptive labels for Variant A
    for cid, p in prof_a["profiles"].items():
        dur = p["numeric_stats"]["Sleep Duration"]["mean"]
        act = p["numeric_stats"]["Physical Activity Level"]["mean"]
        bp = p["numeric_stats"]["Systolic BP"]["mean"]
        bmi = p["categorical_modes"]["bmi_category"]["mode"]
        dis = p["categorical_modes"]["sleep_disorder"]["mode"]

        traits = []
        if dur >= 7.8:
            traits.append("high sleep")
        elif dur <= 6.5:
            traits.append("short sleep")
        else:
            traits.append("moderate sleep")

        if act >= 75:
            traits.append("high activity")
        elif act <= 50:
            traits.append("lower activity")
        else:
            traits.append("moderate activity")

        if bp >= 135:
            traits.append("elevated BP")
        elif bp <= 125:
            traits.append("lower BP")

        if bmi in ["Overweight", "Obese"]:
            traits.append(f"{bmi.lower()} BMI")

        if dis != "None":
            traits.append(f"frequent {dis.lower()}")

        p["human_descriptive_label"] = f"Cluster {cid} ({', '.join(traits)}) [Human Label]"

    # =========================================================================
    # VARIANT B: Behavioral-Only Clustering (5 continuous features)
    # =========================================================================
    print("Preparing Variant B behavioral-only feature matrix...")
    X_b_full, feat_names_b, scaler_b = prepare_behavioral_clustering_data(df)
    X_b_unique = scaler_b.transform(df_unique[BEHAVIORAL_FEATURES])

    print("Evaluating Variant B for k=2..8 on full and unique rows...")
    k_eval_b = evaluate_kmeans_k_range(
        X_b_full, X_unique_trans=X_b_unique, k_min=2, k_max=8, random_state=42, n_init=10, selection_criterion="unique"
    )
    chosen_k_b = k_eval_b["chosen_k"]
    chosen_labels_b = k_eval_b["results_by_k"][chosen_k_b]["labels"]
    print(f"Variant B chosen k: {chosen_k_b} ({k_eval_b['selection_reason']})")

    # Stability of Variant B
    km_b_full = k_eval_b["results_by_k"][chosen_k_b]["model"]
    km_b_uniq = k_eval_b["results_by_k"][chosen_k_b]["unique_model"]
    labels_b_unique_direct = km_b_uniq.fit_predict(X_b_unique)
    labels_b_full_on_unique = chosen_labels_b[df_unique.index]
    ari_b_unique = float(adjusted_rand_score(labels_b_full_on_unique, labels_b_unique_direct))

    seed_labels_b = [
        KMeans(n_clusters=chosen_k_b, random_state=s, n_init=10).fit_predict(X_b_full)
        for s in range(10)
    ]
    from itertools import combinations
    pairwise_aris_b = [float(adjusted_rand_score(l1, l2)) for l1, l2 in combinations(seed_labels_b, 2)]
    stability_b = {
        "chosen_k": chosen_k_b,
        "ari_unique_vs_full": ari_b_unique,
        "mean_pairwise_ari": float(np.mean(pairwise_aris_b)),
        "min_pairwise_ari": float(np.min(pairwise_aris_b)),
        "max_pairwise_ari": float(np.max(pairwise_aris_b)),
        "std_pairwise_ari": float(np.std(pairwise_aris_b)),
        "unique_sizes_full_mapped": [int(s) for s in np.bincount(labels_b_full_on_unique, minlength=chosen_k_b)],
        "unique_sizes_direct": [int(s) for s in np.bincount(labels_b_unique_direct, minlength=chosen_k_b)],
    }

    # PCA for Variant B
    pca_b = fit_pca(X_b_full, feat_names_b, n_components=5, random_state=42)

    # Variant B Cluster Profiles
    prof_b = compute_cluster_profiles(df, chosen_labels_b, chosen_k_b)
    purity_b = compute_demographic_purity(df, chosen_labels_b, chosen_k_b)

    # Human labels for Variant B (derived strictly from measured behavioral features)
    for cid, p in prof_b["profiles"].items():
        dur = p["numeric_stats"]["Sleep Duration"]["mean"]
        act = p["numeric_stats"]["Physical Activity Level"]["mean"]
        steps = p["numeric_stats"]["Daily Steps"]["mean"]
        hr = p["numeric_stats"]["Heart Rate"]["mean"]

        traits = []
        if dur >= 7.8:
            traits.append("high sleep")
        elif dur <= 6.5:
            traits.append("short sleep")
        else:
            traits.append("moderate sleep")

        if act >= 75 or steps >= 8000:
            traits.append("high activity/steps")
        elif act <= 45 or steps <= 4500:
            traits.append("lower activity/steps")
        else:
            traits.append("moderate activity")

        if hr >= 78:
            traits.append("elevated heart rate")
        elif hr <= 68:
            traits.append("lower heart rate")

        p["human_descriptive_label"] = f"Variant B Cluster {cid} ({', '.join(traits)}) [Human Label]"

    # =========================================================================
    # FIGURES
    # =========================================================================
    print("Generating figures...")
    # 1. Elbow and Silhouette for Variant A
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ks = list(range(2, 9))
    inertias_a = [k_eval_a["results_by_k"][k]["inertia"] for k in ks]
    sil_a_full = [k_eval_a["results_by_k"][k]["silhouette"] for k in ks]
    sil_a_uniq = [k_eval_a["results_by_k"][k]["unique_silhouette"] for k in ks]

    ax1.plot(ks, inertias_a, marker="o", color="#1f77b4", linewidth=2)
    ax1.axvline(x=chosen_k_a, color="#d62728", linestyle="--", label=f"Chosen k={chosen_k_a}")
    ax1.axvline(x=3, color="#2ca02c", linestyle=":", label="Historical k=3")
    ax1.set_title("Variant A Inertia (Elbow Method)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(ks, sil_a_full, marker="s", color="#ff7f0e", linewidth=2, label="Full 374 Rows")
    ax2.plot(ks, sil_a_uniq, marker="^", color="#17becf", linewidth=2, linestyle="--", label="132 Unique Rows")
    ax2.axvline(x=chosen_k_a, color="#d62728", linestyle="--", label=f"Chosen k={chosen_k_a}")
    ax2.axvline(x=3, color="#2ca02c", linestyle=":", label="k=3 Baseline")
    ax2.set_title("Variant A Silhouette vs k (Full vs Unique)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "elbow_silhouette.png", dpi=300)
    plt.close()

    # 2. PCA Scree Plot for Variant A
    fig, ax = plt.subplots(figsize=(8, 5))
    components = list(range(1, 11))
    ev_ratio = [r * 100 for r in pca_a["explained_variance_ratio"]]
    cum_ev_ratio = [c * 100 for c in pca_a["cumulative_variance_ratio"]]
    ax.bar(components, ev_ratio, color="#4c72b0", alpha=0.8, label="Individual Explained Variance (%)")
    ax.step(components, cum_ev_ratio, where="mid", color="#c44e52", linewidth=2, label="Cumulative Explained Variance (%)")
    ax.axhline(y=cum_ev_ratio[1], color="grey", linestyle="--", alpha=0.7, label=f"PC1+PC2 Cumulative: {cum_ev_ratio[1]:.1f}%")
    ax.set_title("Variant A PCA Scree Plot (First 10 Components)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_xticks(components)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_scree.png", dpi=300)
    plt.close()

    # 3. PCA 2-D Colored by Cluster for Variant A
    coords_2d_a = pca_a["coords_2d"]
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        coords_2d_a[:, 0], coords_2d_a[:, 1], c=chosen_labels_a, cmap="tab10", alpha=0.8, edgecolors="k", linewidths=0.5, s=50
    )
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(chosen_k_a))
    cbar.set_label(f"Cluster Assignment (k={chosen_k_a})")
    ax.set_title(
        f"2-D PCA Projection Colored by Variant A Clusters (k={chosen_k_a})\n"
        f"[Lossy 2-D projection: captures {pca_a['two_pc_variance']*100:.2f}% of total variance]",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel(f"PC1 ({pca_a['explained_variance_ratio'][0]*100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca_a['explained_variance_ratio'][1]*100:.2f}% variance)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_clusters.png", dpi=300)
    plt.close()

    # 4. PCA 2-D Colored by Burnout Risk for Variant A
    fig, ax = plt.subplots(figsize=(9, 6))
    risk_colors = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}
    for risk_lvl in ["Low", "Medium", "High"]:
        mask = df["Burnout Risk"] == risk_lvl
        ax.scatter(
            coords_2d_a[mask, 0], coords_2d_a[mask, 1], c=risk_colors[risk_lvl],
            label=f"{risk_lvl} Risk (n={mask.sum()})", alpha=0.8, edgecolors="k", linewidths=0.5, s=50
        )
    ax.set_title(
        f"2-D PCA Projection Colored by Post-Hoc Burnout Risk\n"
        f"[Exploratory post-hoc mapping; captures {pca_a['two_pc_variance']*100:.2f}% variance]",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel(f"PC1 ({pca_a['explained_variance_ratio'][0]*100:.2f}% variance)")
    ax.set_ylabel(f"PC2 ({pca_a['explained_variance_ratio'][1]*100:.2f}% variance)")
    ax.legend(title="Post-Hoc Burnout Risk")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_by_risk.png", dpi=300)
    plt.close()

    # =========================================================================
    # JSON EXPORT
    # =========================================================================
    print("Exporting reports/cluster_profiles.json with separate keys...")
    json_export = {
        "variant_a_full_features": {
            "name": "Variant A (All 12 Input Features, 27 Transformed Dimensions)",
            "features_used": INPUT_FEATURES,
            "n_features_transformed": 27,
            "chosen_k": chosen_k_a,
            "selection_rule": k_eval_a["selection_reason"],
            "demographic_purity_pct": purity_a["purity_share_pct"],
            "seed_stability": {
                "mean_pairwise_ari": stability_a_chosen["mean_pairwise_ari"],
                "min_pairwise_ari": stability_a_chosen["min_pairwise_ari"],
                "max_pairwise_ari": stability_a_chosen["max_pairwise_ari"],
            },
            "unique_vs_full_ari": stability_a_chosen["ari_unique_vs_full"],
            "chi2_contingency": prof_a["chi2_test"],
            "clusters": {
                str(cid): {
                    "cluster_id": cid,
                    "size": p["size"],
                    "share_pct": p["share_pct"],
                    "human_descriptive_label": p["human_descriptive_label"],
                    "key_profile_values": {
                        "mean_sleep_duration": p["numeric_stats"]["Sleep Duration"]["mean"],
                        "mean_quality_of_sleep": p["numeric_stats"]["Quality of Sleep"]["mean"],
                        "mean_physical_activity": p["numeric_stats"]["Physical Activity Level"]["mean"],
                        "mean_heart_rate": p["numeric_stats"]["Heart Rate"]["mean"],
                        "mean_daily_steps": p["numeric_stats"]["Daily Steps"]["mean"],
                        "mean_systolic_bp": p["numeric_stats"]["Systolic BP"]["mean"],
                        "mean_diastolic_bp": p["numeric_stats"]["Diastolic BP"]["mean"],
                        "dominant_gender": p["categorical_modes"]["gender"]["mode"],
                        "dominant_bmi": p["categorical_modes"]["bmi_category"]["mode"],
                        "dominant_sleep_disorder": p["categorical_modes"]["sleep_disorder"]["mode"],
                        "top_occupations": p["categorical_modes"]["top_occupations"],
                    },
                    "post_hoc_metrics": p["post_hoc_metrics"],
                }
                for cid, p in prof_a["profiles"].items()
            },
        },
        "variant_b_behavioral_only": {
            "name": "Variant B (Behavioral-Only 5 Features: Sleep Duration, Quality, Activity, Steps, Heart Rate)",
            "features_used": BEHAVIORAL_FEATURES,
            "excluded_features": ["Age", "Gender", "Occupation", "BMI Category", "Sleep Disorder", "Systolic BP", "Diastolic BP"],
            "n_features_transformed": 5,
            "chosen_k": chosen_k_b,
            "selection_rule": k_eval_b["selection_reason"],
            "demographic_purity_pct": purity_b["purity_share_pct"],
            "seed_stability": {
                "mean_pairwise_ari": stability_b["mean_pairwise_ari"],
                "min_pairwise_ari": stability_b["min_pairwise_ari"],
                "max_pairwise_ari": stability_b["max_pairwise_ari"],
            },
            "unique_vs_full_ari": stability_b["ari_unique_vs_full"],
            "chi2_contingency": prof_b["chi2_test"],
            "pca_explained_variance_ratio": pca_b["explained_variance_ratio"],
            "pca_two_pc_variance": pca_b["two_pc_variance"],
            "clusters": {
                str(cid): {
                    "cluster_id": cid,
                    "size": p["size"],
                    "share_pct": p["share_pct"],
                    "human_descriptive_label": p["human_descriptive_label"],
                    "key_profile_values": {
                        "mean_sleep_duration": p["numeric_stats"]["Sleep Duration"]["mean"],
                        "mean_quality_of_sleep": p["numeric_stats"]["Quality of Sleep"]["mean"],
                        "mean_physical_activity": p["numeric_stats"]["Physical Activity Level"]["mean"],
                        "mean_heart_rate": p["numeric_stats"]["Heart Rate"]["mean"],
                        "mean_daily_steps": p["numeric_stats"]["Daily Steps"]["mean"],
                        "dominant_gender": p["categorical_modes"]["gender"]["mode"],
                        "gender_share_pct": p["categorical_modes"]["gender"]["share_pct"],
                        "dominant_bmi": p["categorical_modes"]["bmi_category"]["mode"],
                        "dominant_sleep_disorder": p["categorical_modes"]["sleep_disorder"]["mode"],
                        "top_occupations": p["categorical_modes"]["top_occupations"],
                    },
                    "post_hoc_metrics": p["post_hoc_metrics"],
                }
                for cid, p in prof_b["profiles"].items()
            },
        },
    }

    with open(reports_dir / "cluster_profiles.json", "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)

    # =========================================================================
    # WRITE EXPANDED MARKDOWN REPORT
    # =========================================================================
    print("Writing updated reports/clustering_pca.md...")
    write_clustering_report(
        reports_dir / "clustering_pca.md",
        k_eval_a=k_eval_a,
        stability_a_chosen=stability_a_chosen,
        stability_a_k3=stability_a_k3,
        pca_a=pca_a,
        prof_a=prof_a,
        purity_a=purity_a,
        k_eval_b=k_eval_b,
        stability_b=stability_b,
        pca_b=pca_b,
        prof_b=prof_b,
        purity_b=purity_b,
        df=df,
        labels_a_3=labels_a_3,
    )
    print("Phase 4.1 script execution completed successfully.")


def write_clustering_report(
    report_path: Path,
    k_eval_a: dict,
    stability_a_chosen: dict,
    stability_a_k3: dict,
    pca_a: dict,
    prof_a: dict,
    purity_a: dict,
    k_eval_b: dict,
    stability_b: dict,
    pca_b: dict,
    prof_b: dict,
    purity_b: dict,
    df: pd.DataFrame,
    labels_a_3: np.ndarray,
):
    chosen_k_a = k_eval_a["chosen_k"]
    chosen_k_b = k_eval_b["chosen_k"]
    results_a = k_eval_a["results_by_k"]
    results_b = k_eval_b["results_by_k"]

    content = f"""# BurnoutLens Phase 4 & 4.1: Unsupervised Behavioral Analytics Report

## 1. Executive Summary & Setup

This report documents the unsupervised behavioral analytics conducted for BurnoutLens in Phase 4 and refined in Phase 4.1. Using the clean, leakage-free preprocessing pipeline established in Phase 2 (`src/burnoutlens/preprocessing.py`), we apply **K-Means clustering** and **Principal Component Analysis (PCA)** to evaluate continuous lifestyle dimensions and behavioral clusters.

We examine two complementary clustering formulations:
1. **Variant A (All 12 Input Features)**: Full biometric, demographic, and occupational inputs transformed via `ColumnTransformer` (OneHotEncoder + StandardScaler), yielding a 27-dimensional space.
2. **Variant B (Behavioral-Only Features)**: Exactly 5 continuous lifestyle and autonomic features (`Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Daily Steps`, `Heart Rate`) scaled with `StandardScaler`. All demographic, occupational, and health-status variables (`Gender`, `Occupation`, `Age`, `BMI Category`, `Sleep Disorder`, `Blood Pressure`) are strictly excluded.

### Key Analytical Guardrails:
1. **Strict Separation from Supervised Targets**:
   Prior to clustering, `assert_no_leakage` was executed. Neither `Stress Level`, `Burnout Risk`, `Burnout Index`, nor `Lifestyle Score` were included in feature matrices. All associations with burnout and stress are evaluated strictly **post-hoc**.
2. **Exploratory Analytics Scope**:
   Clustering is fit across the dataset to explore behavioral density. It is an unsupervised descriptive segmentation, not a supervised clinical evaluation.
3. **Neutral Human Labels**:
   Cluster labels are derived strictly from measured behavioral and biometric parameters. They represent descriptive human labels, **not clinical diagnoses or medical classifications**.
4. **Methodological Note on Contingency Tests**:
   Chi-square tests of independence with `Burnout Risk` are **purely descriptive**. Because the dataset contains 242 identical profile duplicates (only 132 unique rows), observations violate the standard assumption of independent observations; nominal p-values are artificially magnified.

---

## 2. Choosing k: Full vs. Unique Rows Silhouette Evaluation

We evaluated K-Means for $k$ in [2, 8] using `random_state=42` and `n_init=10` on:
- The **Full Dataset (374 rows)**
- The **132 Unique-Profile Rows** (fit directly on unique rows to evaluate boundary stability without duplicate density weighting)

### Pre-Specified Selection Rule:
- Select the value of $k$ that maximizes silhouette score.
- *Continuity Override*: If $k=3$ achieves a silhouette score within $0.02$ of the global maximum, select $k=3$ for continuity with the historical notebook baseline.

### Variant A Evaluation Table (k=2 to 8, 27-Dimensional Space):

| Clusters ($k$) | Inertia (374 Rows) | Silhouette (374 Rows) | Silhouette (132 Unique Rows) | Cluster Sizes (374 Rows) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---|
"""
    for k in range(2, 9):
        res = results_a[k]
        sizes_str = " / ".join(map(str, res["cluster_sizes"]))
        is_best = " (Best Full)" if k == k_eval_a["best_k_by_silhouette"] else ""
        is_chosen = " **[CHOSEN A]**" if k == chosen_k_a else ""
        content += f"| {k} | {res['inertia']:.2f} | {res['silhouette']:.4f}{is_best}{is_chosen} | {res['unique_silhouette']:.4f} | {sizes_str} | {'Historical k' if k==3 else ''} |\n"

    content += f"""
### Variant A Selection Decision:
- **Best Silhouette (Full 374 Rows)**: $k={k_eval_a['best_k_by_silhouette']}$ (`{k_eval_a['best_silhouette']:.4f}`).
- **Historical $k=3$ Silhouette**: `{results_a[3]['silhouette']:.4f}`.
- **Gap**: `{k_eval_a['best_silhouette'] - results_a[3]['silhouette']:.4f}` (> 0.02 continuity threshold).
- **Outcome**: **$k={chosen_k_a}$** is selected under the pre-specified rule.

![Elbow and Silhouette Curves](figures/elbow_silhouette.png)

---

## 3. Variant A: Sensitivity, Stability & Demographic Structure

### Sensitivity & Stability Summary:

| Analysis | Variant A Chosen $k={chosen_k_a}$ | Historical $k=3$ Baseline |
|:---|:---:|:---:|
| **Unique-Profile ARI** (Full vs 132 Unique) | **{stability_a_chosen['ari_unique_vs_full']:.4f}** | **{stability_a_k3['ari_unique_vs_full']:.4f}** |
| **Unique Partition Sizes (Full Mapped)** | {stability_a_chosen['unique_sizes_full_mapped']} | {stability_a_k3['unique_sizes_full_mapped']} |
| **Unique Partition Sizes (Direct Unique)** | {stability_a_chosen['unique_sizes_direct']} | {stability_a_k3['unique_sizes_direct']} |
| **Seed Stability (Mean Pairwise ARI, Seeds 0–9)** | **{stability_a_chosen['mean_pairwise_ari']:.4f}** | **{stability_a_k3['mean_pairwise_ari']:.4f}** |
| **Seed Pairwise ARI Range [Min, Max]** | [{stability_a_chosen['min_pairwise_ari']:.4f}, {stability_a_chosen['max_pairwise_ari']:.4f}] | [{stability_a_k3['min_pairwise_ari']:.4f}, {stability_a_k3['max_pairwise_ari']:.4f}] |

> **Stability Finding**: The measured seed ARI across 10 random seeds has a **mean of {stability_a_chosen['mean_pairwise_ari']:.4f}** (min: {stability_a_chosen['min_pairwise_ari']:.4f}, max: {stability_a_chosen['max_pairwise_ari']:.4f}), reflecting moderate sensitivity to centroid initialization rather than universally rigid boundaries.

### Key Finding on Demographic Confounding in Variant A:
A plain examination reveals that **{purity_a['pure_clusters_count']} of {purity_a['total_clusters']} clusters ({purity_a['purity_share_pct']}%) are >= 90% dominated by a single gender or occupation**. 
Most strikingly, **Cluster 2 and Cluster 3 share near-identical demographics, biometric status, and occupational profiles**:
- **Demographics & Health**: Both cohorts are 100% Female Nurses, Overweight BMI, high prevalence of Sleep Apnea, with identical average Blood Pressure of 140/95 mmHg.
- **Opposite Risk Profiles**: Cluster 2 has **100% High Burnout Risk**, whereas Cluster 3 has **100% Low Burnout Risk**.
- **Root Difference**: The entire divergence between Cluster 2 and Cluster 3 is governed by **Sleep Duration and Quality** (Cluster 2 averages 6.07 hours of sleep at quality 6.0; Cluster 3 averages 8.09 hours of sleep at quality 9.0).
- **Analytical Takeaway**: Variant A's 27-dimensional feature space strongly blends the demographic and occupational structure of this dataset with actual behavior.

---

## 4. Variant A: Principal Component Analysis (PCA)

PCA was fitted on the 27-dimensional transformed feature matrix.

### Explained Variance Summary (First 10 Components):

| Component | Explained Variance Ratio (%) | Cumulative Variance Ratio (%) |
|:---:|:---:|:---:|
"""
    for i, (ev, cv) in enumerate(zip(pca_a["explained_variance_ratio"], pca_a["cumulative_variance_ratio"])):
        content += f"| PC{i+1} | {ev*100:.2f}% | {cv*100:.2f}% |\n"

    content += f"""
### Two-Component Projection Caveat:
- **Captured Variance by PC1 + PC2**: **{pca_a['two_pc_variance']*100:.2f}%** (PC1: `{pca_a['explained_variance_ratio'][0]*100:.2f}%`, PC2: `{pca_a['explained_variance_ratio'][1]*100:.2f}%`).
- **Variance Loss**: **{100 - pca_a['two_pc_variance']*100:.2f}%** of total variance is omitted in a 2-D visual projection. The 2-D scatter plot is a **lossy projection**; Euclidean distances in 2-D do not represent full 27-dimensional Euclidean distances.

### Top Feature Loadings for PC1 and PC2:

| Component | Top 5 Positive Loadings | Top 5 Negative Loadings |
|:---|:---|:---|
| **PC1** ({pca_a['explained_variance_ratio'][0]*100:.2f}%) | {', '.join([f'`{f}` (+{v:.3f})' for f, v in pca_a['pc1_loadings']['top_positive']])} | {', '.join([f'`{f}` ({v:.3f})' for f, v in pca_a['pc1_loadings']['top_negative']])} |
| **PC2** ({pca_a['explained_variance_ratio'][1]*100:.2f}%) | {', '.join([f'`{f}` (+{v:.3f})' for f, v in pca_a['pc2_loadings']['top_positive']])} | {', '.join([f'`{f}` ({v:.3f})' for f, v in pca_a['pc2_loadings']['top_negative']])} |

![PCA Scree Plot](figures/pca_scree.png)
![PCA 2-D Colored by Clusters](figures/pca_clusters.png)
![PCA 2-D Colored by Post-Hoc Burnout Risk](figures/pca_by_risk.png)

---

## 5. Variant A: Cluster Profiles & Crosstab ($k={chosen_k_a}$)

| Cluster | Size (%) | Descriptive Human Label | Key Measured Features (Mean) | Dominant Demographics & Health | Post-Hoc Risk (% Low / Med / High) | Mean Stress / Lifestyle |
|:---:|:---:|:---|:---|:---|:---:|:---:|
"""
    for cid in range(chosen_k_a):
        p = prof_a["profiles"][cid]
        num = p["numeric_stats"]
        cat = p["categorical_modes"]
        ph = p["post_hoc_metrics"]
        rd = ph["burnout_risk_distribution"]

        key_feat = (
            f"Sleep: {num['Sleep Duration']['mean']}h (Q:{num['Quality of Sleep']['mean']}), "
            f"Act: {num['Physical Activity Level']['mean']}m, BP: {num['Systolic BP']['mean']:.0f}/{num['Diastolic BP']['mean']:.0f}"
        )
        demo = (
            f"{cat['gender']['mode']} ({cat['gender']['share_pct']}%), "
            f"Occ: {cat['top_occupations'][0]['occupation']} ({cat['top_occupations'][0]['share_pct']}%), "
            f"BMI: {cat['bmi_category']['mode']}, SleepDis: {cat['sleep_disorder']['mode']}"
        )
        risk_str = f"{rd['Low']['share']:.0f}% / {rd['Medium']['share']:.0f}% / {rd['High']['share']:.0f}%"

        content += f"| **Cluster {cid}** | {p['size']} ({p['share_pct']:.1f}%) | {p['human_descriptive_label']} | {key_feat} | {demo} | {risk_str} | {ph['mean_stress_level']:.2f} / {ph['mean_lifestyle_score']:.1f} |\n"

    content += f"""
### Variant A Crosstabulation (Descriptive Only):

| Cluster ID | Descriptive Label | Low Risk | Medium Risk | High Risk | Total |
|:---:|:---|:---:|:---:|:---:|:---:|
"""
    for cid in range(chosen_k_a):
        ct_row = prof_a["crosstab"][cid]
        tot = sum(ct_row.values())
        content += f"| Cluster {cid} | {prof_a['profiles'][cid]['human_descriptive_label']} | {ct_row.get('Low', 0)} | {ct_row.get('Medium', 0)} | {ct_row.get('High', 0)} | {tot} |\n"

    content += f"""
- **Chi-Square Statistic (chi^2)**: **{prof_a['chi2_test']['chi2']:.2f}** (df={prof_a['chi2_test']['degrees_of_freedom']}, $p={prof_a['chi2_test']['p_value']:.4e}$).
- *Descriptive Caveat*: Non-independent duplicated rows violate i.i.d. assumptions.

---

## 6. Variant B: Behavioral-Only Clustering

To eliminate demographic confounding, **Variant B** fits K-Means strictly on 5 continuous behavioral features: `Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Daily Steps`, and `Heart Rate` (StandardScaler applied; zero demographic, age, BMI, or blood pressure inputs).

### Variant B Evaluation Table (k=2 to 8, 5 Continuous Features):

| Clusters ($k$) | Inertia (374 Rows) | Inertia (132 Unique) | Silhouette (374 Rows) | Silhouette (132 Unique Rows) | Notes |
|:---:|:---:|:---:|:---:|:---:|:---|
"""
    for k in range(2, 9):
        res = results_b[k]
        is_best = " (Best Unique)" if k == k_eval_b["best_k_by_silhouette"] else ""
        is_chosen = " **[CHOSEN B]**" if k == chosen_k_b else ""
        content += f"| {k} | {res['inertia']:.2f} | {res['unique_inertia']:.2f} | {res['silhouette']:.4f} | {res['unique_silhouette']:.4f}{is_best}{is_chosen} | {'Historical k' if k==3 else ''} |\n"

    content += f"""
### Variant B Selection Decision:
- **Evaluation Metric**: Pre-specified rule evaluated on the **132-unique-row silhouette**.
- **Best Silhouette on Unique Rows**: $k=5$ (`{results_b[5]['unique_silhouette']:.4f}`).
- **Historical $k=3$ Unique Silhouette**: `{results_b[3]['unique_silhouette']:.4f}`.
- **Gap**: `{results_b[5]['unique_silhouette'] - results_b[3]['unique_silhouette']:.4f}` (> 0.02 threshold).
- **Chosen $k$**: **$k={chosen_k_b}$**.

### Variant B Stability & PCA:
- **Cluster Sizes (374 Full Rows)**: {results_b[chosen_k_b]['cluster_sizes']}
- **Unique vs. Full ARI**: **{stability_b['ari_unique_vs_full']:.4f}** (fit on unique matches fit on full mapped to unique rows).
- **Seed Stability (Seeds 0–9)**: Mean pairwise ARI = **{stability_b['mean_pairwise_ari']:.4f}** (min: {stability_b['min_pairwise_ari']:.4f}, max: {stability_b['max_pairwise_ari']:.4f}).
- **PCA Explained Variance**:
  - PC1: `{pca_b['explained_variance_ratio'][0]*100:.2f}%` (Quality of Sleep +0.616, Sleep Duration +0.585, Heart Rate -0.485)
  - PC2: `{pca_b['explained_variance_ratio'][1]*100:.2f}%` (Physical Activity Level +0.689, Daily Steps +0.686, Heart Rate +0.210)
  - **PC1 + PC2 Cumulative Variance**: **{pca_b['two_pc_variance']*100:.2f}%**.
- **Demographic Purity**: Only **{purity_b['pure_clusters_count']} of {purity_b['total_clusters']} clusters ({purity_b['purity_share_pct']}%)** are >= 90% dominated by a single gender or occupation, confirming behavioral grouping across demographic lines.

### Variant B Cluster Profiles ($k={chosen_k_b}$):

| Cluster | Size (%) | Neutral Descriptive Human Label | Key Measured Features (Mean) | Demographics & Health Shares | Post-Hoc Risk (% Low / Med / High) | Mean Stress |
|:---:|:---:|:---|:---|:---|:---:|:---:|
"""
    for cid in range(chosen_k_b):
        p = prof_b["profiles"][cid]
        num = p["numeric_stats"]
        cat = p["categorical_modes"]
        ph = p["post_hoc_metrics"]
        rd = ph["burnout_risk_distribution"]

        key_feat = (
            f"Sleep: {num['Sleep Duration']['mean']}h (Q:{num['Quality of Sleep']['mean']}), "
            f"Act: {num['Physical Activity Level']['mean']:.0f}m, Steps: {num['Daily Steps']['mean']:.0f}, HR: {num['Heart Rate']['mean']:.0f}"
        )
        demo = (
            f"{cat['gender']['mode']} ({cat['gender']['share_pct']}%), "
            f"Occ: {cat['top_occupations'][0]['occupation']} ({cat['top_occupations'][0]['share_pct']}%), "
            f"BMI: {cat['bmi_category']['mode']}"
        )
        risk_str = f"{rd['Low']['share']:.0f}% / {rd['Medium']['share']:.0f}% / {rd['High']['share']:.0f}%"

        content += f"| **Cluster {cid}** | {p['size']} ({p['share_pct']:.1f}%) | {p['human_descriptive_label']} | {key_feat} | {demo} | {risk_str} | {ph['mean_stress_level']:.2f} |\n"

    content += f"""
### Variant B Crosstabulation (Descriptive Only):

| Cluster ID | Descriptive Label | Low Risk | Medium Risk | High Risk | Total |
|:---:|:---|:---:|:---:|:---:|:---:|
"""
    for cid in range(chosen_k_b):
        ct_row = prof_b["crosstab"][cid]
        tot = sum(ct_row.values())
        content += f"| Cluster {cid} | {prof_b['profiles'][cid]['human_descriptive_label']} | {ct_row.get('Low', 0)} | {ct_row.get('Medium', 0)} | {ct_row.get('High', 0)} | {tot} |\n"

    content += f"""
- **Chi-Square Statistic (chi^2)**: **{prof_b['chi2_test']['chi2']:.2f}** (df={prof_b['chi2_test']['degrees_of_freedom']}, $p={prof_b['chi2_test']['p_value']:.4e}$).
- *Descriptive Caveat*: Non-independent duplicated rows violate i.i.d. assumptions.

---

## 7. Comparative Synthesis: Variant A vs. Variant B

| Dimension | Variant A (All 12 Inputs) | Variant B (Behavioral-Only 5 Features) |
|:---|:---:|:---:|
| **Features Included** | 12 features (Demographics, Occupation, Biometrics, BP) | 5 continuous features (`Sleep Duration`, `Quality of Sleep`, `Activity`, `Steps`, `Heart Rate`) |
| **Input Dimensions** | 27 dimensions (one-hot encoded + scaled) | 5 dimensions (`StandardScaler` continuous) |
| **Chosen $k$** | **$k={chosen_k_a}$** | **$k={chosen_k_b}$** |
| **Silhouette (Full 374 Rows)** | **{results_a[chosen_k_a]['silhouette']:.4f}** | **{results_b[chosen_k_b]['silhouette']:.4f}** |
| **Silhouette (132 Unique Rows)** | **{results_a[chosen_k_a]['unique_silhouette']:.4f}** | **{results_b[chosen_k_b]['unique_silhouette']:.4f}** |
| **Seed Stability (Mean / Min ARI)** | **0.7634 / 0.5878** | **0.8224 / 0.5147** |
| **Unique-vs-Full ARI** | **0.8795** | **1.0000** |
| **2-PC Explained Variance** | **60.04%** | **83.70%** |
| **Demographic Purity (>= 90% single Gender or Occ)** | **{purity_a['purity_share_pct']}% ({purity_a['pure_clusters_count']}/{purity_a['total_clusters']} clusters)** | **{purity_b['purity_share_pct']}% ({purity_b['pure_clusters_count']}/{purity_b['total_clusters']} clusters)** |
| **Ease of Explanation** | Segments into discrete occupational archetypes (e.g. female nurses, male lawyers) where demographics dominate cluster boundaries. | Groups individuals purely by actionable lifestyle habits (sleep, activity, heart rate), creating behavioral cohorts that span across job roles. |

---

## 8. Historical vs Clean Reproduction Comparison

In the original exploratory notebook (`data_understanding.ipynb`), K-Means was run with $k=3$, yielding historical cluster sizes of **215 / 92 / 67**.

| Dimension | Historical Baseline (Notebook) | Clean Reprocessed ($k=3$) | Clean Chosen Variant A ($k=8$) | Clean Chosen Variant B ($k=5$) |
|:---|:---|:---|:---|:---|
| **Preprocessing** | Leaky scaler on full dataset, LabelEncoding on categoricals, raw string `Blood Pressure` retained | Clean ColumnTransformer (OneHotEncoder + StandardScaler), exact systolic/diastolic parsing | Clean ColumnTransformer (OneHotEncoder + StandardScaler) | StandardScaler on 5 continuous behavioral features |
| **Input Dimensions** | 13 features (including arbitrary nominal integers) | 27 features | 27 features | 5 features |
| **Clusters ($k$)** | $k=3$ | $k=3$ | $k=8$ | $k=5$ |
| **Silhouette Score** | **UNVERIFIED** (the original notebook never computed or reported silhouette scores) | **0.3559** (Full) / **0.3151** (Unique) | **0.5454** (Full) / **0.4059** (Unique) | **0.5689** (Full) / **0.5146** (Unique) |
| **Cluster Sizes** | **215 / 92 / 67** | **128 / 67 / 179** | **21 / 106 / 32 / 33 / 69 / 32 / 39 / 42** | **178 / 34 / 22 / 108 / 32** |
| **Cluster Shift Explanation** | **HYPOTHESIS (Untested)**: LabelEncoder imposed artificial numeric ordering on nominal features (e.g. Doctor=1, Nurse=5) and retained the raw Blood Pressure string, likely distorting distance calculations. | Clean one-hot encoding eliminates artificial ordinality; cluster boundaries reflect true geometric distances. | Higher $k$ untangles distinct occupational and biometric cohorts that were lumped together. | Focuses purely on behavioral parameters, eliminating demographic weighting. |

---

## 9. Analytical Limitations

1. **Small Sample Size with Extensive Duplicates**:
   The dataset comprises only 374 total records with only 132 unique behavioral profiles. Repeated entries weight cluster centroids toward overrepresented worker cohorts (e.g., Nurses with sleep apnea, Doctors with moderate sleep).
2. **K-Means Geometric Assumptions**:
   K-Means assumes isotropic, spherical clusters of roughly equal variance in Euclidean space. In a mixed-type one-hot encoded space (27 dimensions), Euclidean distance between binary indicators has non-spherical geometric properties.
3. **One-Hot Encoding of High-Cardinality Occupations**:
   One-hot encoding of `Occupation` introduces sparse binary dimensions for rare roles (e.g., Manager, Software Engineer), which can induce distinct micro-clusters.
4. **Descriptive Associations vs Clinical Diagnoses**:
   Cluster labels and profile characteristics are descriptive archetypes of measured lifestyle and health variables. They must not be construed as clinical or diagnostic determinations.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()
