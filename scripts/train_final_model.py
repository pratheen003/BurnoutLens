"""Train and serialize final BurnoutLens models and artifacts.

Writes:
- models/burnout_pipeline.joblib
- models/explain_background.json
- models/behavior_clusters.joblib
- models/pca_points_variant_b.json
- models/model_manifest.json
"""

import sys
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
import shap

from burnoutlens.analytics import BEHAVIORAL_FEATURES
from burnoutlens.config import INPUT_FEATURES
from burnoutlens.data import load_raw
from burnoutlens.features import clean_data, make_target
from burnoutlens.leakage import assert_no_leakage
from burnoutlens.preprocessing import build_preprocessor


def compute_file_sha256(filepath: Path) -> str:
    """Compute sha256 hex digest of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    root_dir = Path(__file__).resolve().parent.parent
    models_dir = root_dir / "models"
    reports_dir = root_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading raw and cleaned dataset...")
    raw_csv_path = root_dir / "Sleep_health_and_lifestyle_dataset.csv"
    raw_sha256 = compute_file_sha256(raw_csv_path)

    df = clean_data(load_raw())
    n_rows = len(df)
    assert n_rows == 374, f"Expected 374 rows, got {n_rows}"
    assert_no_leakage(df[INPUT_FEATURES])

    class_mapping = {"Low": 0, "Medium": 1, "High": 2}
    y = make_target(df).map(class_mapping).values

    # Step 1: Fit and save the selected Phase 3 pipeline
    print("2. Fitting Phase 3 selected Logistic Regression pipeline on all 374 rows...")
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(df[INPUT_FEATURES], y)

    pipeline_path = models_dir / "burnout_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"Saved pipeline to {pipeline_path}")

    # Step 2: Background mean over 132 unique rows
    print("3. Computing background mean over 132 unique profile rows...")
    df_unique = df.drop_duplicates(subset=INPUT_FEATURES).copy()
    assert len(df_unique) == 132, f"Expected 132 unique rows, got {len(df_unique)}"

    preprocessor = pipeline.named_steps["preprocessor"]
    X_unique_trans = preprocessor.transform(df_unique[INPUT_FEATURES])
    assert X_unique_trans.shape == (132, 27), f"Expected (132, 27), got {X_unique_trans.shape}"

    bg_mean = np.mean(X_unique_trans, axis=0).astype(float).tolist()
    feature_names = list(preprocessor.get_feature_names_out())
    assert len(bg_mean) == 27, f"Expected 27 mean floats, got {len(bg_mean)}"

    explain_bg_data = {
        "n_unique_rows": 132,
        "n_features_transformed": len(feature_names),
        "transformed_feature_names": feature_names,
        "background_mean": bg_mean,
    }
    bg_path = models_dir / "explain_background.json"
    with open(bg_path, "w", encoding="utf-8") as f:
        json.dump(explain_bg_data, f, indent=2)
    print(f"Saved background mean to {bg_path}")

    # Step 3: Variant B Behavioral Clustering
    print("4. Fitting Variant B behavioral clustering (k=5, 5 features)...")
    X_behav = df[BEHAVIORAL_FEATURES].copy()
    assert_no_leakage(X_behav)

    scaler_b = StandardScaler()
    X_behav_scaled = scaler_b.fit_transform(X_behav)

    kmeans_b = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans_b.fit_predict(X_behav_scaled)
    cluster_sizes = [int(s) for s in np.bincount(cluster_labels, minlength=5)]
    print(f"Observed cluster sizes: {cluster_sizes}")

    expected_sizes = [178, 34, 22, 108, 32]
    if cluster_sizes != expected_sizes:
        raise ValueError(
            f"Cluster size assertion failed: observed {cluster_sizes} != expected {expected_sizes}. "
            "Stopping execution per protocol - do not relabel silently."
        )

    # Cross-check cluster IDs and sizes against reports/cluster_profiles.json
    profiles_json_path = reports_dir / "cluster_profiles.json"
    with open(profiles_json_path, "r", encoding="utf-8") as f:
        profiles_data = json.load(f)
    variant_b_clusters = profiles_data["variant_b_behavioral_only"]["clusters"]
    for cid in range(5):
        prof_size = variant_b_clusters[str(cid)]["size"]
        if prof_size != cluster_sizes[cid]:
            raise ValueError(
                f"Cluster {cid} size mismatch with reports/cluster_profiles.json: "
                f"{cluster_sizes[cid]} != {prof_size}."
            )

    pca_b = PCA(n_components=2, random_state=42)
    coords_all = pca_b.fit_transform(X_behav_scaled)

    behavior_clusters_artifact = {
        "features": list(BEHAVIORAL_FEATURES),
        "scaler": scaler_b,
        "kmeans": kmeans_b,
        "pca": pca_b,
    }
    clusters_path = models_dir / "behavior_clusters.joblib"
    joblib.dump(behavior_clusters_artifact, clusters_path)
    print(f"Saved behavioral clusters artifact to {clusters_path}")

    # Step 4: Save PCA 2D Points for 132 Unique Rows
    print("5. Generating models/pca_points_variant_b.json for 132 unique rows...")
    risk_series = make_target(df)
    unique_indices = df_unique.index
    pca_points = []
    for idx in unique_indices:
        pca_points.append({
            "pc1": float(round(coords_all[idx, 0], 6)),
            "pc2": float(round(coords_all[idx, 1], 6)),
            "cluster_id": int(cluster_labels[idx]),
            "burnout_risk": str(risk_series.loc[idx]),
        })
    assert len(pca_points) == 132, f"Expected 132 points, got {len(pca_points)}"

    pca_points_path = models_dir / "pca_points_variant_b.json"
    with open(pca_points_path, "w", encoding="utf-8") as f:
        json.dump(pca_points, f, indent=2)
    print(f"Saved PCA points to {pca_points_path}")

    # Step 5: Read evaluation numbers from reports/supervised_results.csv
    print("6. Reading evaluation metrics from reports/supervised_results.csv...")
    eval_csv_path = reports_dir / "supervised_results.csv"
    eval_df = pd.read_csv(eval_csv_path)

    # Filter for Logistic Regression Protocol C (repeated) and Protocol D (repeated)
    lr_c = eval_df[
        (eval_df["model"] == "Logistic Regression")
        & (eval_df["protocol"] == "Protocol C (Grouped 5-Fold CV 5-Seeds Repeated)")
    ].iloc[0]

    lr_d = eval_df[
        (eval_df["model"] == "Logistic Regression")
        & (eval_df["protocol"] == "Protocol D (Random Stratified 5-Fold CV 5-Seeds Repeated)")
    ].iloc[0]

    eval_summary = {
        "protocol_c_grouped_cv": {
            "protocol_name": str(lr_c["protocol"]),
            "accuracy": float(lr_c["accuracy"]),
            "std_accuracy": float(lr_c["std_accuracy"]),
            "macro_f1": float(lr_c["macro_f1"]),
            "std_macro_f1": float(lr_c["std_macro_f1"]),
        },
        "protocol_d_random_cv": {
            "protocol_name": str(lr_d["protocol"]),
            "accuracy": float(lr_d["accuracy"]),
            "std_accuracy": float(lr_d["std_accuracy"]),
            "macro_f1": float(lr_d["macro_f1"]),
            "std_macro_f1": float(lr_d["std_macro_f1"]),
        },
        "evaluation_note": (
            "There is no held-out score for the all-data model; reported numbers come from cross-validation."
        ),
    }

    # Step 6: Create Manifest
    print("7. Generating models/model_manifest.json...")
    artifacts_to_hash = [
        pipeline_path,
        bg_path,
        clusters_path,
        pca_points_path,
    ]
    artifact_entries = []
    for art in artifacts_to_hash:
        artifact_entries.append({
            "filename": art.name,
            "size_bytes": art.stat().st_size,
            "sha256": compute_file_sha256(art),
        })

    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_environment": {
            "python_version": sys.version,
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "shap_version": shap.__version__,
        },
        "dataset": {
            "raw_csv_filename": raw_csv_path.name,
            "raw_csv_sha256": raw_sha256,
            "n_training_rows": n_rows,
            "n_unique_profile_rows": len(df_unique),
        },
        "selected_model": {
            "name": "Logistic Regression (StandardScaler + OneHotEncoder)",
            "pipeline_specification": "build_preprocessor() + LogisticRegression(max_iter=1000, random_state=42)",
            "selection_reason": (
                "Selected in Phase 3 for highest generalization Macro-F1 (0.9499) under 5-seed repeated "
                "grouped 5-fold cross-validation without leakage, high explainability, and minimal parameter "
                "complexity compared to tree ensembles."
            ),
        },
        "feature_order": INPUT_FEATURES,
        "behavioral_feature_order": list(BEHAVIORAL_FEATURES),
        "class_mapping": class_mapping,
        "cross_validation_metrics": eval_summary,
        "disclaimer": (
            "Lifestyle-based burnout risk estimate for educational purposes. Not a medical diagnosis. "
            "There is no held-out score for the all-data model; reported numbers come from cross-validation."
        ),
        "artifacts": artifact_entries,
    }

    manifest_path = models_dir / "model_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved model manifest to {manifest_path}")
    print("Phase 6 model serialization completed successfully.")


if __name__ == "__main__":
    main()
