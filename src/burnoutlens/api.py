"""FastAPI application for BurnoutLens prediction and analytics service.

Run with:
    uvicorn burnoutlens.api:app --app-dir src --reload
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.staticfiles import StaticFiles

from burnoutlens.service import (
    DISCLAIMER_TEXT,
    BurnoutService,
    ServiceValidationError,
    get_service,
)

app = FastAPI(
    title="BurnoutLens API",
    description="Explainable burnout risk assessment and behavioral analytics service.",
    version="1.0.0",
)

# CORS configuration for localhost origins only
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ServiceValidationError)
async def service_validation_error_handler(request: Request, exc: ServiceValidationError):
    """Translate internal service validation error to HTTP 422 Unprocessable Content."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(exc)},
    )


class PredictRequest(BaseModel):
    """Schema for prediction request with snake_case keys."""

    gender: str = Field(..., description="Biological sex (Female, Male)")
    age: float = Field(..., description="Age in years (27 to 59)")
    occupation: str = Field(..., description="Job occupation title")
    sleep_duration: float = Field(..., description="Average sleep duration in hours (5.8 to 8.5)")
    quality_of_sleep: float = Field(..., description="Self-reported sleep quality (4 to 9)")
    physical_activity_level: float = Field(..., description="Daily physical activity in minutes (30 to 90)")
    bmi_category: str = Field(..., description="BMI Category (Normal, Overweight, Obese)")
    heart_rate: float = Field(..., description="Resting heart rate in bpm (65 to 86)")
    daily_steps: float = Field(..., description="Average daily step count (3000 to 10000)")
    sleep_disorder: str = Field(..., description="Diagnosed sleep disorder (None, Insomnia, Sleep Apnea)")
    systolic_bp: float = Field(..., description="Systolic blood pressure in mmHg (115 to 142)")
    diastolic_bp: float = Field(..., description="Diastolic blood pressure in mmHg (75 to 95)")

    model_config = ConfigDict(extra="forbid")


@app.get("/health", summary="Health Check")
def health_check() -> Dict[str, str]:
    """Return service health status."""
    return {"status": "ok", "service": "BurnoutLens API"}


@app.get("/meta", summary="Model Metadata & Limitations")
def get_metadata() -> Dict[str, Any]:
    """Return schema ranges, manifest summary without internal file paths, and limitations."""
    service = get_service()
    manifest = service.manifest

    safe_artifacts = [
        {"filename": a["filename"], "sha256": a["sha256"], "size_bytes": a["size_bytes"]}
        for a in manifest.get("artifacts", [])
    ]

    manifest_summary = {
        "manifest_version": manifest.get("manifest_version"),
        "created_at_utc": manifest.get("created_at_utc"),
        "runtime_environment": manifest.get("runtime_environment"),
        "dataset_summary": {
            "n_training_rows": manifest.get("dataset", {}).get("n_training_rows"),
            "n_unique_profile_rows": manifest.get("dataset", {}).get("n_unique_profile_rows"),
            "raw_csv_sha256": manifest.get("dataset", {}).get("raw_csv_sha256"),
        },
        "selected_model": manifest.get("selected_model"),
        "feature_order": manifest.get("feature_order"),
        "behavioral_feature_order": manifest.get("behavioral_feature_order"),
        "class_mapping": manifest.get("class_mapping"),
        "cross_validation_metrics": manifest.get("cross_validation_metrics"),
        "artifacts": safe_artifacts,
    }

    limitations = [
        "Lifestyle-based burnout risk estimate for educational and exploratory purposes only. Not a medical diagnosis.",
        "Ground truth is derived deterministically from self-reported Stress Level (Low: 1-4, Medium: 5-6, High: 7-10).",
        "Observational dataset artifact: Higher Daily Steps and Physical Activity correlate with higher risk in this sample because 94% of the 10,000-step cohort are nurses with elevated stress.",
        "Demographic features (Gender, Age, Occupation) are non-actionable context variables rather than intervention targets.",
        "Model probabilities are uncalibrated estimates derived from a logistic regression model trained on a small dataset (N=374).",
        "No held-out test set exists for the final model trained on all data; performance metrics reflect 5-fold cross-validation.",
    ]

    return {
        "feature_schema": service.feature_schema,
        "manifest_summary": manifest_summary,
        "disclaimer": DISCLAIMER_TEXT,
        "limitations": limitations,
    }


@app.post("/predict", summary="Predict Burnout Risk & Explain")
def predict_burnout(request: PredictRequest) -> Dict[str, Any]:
    """Generate risk prediction, probability distribution, SHAP contributions, cluster, and recommendations."""
    service = get_service()
    raw_dict = request.model_dump()
    return service.process_request(raw_dict)


@app.get("/analytics/clusters", summary="Behavioral Clustering Analysis")
def get_clusters() -> Dict[str, Any]:
    """Return Variant B behavioral cluster profiles and 2D PCA projection coordinates for 132 unique profiles."""
    service = get_service()
    return {
        "variant_b_profiles": service.cluster_profiles_variant_b,
        "pca_points_variant_b": service.pca_points_variant_b,
        "disclaimer": DISCLAIMER_TEXT,
    }


@app.get("/analytics/importance", summary="Global Feature Importance")
def get_importance() -> Dict[str, Any]:
    """Return global SHAP importance, permutation importance, and cross-model validation results."""
    service = get_service()
    importance_path = service.reports_dir / "feature_importance.json"
    if not importance_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature importance report not found at {importance_path}",
        )
    with open(importance_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/models/comparison", summary="Supervised Model Evaluation Comparison")
def get_model_comparison() -> Dict[str, Any]:
    """Return cross-validation comparison across supervised model architectures under Protocol C and Protocol D."""
    service = get_service()
    results_csv_path = service.reports_dir / "supervised_results.csv"
    if not results_csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supervised results not found at {results_csv_path}",
        )

    df_results = pd.read_csv(results_csv_path)
    cd_mask = df_results["protocol"].str.contains("Protocol C|Protocol D", na=False)
    filtered_df = df_results[cd_mask]

    models_data: Dict[str, List[Dict[str, Any]]] = {}
    for model_name, grp in filtered_df.groupby("model", sort=False):
        entries = []
        for _, row in grp.iterrows():
            entries.append({
                "protocol": str(row["protocol"]),
                "accuracy": float(row["accuracy"]) if pd.notna(row["accuracy"]) else None,
                "std_accuracy": float(row["std_accuracy"]) if pd.notna(row["std_accuracy"]) else None,
                "macro_f1": float(row["macro_f1"]) if pd.notna(row["macro_f1"]) else None,
                "std_macro_f1": float(row["std_macro_f1"]) if pd.notna(row["std_macro_f1"]) else None,
            })
        models_data[model_name] = entries

    return {
        "models": models_data,
        "note": (
            "Protocol A is a single split and not comparable. Protocol C isolates duplicate profiles across folds "
            "(clean generalization); Protocol D allows identical profiles across folds (overestimating generalization by 1-3 points)."
        ),
        "disclaimer": DISCLAIMER_TEXT,
    }


# Static frontend mounting if directory exists (do not create now)
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists() and frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
