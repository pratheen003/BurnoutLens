"""BurnoutLens Prediction & Analytics Service.

Provides:
- Robust input validation against feature_schema.json with snake_case keys
- Pipeline prediction and uncalibrated model probabilities
- Exact SHAP feature contributions on the decision_function scale via mean-vector formula
- Variant B behavioral clustering assignment & PCA coordinates
- Static rule-based wellness recommendations (not derived from model)
- Mandatory educational disclaimer on all outputs
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn

from burnoutlens.analytics import BEHAVIORAL_FEATURES
from burnoutlens.config import INPUT_FEATURES
from burnoutlens.explain import CLASS_NAMES, map_transformed_to_original


FORBIDDEN_COLUMNS = {
    "Stress Level",
    "stress_level",
    "Burnout Risk",
    "burnout_risk",
    "Burnout Index",
    "burnout_index",
    "Lifestyle Score",
    "lifestyle_score",
    "Person ID",
    "person_id",
}

SNAKE_TO_CANONICAL = {
    "gender": "Gender",
    "age": "Age",
    "occupation": "Occupation",
    "sleep_duration": "Sleep Duration",
    "quality_of_sleep": "Quality of Sleep",
    "physical_activity_level": "Physical Activity Level",
    "bmi_category": "BMI Category",
    "heart_rate": "Heart Rate",
    "daily_steps": "Daily Steps",
    "sleep_disorder": "Sleep Disorder",
    "systolic_bp": "Systolic BP",
    "diastolic_bp": "Diastolic BP",
}

CANONICAL_TO_SNAKE = {v: k for k, v in SNAKE_TO_CANONICAL.items()}

LIFESTYLE_FEATURES = {
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Daily Steps",
    "Heart Rate",
}

HEALTH_INDICATORS = {
    "BMI Category",
    "Sleep Disorder",
    "Systolic BP",
    "Diastolic BP",
}

CONTEXT_FEATURES = {
    "Gender",
    "Age",
    "Occupation",
}

DISCLAIMER_TEXT = "Lifestyle-based burnout risk estimate for educational purposes. Not a medical diagnosis."


class ServiceValidationError(ValueError):
    """Raised when input validation fails."""
    pass


class BurnoutService:
    """Prediction and analytics service for BurnoutLens."""

    def __init__(
        self,
        models_dir: Optional[Union[str, Path]] = None,
        reports_dir: Optional[Union[str, Path]] = None,
    ):
        pkg_dir = Path(__file__).resolve().parent
        root_dir = pkg_dir.parent.parent

        self.models_dir = Path(models_dir) if models_dir else root_dir / "models"
        self.reports_dir = Path(reports_dir) if reports_dir else root_dir / "reports"

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all model artifacts and schemas relative to package directory."""
        manifest_path = self.models_dir / "model_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Model manifest not found at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        manifest_sklearn = (
            self.manifest.get("runtime_environment", {}).get("scikit_learn_version")
        )
        if manifest_sklearn and manifest_sklearn != sklearn.__version__:
            warnings.warn(
                f"Current scikit-learn version ({sklearn.__version__}) differs from "
                f"manifest serialized version ({manifest_sklearn}).",
                UserWarning,
                stacklevel=2,
            )

        pipeline_path = self.models_dir / "burnout_pipeline.joblib"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Burnout pipeline not found at {pipeline_path}")
        self.pipeline = joblib.load(pipeline_path)

        bg_path = self.models_dir / "explain_background.json"
        if not bg_path.exists():
            raise FileNotFoundError(f"Background mean not found at {bg_path}")
        with open(bg_path, "r", encoding="utf-8") as f:
            self.bg_data = json.load(f)
        self.bg_mean = np.array(self.bg_data["background_mean"], dtype=float)
        self.transformed_feature_names = self.bg_data["transformed_feature_names"]

        clusters_path = self.models_dir / "behavior_clusters.joblib"
        if not clusters_path.exists():
            raise FileNotFoundError(f"Behavioral clusters not found at {clusters_path}")
        clusters_data = joblib.load(clusters_path)
        self.behavior_scaler = clusters_data["scaler"]
        self.behavior_kmeans = clusters_data["kmeans"]
        self.behavior_pca = clusters_data["pca"]

        pca_points_path = self.models_dir / "pca_points_variant_b.json"
        if not pca_points_path.exists():
            raise FileNotFoundError(f"PCA points not found at {pca_points_path}")
        with open(pca_points_path, "r", encoding="utf-8") as f:
            self.pca_points_variant_b = json.load(f)

        schema_path = self.reports_dir / "feature_schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found at {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
        self.feature_schema = schema_data["features"]

        profiles_path = self.reports_dir / "cluster_profiles.json"
        if not profiles_path.exists():
            raise FileNotFoundError(f"Cluster profiles not found at {profiles_path}")
        with open(profiles_path, "r", encoding="utf-8") as f:
            profiles_data = json.load(f)
        self.cluster_profiles_variant_b = profiles_data["variant_b_behavioral_only"]["clusters"]

        self.orig_map = map_transformed_to_original(self.transformed_feature_names)

    def validate_and_standardize_input(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input dictionary and convert snake_case keys to canonical dataset columns.

        Parameters
        ----------
        raw_input : Dict[str, Any]
            User input dictionary (with snake_case or canonical keys).

        Returns
        -------
        Dict[str, Any]
            Validated dictionary keyed by canonical INPUT_FEATURES.

        Raises
        ------
        ServiceValidationError
            If any forbidden, missing, unseen, or out-of-range values are detected.
        """
        # 1. Check for forbidden columns
        for key in raw_input.keys():
            if key in FORBIDDEN_COLUMNS:
                raise ServiceValidationError(f"Forbidden column '{key}' detected in request input.")

        # 2. Map snake_case to canonical
        canonical_input: Dict[str, Any] = {}
        for key, val in raw_input.items():
            if key in SNAKE_TO_CANONICAL:
                canonical_input[SNAKE_TO_CANONICAL[key]] = val
            elif key in INPUT_FEATURES:
                canonical_input[key] = val
            else:
                raise ServiceValidationError(f"Unrecognized input field '{key}'.")

        # 3. Check for missing required features
        for feat in INPUT_FEATURES:
            if feat not in canonical_input:
                snake_name = CANONICAL_TO_SNAKE.get(feat, feat)
                raise ServiceValidationError(f"Missing required field: '{snake_name}'.")

        # 4. Validate types and ranges against schema
        clean_row: Dict[str, Any] = {}
        for feat in INPUT_FEATURES:
            val = canonical_input[feat]
            spec = self.feature_schema[feat]
            field_name_for_err = CANONICAL_TO_SNAKE.get(feat, feat)

            if spec["type"] == "numeric":
                try:
                    num_val = float(val)
                except (ValueError, TypeError):
                    raise ServiceValidationError(
                        f"Field '{field_name_for_err}' must be a number, got '{val}'."
                    )
                if num_val < spec["min"] or num_val > spec["max"]:
                    raise ServiceValidationError(
                        f"Field '{field_name_for_err}' value {num_val} is out of allowed range "
                        f"[{spec['min']}, {spec['max']}]."
                    )
                clean_row[feat] = num_val
            elif spec["type"] == "categorical":
                str_val = str(val).strip()
                if str_val not in spec["allowed_categories"]:
                    raise ServiceValidationError(
                        f"Field '{field_name_for_err}' has unseen category '{str_val}'. "
                        f"Allowed categories: {spec['allowed_categories']}."
                    )
                clean_row[feat] = str_val

        return clean_row

    def predict(self, clean_row: Dict[str, Any]) -> Dict[str, Any]:
        """Compute model prediction and uncalibrated probabilities.

        Parameters
        ----------
        clean_row : Dict[str, Any]
            Validated feature dictionary.

        Returns
        -------
        Dict[str, Any]
            Predicted class, model probabilities, and probability note.
        """
        df_row = pd.DataFrame([clean_row])[INPUT_FEATURES]
        probs = self.pipeline.predict_proba(df_row)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]

        model_probabilities = {
            CLASS_NAMES[i]: float(round(probs[i], 4)) for i in range(len(CLASS_NAMES))
        }

        return {
            "predicted_class": pred_class,
            "model_probability": model_probabilities,
            "probability_note": (
                "Model probabilities are uncalibrated estimates derived from a logistic regression "
                "model trained on a small dataset (N=374). They represent relative model scores rather "
                "than true clinical risk probabilities."
            ),
        }

    def contributions(
        self,
        clean_row: Dict[str, Any],
        predicted_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute SHAP feature contributions for the predicted class on the decision_function scale.

        Formula: coef * (x_transformed - background_mean), aggregated to original 12 features.

        Parameters
        ----------
        clean_row : Dict[str, Any]
            Validated feature dictionary.
        predicted_class : Optional[str]
            Target class to explain. If None, uses the model's predicted class.

        Returns
        -------
        Dict[str, Any]
            Base value, decision value, and list of per-feature contributions sorted by |contribution|.
        """
        if predicted_class is None:
            pred_res = self.predict(clean_row)
            predicted_class = pred_res["predicted_class"]

        c_idx = CLASS_NAMES.index(predicted_class)
        df_row = pd.DataFrame([clean_row])[INPUT_FEATURES]
        x_trans = self.pipeline.named_steps["preprocessor"].transform(df_row)[0]

        lr = self.pipeline.named_steps["classifier"]
        w = lr.coef_[c_idx]
        b = lr.intercept_[c_idx]

        diff = x_trans - self.bg_mean
        shap_trans = w * diff

        base_val = float(b + np.dot(w, self.bg_mean))

        # Aggregate 27 transformed features to 12 original features
        feat_contribs: Dict[str, float] = {f: 0.0 for f in INPUT_FEATURES}
        for t_idx, orig_f in enumerate(self.orig_map):
            feat_contribs[orig_f] += float(shap_trans[t_idx])

        feature_items: List[Dict[str, Any]] = []
        for feat in INPUT_FEATURES:
            signed_c = float(round(feat_contribs[feat], 4))
            user_val = clean_row[feat]
            direction = (
                f"pushes toward {predicted_class}"
                if signed_c > 0
                else f"pushes away from {predicted_class}"
            )

            if feat in LIFESTYLE_FEATURES:
                group = "lifestyle"
            elif feat in HEALTH_INDICATORS:
                group = "health_indicator"
            else:
                group = "context"

            feature_items.append({
                "feature": feat,
                "user_value": user_val,
                "signed_contribution": signed_c,
                "direction": direction,
                "group": group,
            })

        feature_items.sort(key=lambda x: abs(x["signed_contribution"]), reverse=True)

        return {
            "predicted_class": predicted_class,
            "base_value": float(round(base_val, 4)),
            "features": feature_items,
        }

    def cluster(self, clean_row: Dict[str, Any]) -> Dict[str, Any]:
        """Compute Variant B behavioral cluster assignment and PCA coordinates.

        Parameters
        ----------
        clean_row : Dict[str, Any]
            Validated feature dictionary.

        Returns
        -------
        Dict[str, Any]
            Cluster id, profile label, profile summary, and user PCA coordinates.
        """
        df_behav = pd.DataFrame([{f: clean_row[f] for f in BEHAVIORAL_FEATURES}])[BEHAVIORAL_FEATURES]
        behav_scaled = self.behavior_scaler.transform(df_behav)

        cluster_id = int(self.behavior_kmeans.predict(behav_scaled)[0])
        pca_coords = self.behavior_pca.transform(behav_scaled)[0]

        profile_info = self.cluster_profiles_variant_b.get(str(cluster_id), {})

        return {
            "cluster_id": cluster_id,
            "profile_label": profile_info.get("human_descriptive_label", f"Cluster {cluster_id}"),
            "profile_summary": profile_info.get("key_profile_values", {}),
            "user_pca_coordinates": {
                "pc1": float(round(pca_coords[0], 6)),
                "pc2": float(round(pca_coords[1], 6)),
            },
        }

    def recommendations(self, clean_row: Dict[str, Any]) -> Dict[str, Any]:
        """Generate static rule-based general wellness information.

        NOT derived from model weights or SHAP directions.
        Rules:
        - sleep_duration < 7 -> general guidance toward consistent 7-9 hours.
        - quality_of_sleep <= 6 -> general sleep-hygiene tips.
        - ALWAYS add recommendation to seek professional support if overwhelmed.
        NO rules for steps, physical activity, heart rate, BMI, or BP.

        Parameters
        ----------
        clean_row : Dict[str, Any]
            Validated feature dictionary.

        Returns
        -------
        Dict[str, Any]
            List of static wellness suggestions and explicit non-medical disclaimer.
        """
        wellness_text: List[str] = []

        if clean_row["Sleep Duration"] < 7.0:
            wellness_text.append(
                "Aim for a consistent sleep schedule targeting approximately 7-9 hours of restful sleep per night."
            )

        if clean_row["Quality of Sleep"] <= 6:
            wellness_text.append(
                "General sleep hygiene tips: Maintain a cool, quiet, and dark sleep environment, "
                "limit screen exposure in the hour before bedtime, and avoid caffeine in the late afternoon/evening."
            )

        # Always included
        wellness_text.append(
            "If you feel persistently overwhelmed, consider talking to a trusted person or a qualified professional."
        )

        return {
            "field_category": "general wellness information, not medical advice",
            "items": wellness_text,
        }

    def process_request(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Full end-to-end processing pipeline for a prediction request.

        Parameters
        ----------
        raw_input : Dict[str, Any]
            Raw request dictionary.

        Returns
        -------
        Dict[str, Any]
            Consolidated prediction, explanations, cluster assignment, recommendations, and disclaimer.
        """
        clean_row = self.validate_and_standardize_input(raw_input)

        prediction_res = self.predict(clean_row)
        pred_class = prediction_res["predicted_class"]

        contributions_res = self.contributions(clean_row, predicted_class=pred_class)
        cluster_res = self.cluster(clean_row)
        recommendations_res = self.recommendations(clean_row)

        return {
            "prediction": prediction_res,
            "contributions": contributions_res,
            "cluster": cluster_res,
            "recommendations": recommendations_res,
            "disclaimer": DISCLAIMER_TEXT,
        }


# Global singleton instance for clean service reuse
_service_instance: Optional[BurnoutService] = None


def get_service() -> BurnoutService:
    """Retrieve or initialize singleton BurnoutService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = BurnoutService()
    return _service_instance
