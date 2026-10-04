"""Explainability module for BurnoutLens using SHAP and permutation importance."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
import shap

from burnoutlens.config import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_COLUMNS,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
)
from burnoutlens.features import add_duplicate_group_id, make_target
from burnoutlens.leakage import assert_no_leakage
from burnoutlens.preprocessing import build_preprocessor

TARGET_MAPPING = {"Low": 0, "Medium": 1, "High": 2}
CLASS_NAMES = ["Low", "Medium", "High"]


def map_transformed_to_original(transformed_names: List[str]) -> List[str]:
    """Map each transformed ColumnTransformer feature name to its original INPUT_FEATURE.

    Parameters
    ----------
    transformed_names : List[str]
        List of 27 transformed feature names from get_feature_names_out().

    Returns
    -------
    List[str]
        List of 27 original feature names corresponding to each transformed column.
    """
    mapping = []
    for name in transformed_names:
        # Check exact match first (numeric features)
        if name in INPUT_FEATURES:
            mapping.append(name)
        else:
            # Check prefix for categorical features (e.g. Occupation_Doctor -> Occupation)
            matched = [cat for cat in CATEGORICAL_FEATURES if name.startswith(cat + "_")]
            if len(matched) == 1:
                mapping.append(matched[0])
            else:
                raise ValueError(
                    f"Could not uniquely map transformed feature '{name}' to an original feature."
                )

    if len(mapping) != len(transformed_names):
        raise ValueError("Mapping length does not match input transformed feature count.")
    for orig in mapping:
        if orig not in INPUT_FEATURES:
            raise ValueError(f"Mapped feature '{orig}' is not in INPUT_FEATURES.")

    return mapping


def fit_explainability_pipeline(
    df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[Pipeline, Any, np.ndarray, List[str], List[str]]:
    """Fit Phase 3 Logistic Regression pipeline on all rows and initialize SHAP LinearExplainer.

    Background data is strictly set to the 132 unique-profile rows (transformed), avoiding
    duplicate weighting in SHAP reference expectations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    random_state : int
        Random seed for LogisticRegression.

    Returns
    -------
    Tuple[Pipeline, Any, np.ndarray, List[str], List[str]]
        Fitted pipeline, SHAP LinearExplainer, transformed full matrix (374, 27),
        transformed feature names, and mapped original feature names.
    """
    X = df[INPUT_FEATURES].copy()
    assert_no_leakage(X)
    y = make_target(df).map(TARGET_MAPPING).values

    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])
    pipe.fit(X, y)

    trans_names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
    orig_map = map_transformed_to_original(trans_names)

    # Background: 132 unique-profile rows
    df_unique = df.drop_duplicates(subset=INPUT_FEATURES).copy()
    X_unique_trans = pipe.named_steps["preprocessor"].transform(df_unique[INPUT_FEATURES])
    X_full_trans = pipe.named_steps["preprocessor"].transform(X)

    masker = shap.maskers.Independent(data=X_unique_trans, max_samples=len(X_unique_trans))
    explainer = shap.LinearExplainer(pipe.named_steps["classifier"], masker)

    return pipe, explainer, X_full_trans, trans_names, orig_map


def aggregate_shap_to_original(
    shap_values: np.ndarray,
    orig_feature_map: List[str],
) -> np.ndarray:
    """Aggregate transformed one-hot SHAP values to original features by summation.

    Because one-hot components are linearly combined, summing their SHAP contributions
    strictly preserves additivity while collapsing back to the 12 original features.

    Parameters
    ----------
    shap_values : np.ndarray
        Array of shape (n_samples, 27, n_classes).
    orig_feature_map : List[str]
        List of length 27 mapping transformed columns to original features.

    Returns
    -------
    np.ndarray
        Aggregated SHAP array of shape (n_samples, 12, n_classes).
    """
    n_samples, n_trans, n_classes = shap_values.shape
    orig_shap = np.zeros((n_samples, len(INPUT_FEATURES), n_classes))

    for f_idx, feat in enumerate(INPUT_FEATURES):
        cols_idx = [i for i, m in enumerate(orig_feature_map) if m == feat]
        orig_shap[:, f_idx, :] = shap_values[:, cols_idx, :].sum(axis=1)

    return orig_shap


def compute_global_shap_importance(
    orig_shap_values: np.ndarray,
) -> Dict[str, Any]:
    """Compute global mean |SHAP| values overall and per class across original features.

    Parameters
    ----------
    orig_shap_values : np.ndarray
        Aggregated SHAP array of shape (n_samples, 12, n_classes).

    Returns
    -------
    Dict[str, Any]
        Dictionary with overall and per-class ranked feature importance.
    """
    mean_abs_overall = np.mean(np.abs(orig_shap_values), axis=(0, 2))
    overall_ranking = [
        {"feature": INPUT_FEATURES[i], "mean_abs_shap": float(mean_abs_overall[i])}
        for i in np.argsort(-mean_abs_overall)
    ]

    by_class = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        mean_abs_c = np.mean(np.abs(orig_shap_values[:, :, c_idx]), axis=0)
        ranked_c = [
            {"feature": INPUT_FEATURES[i], "mean_abs_shap": float(mean_abs_c[i])}
            for i in np.argsort(-mean_abs_c)
        ]
        by_class[c_name] = ranked_c

    return {
        "overall_ranking": overall_ranking,
        "by_class_ranking": by_class,
    }


def compute_permutation_importance(
    df: pd.DataFrame,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute held-out permutation importance using StratifiedGroupKFold on original features.

    Features are permuted on the validation fold *before* the preprocessing pipeline,
    properly evaluating the impact of permuting categorical features as whole entities.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    n_splits : int
        Number of CV folds (default 5).
    n_repeats : int
        Number of permutation repeats per fold (default 10).
    random_state : int
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing mean and std of macro-F1 drop per original feature.
    """
    df_grouped = add_duplicate_group_id(df)
    X = df_grouped[INPUT_FEATURES].copy()
    assert_no_leakage(X)
    y = make_target(df_grouped).map(TARGET_MAPPING).values
    groups = df_grouped["dup_group"].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_importances = {col: [] for col in INPUT_FEATURES}

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ])
        pipe.fit(X_train, y_train)
        baseline_f1 = f1_score(y_val, pipe.predict(X_val), average="macro")

        for col in INPUT_FEATURES:
            repeat_drops = []
            for r in range(n_repeats):
                rng = np.random.RandomState(random_state + fold * 100 + r)
                X_val_perm = X_val.copy()
                perm_idx = rng.permutation(len(X_val))
                X_val_perm[col] = X_val[col].iloc[perm_idx].values

                perm_f1 = f1_score(y_val, pipe.predict(X_val_perm), average="macro")
                repeat_drops.append(baseline_f1 - perm_f1)
            fold_importances[col].append(float(np.mean(repeat_drops)))

    summary = []
    for col in INPUT_FEATURES:
        mean_drop = float(np.mean(fold_importances[col]))
        std_drop = float(np.std(fold_importances[col]))
        summary.append({
            "feature": col,
            "mean_f1_drop": mean_drop,
            "std_f1_drop": std_drop,
        })

    summary = sorted(summary, key=lambda x: x["mean_f1_drop"], reverse=True)
    return {
        "ranking": summary,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
    }


def load_feature_schema(schema_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load feature schema from reports/feature_schema.json."""
    if schema_path is None:
        schema_path = Path(__file__).resolve().parent.parent.parent / "reports" / "feature_schema.json"
    else:
        schema_path = Path(schema_path)

    if not schema_path.exists():
        raise FileNotFoundError(f"Feature schema not found at {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["features"]


def explain_row(
    pipeline: Pipeline,
    explainer: Any,
    row_dict: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Explain a single user input record with prediction, probabilities, and top features.

    Validates input against reports/feature_schema.json categories and ranges.
    Raises ValueError for out-of-range values or unseen categories.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted inference pipeline.
    explainer : Any
        Fitted SHAP LinearExplainer.
    row_dict : Dict[str, Any]
        Dictionary of input features.
    schema : Optional[Dict[str, Any]]
        Optional loaded schema. Defaults to reports/feature_schema.json.

    Returns
    -------
    Dict[str, Any]
        Dictionary with predicted class, class probabilities, base value, and top 5 features.
    """
    # 1. Check for forbidden leakage columns
    for col in FORBIDDEN_COLUMNS:
        if col in row_dict:
            raise ValueError(f"Forbidden target/score column '{col}' detected in input.")

    # 2. Load schema if not provided
    if schema is None:
        schema = load_feature_schema()

    # 3. Validate features against schema
    for feat in INPUT_FEATURES:
        if feat not in row_dict:
            raise ValueError(f"Missing required input feature '{feat}'.")

        val = row_dict[feat]
        if feat not in schema:
            continue
        feat_spec = schema[feat]

        if feat_spec["type"] == "numeric":
            try:
                num_val = float(val)
            except (ValueError, TypeError):
                raise ValueError(f"Feature '{feat}' must be numeric, got {type(val)}: {val}")

            min_val = feat_spec["min"]
            max_val = feat_spec["max"]
            if num_val < min_val or num_val > max_val:
                raise ValueError(
                    f"Feature '{feat}' value {num_val} is outside allowed range [{min_val}, {max_val}]."
                )
        elif feat_spec["type"] == "categorical":
            str_val = str(val)
            allowed = feat_spec["allowed_categories"]
            if str_val not in allowed:
                raise ValueError(
                    f"Feature '{feat}' value '{str_val}' is an unseen category. Allowed: {allowed}."
                )

    # 4. Predict
    df_row = pd.DataFrame([row_dict])[INPUT_FEATURES]
    pred_idx = int(pipeline.predict(df_row)[0])
    pred_class = CLASS_NAMES[pred_idx]
    probs = pipeline.predict_proba(df_row)[0]

    # 5. Transform and compute SHAP
    preprocessor = pipeline.named_steps["preprocessor"]
    X_trans = preprocessor.transform(df_row)
    trans_names = list(preprocessor.get_feature_names_out())
    orig_map = map_transformed_to_original(trans_names)

    shap_out = explainer(X_trans)
    row_shap_27 = shap_out.values[0, :, pred_idx]
    base_val = float(shap_out.base_values[0, pred_idx])

    # 6. Aggregate to 12 original features
    orig_contributions = {}
    for feat in INPUT_FEATURES:
        cols_idx = [i for i, m in enumerate(orig_map) if m == feat]
        orig_contributions[feat] = float(row_shap_27[cols_idx].sum())

    # 7. Select top 5 features by absolute contribution
    sorted_features = sorted(orig_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_5 = []
    for feat, signed_contrib in sorted_features[:5]:
        direction = f"pushes toward {pred_class}" if signed_contrib > 0 else "pushes away"
        top_5.append({
            "feature": feat,
            "user_value": row_dict[feat],
            "signed_contribution": float(round(signed_contrib, 4)),
            "abs_contribution": float(round(abs(signed_contrib), 4)),
            "direction": direction,
        })

    return {
        "predicted_class": pred_class,
        "predicted_class_index": pred_idx,
        "class_probabilities": {
            c_name: float(round(probs[i], 4)) for i, c_name in enumerate(CLASS_NAMES)
        },
        "base_value": float(round(base_val, 4)),
        "top_features": top_5,
    }
