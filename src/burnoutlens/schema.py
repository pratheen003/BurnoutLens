"""Feature schema generation from verified data for API and validation contracts."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

from burnoutlens.config import (
    CATEGORICAL_FEATURES,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
)
from burnoutlens.data import load_raw
from burnoutlens.features import clean_data


def generate_feature_schema(
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Derive the input feature validation schema directly from the raw dataset.

    Parameters
    ----------
    output_path : Optional[Union[str, Path]]
        Path where reports/feature_schema.json will be saved.
        Defaults to PROJECT_ROOT / "reports" / "feature_schema.json".

    Returns
    -------
    Dict[str, Any]
        Schema dictionary mapping each of the 12 input features to its validation specification.
    """
    df_raw = load_raw()
    cleaned = clean_data(df_raw)

    schema: Dict[str, Any] = {
        "metadata": {
            "source_dataset": "data/raw/Sleep_health_and_lifestyle_dataset.csv",
            "total_records_derived_from": len(cleaned),
            "num_input_features": len(INPUT_FEATURES),
            "feature_list": INPUT_FEATURES,
        },
        "features": {},
    }

    # Numeric features
    for col in NUMERIC_FEATURES:
        series = cleaned[col]
        schema["features"][col] = {
            "type": "numeric",
            "dtype": str(series.dtype),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }

    # Categorical features
    for col in CATEGORICAL_FEATURES:
        series = cleaned[col]
        allowed = sorted(series.unique().tolist())
        counts = series.value_counts().to_dict()
        schema["features"][col] = {
            "type": "categorical",
            "dtype": "string",
            "allowed_categories": allowed,
            "category_counts": counts,
        }

    out_file = (
        Path(output_path)
        if output_path is not None
        else PROJECT_ROOT / "reports" / "feature_schema.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return schema


if __name__ == "__main__":
    schema_res = generate_feature_schema()
    print(f"Generated schema with {len(schema_res['features'])} features.")
