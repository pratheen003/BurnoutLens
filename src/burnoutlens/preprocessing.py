"""Model preprocessing pipeline factory."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from burnoutlens.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Construct an unfitted ColumnTransformer preprocessor.

    Applies:
    - StandardScaler to NUMERIC_FEATURES (8 features: Age, Sleep Duration, Quality of Sleep,
      Physical Activity Level, Heart Rate, Daily Steps, Systolic BP, Diastolic BP)
    - OneHotEncoder(handle_unknown="ignore", sparse_output=False) to CATEGORICAL_FEATURES
      (4 features: Gender, Occupation, BMI Category, Sleep Disorder)

    Nothing is fitted at import time or on the full dataset.
    Calling get_feature_names_out() is supported after fitting.

    Returns
    -------
    ColumnTransformer
        Unfitted ColumnTransformer instance.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
