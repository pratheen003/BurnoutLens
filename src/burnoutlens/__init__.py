"""BurnoutLens package: Lifestyle-based burnout risk assessment."""

from burnoutlens.config import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_COLUMNS,
    INPUT_FEATURES,
    LABELS,
    LOW_THRESHOLD,
    MEDIUM_THRESHOLD,
    NUMERIC_FEATURES,
    RAW_CSV_PATH,
    TARGET_COL,
)
from burnoutlens.data import load_raw
from burnoutlens.features import (
    add_duplicate_group_id,
    clean_data,
    compute_lifestyle_score,
    make_target,
)
from burnoutlens.leakage import assert_no_leakage
from burnoutlens.preprocessing import build_preprocessor

__all__ = [
    "RAW_CSV_PATH",
    "TARGET_COL",
    "LABELS",
    "LOW_THRESHOLD",
    "MEDIUM_THRESHOLD",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "INPUT_FEATURES",
    "FORBIDDEN_COLUMNS",
    "load_raw",
    "clean_data",
    "make_target",
    "compute_lifestyle_score",
    "add_duplicate_group_id",
    "build_preprocessor",
    "assert_no_leakage",
]
