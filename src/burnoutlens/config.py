"""Configuration and feature definitions for BurnoutLens."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "Sleep_health_and_lifestyle_dataset.csv"

# Target configuration
TARGET_COL = "Burnout Risk"
LABELS = ["Low", "Medium", "High"]

# Target mapping thresholds (based on Stress Level integer values)
# stress <= LOW_THRESHOLD -> "Low"
# stress <= MEDIUM_THRESHOLD -> "Medium"
# stress > MEDIUM_THRESHOLD -> "High"
LOW_THRESHOLD = 4
MEDIUM_THRESHOLD = 6

# Feature groups for modeling (12 model input features, Blood Pressure string dropped)
NUMERIC_FEATURES = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Heart Rate",
    "Daily Steps",
    "Systolic BP",
    "Diastolic BP",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Occupation",
    "BMI Category",
    "Sleep Disorder",
]

INPUT_FEATURES = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "BMI Category",
    "Heart Rate",
    "Daily Steps",
    "Sleep Disorder",
    "Systolic BP",
    "Diastolic BP",
]

# Columns forbidden in feature matrix X to prevent data leakage
FORBIDDEN_COLUMNS = [
    "Burnout Risk",
    "Stress Level",
    "Burnout Index",
    "Lifestyle Score",
    "Lifestyle Category",
    "Cluster",
    "Person ID",
]
