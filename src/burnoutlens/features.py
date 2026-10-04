"""Feature engineering, data cleaning, and target preparation."""

import hashlib
import pandas as pd

from burnoutlens.config import (
    INPUT_FEATURES,
    LOW_THRESHOLD,
    MEDIUM_THRESHOLD,
    TARGET_COL,
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Sleep Health & Lifestyle dataset.

    Performs the following cleaning operations without mutating input:
    1. Strips leading/trailing whitespace on all string columns.
    2. Splits 'Blood Pressure' into integer 'Systolic BP' and 'Diastolic BP',
       and DROPS the raw 'Blood Pressure' string column.
    3. Normalizes BMI Category: maps 'Normal Weight' -> 'Normal'.
    4. Drops 'Person ID' identifier column if present.
    5. Arranges columns so INPUT_FEATURES are first, followed by 'Stress Level' (if present).

    Parameters
    ----------
    df : pd.DataFrame
        Input raw or uncleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    cleaned = df.copy()

    # 1. Strip whitespace on string columns
    str_cols = [
        c
        for c in cleaned.columns
        if cleaned[c].dtype == "object" or str(cleaned[c].dtype).startswith("str")
    ]
    for col in str_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    # 2. Split Blood Pressure into Systolic BP and Diastolic BP, drop raw string
    if "Blood Pressure" in cleaned.columns:
        bp_split = cleaned["Blood Pressure"].astype(str).str.split("/", expand=True)
        cleaned["Systolic BP"] = bp_split[0].astype(int)
        cleaned["Diastolic BP"] = bp_split[1].astype(int)
        cleaned.drop(columns=["Blood Pressure"], inplace=True)

    # 3. Normalize BMI: "Normal Weight" -> "Normal"
    if "BMI Category" in cleaned.columns:
        cleaned["BMI Category"] = cleaned["BMI Category"].replace(
            {"Normal Weight": "Normal"}
        )

    # 4. Drop Person ID
    if "Person ID" in cleaned.columns:
        cleaned.drop(columns=["Person ID"], inplace=True)

    # 5. Order columns: INPUT_FEATURES first, then Stress Level if present
    ordered = [c for c in INPUT_FEATURES if c in cleaned.columns]
    if "Stress Level" in cleaned.columns:
        ordered.append("Stress Level")
    for c in cleaned.columns:
        if c not in ordered:
            ordered.append(c)

    return cleaned[ordered]


def make_target(df: pd.DataFrame) -> pd.Series:
    """Derive target Burnout Risk from Stress Level.

    Mapping:
    - stress <= 4 -> "Low"
    - stress <= 6 -> "Medium"
    - stress >= 7 -> "High"

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'Stress Level' column.

    Returns
    -------
    pd.Series
        Target Series named 'Burnout Risk' with values in {'Low', 'Medium', 'High'}.
    """
    if "Stress Level" not in df.columns:
        raise KeyError("'Stress Level' column is required to derive target.")

    def _map_stress(stress: int) -> str:
        if stress <= LOW_THRESHOLD:
            return "Low"
        elif stress <= MEDIUM_THRESHOLD:
            return "Medium"
        else:
            return "High"

    target = df["Stress Level"].apply(_map_stress)
    target.name = TARGET_COL
    return target


def compute_lifestyle_score(df: pd.DataFrame) -> pd.Series:
    """Compute Lifestyle Score copied exactly from original notebook formula.

    FOR ANALYTICS ONLY. Never use this as an input feature for ML models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing sleep, activity, steps, and heart rate features.

    Returns
    -------
    pd.Series
        Lifestyle scores rounded to 2 decimal places.
    """

    def _calc_row(row):
        score = 0

        # Sleep Duration (Ideal: 7-9 hrs)
        if 7 <= row["Sleep Duration"] <= 9:
            score += 20
        elif 6 <= row["Sleep Duration"] < 7:
            score += 15
        else:
            score += 8

        # Quality of Sleep (scaled out of 9)
        score += (row["Quality of Sleep"] / 9) * 20

        # Physical Activity (scaled out of 90)
        score += (row["Physical Activity Level"] / 90) * 20

        # Daily Steps (capped at 10,000)
        score += min(row["Daily Steps"] / 10000, 1) * 20

        # Heart Rate (Ideal: 60-75 BPM)
        if 60 <= row["Heart Rate"] <= 75:
            score += 20
        elif 76 <= row["Heart Rate"] <= 80:
            score += 15
        else:
            score += 8

        return round(score, 2)

    scores = df.apply(_calc_row, axis=1)
    scores.name = "Lifestyle Score"
    return scores


def add_duplicate_group_id(df: pd.DataFrame) -> pd.DataFrame:
    """Assign deterministic hash ID for identical input-feature records.

    Used later for grouped cross-validation splitting to prevent data leakage.
    Does NOT remove any rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        New DataFrame with added 'dup_group' column.
    """
    df_out = df.copy()
    feature_cols = [c for c in INPUT_FEATURES if c in df_out.columns]
    if not feature_cols:
        raise ValueError("No input feature columns found to compute duplicate groups.")

    def _hash_row(row):
        val_str = "|".join(str(row[c]) for c in feature_cols)
        return hashlib.sha256(val_str.encode("utf-8")).hexdigest()[:16]

    df_out["dup_group"] = df_out.apply(_hash_row, axis=1)
    return df_out
