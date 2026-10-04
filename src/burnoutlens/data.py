"""Data loading functions for BurnoutLens."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from burnoutlens.config import RAW_CSV_PATH


def load_raw(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load the raw dataset.

    Uses keep_default_na=False so that literal "None" in Sleep Disorder is preserved as
    the string "None" rather than being converted to NaN. Never writes or modifies the
    underlying file.

    Parameters
    ----------
    csv_path : Optional[Union[str, Path]]
        Optional path to raw CSV file. Defaults to RAW_CSV_PATH.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    path = Path(csv_path) if csv_path is not None else RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {path}")

    # Read-only load with keep_default_na=False
    df = pd.read_csv(path, keep_default_na=False)
    return df
