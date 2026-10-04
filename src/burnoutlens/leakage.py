"""Data leakage assertion guards."""

from typing import Union
import pandas as pd

from burnoutlens.config import FORBIDDEN_COLUMNS


def assert_no_leakage(X: Union[pd.DataFrame, pd.Series]) -> None:
    """Validate that feature matrix X does not contain any forbidden target or leakage columns.

    Parameters
    ----------
    X : Union[pd.DataFrame, pd.Series]
        Feature matrix or column collection to inspect.

    Raises
    ------
    ValueError
        If any column in FORBIDDEN_COLUMNS is present in X.
    """
    if isinstance(X, pd.DataFrame):
        cols = list(X.columns)
    elif isinstance(X, pd.Series):
        cols = [X.name] if X.name is not None else []
    else:
        cols = list(X)

    leaked = [col for col in cols if col in FORBIDDEN_COLUMNS]
    if leaked:
        raise ValueError(
            f"Data leakage detected! Forbidden column(s) found in feature set: {leaked}. "
            f"Allowed input features are strictly independent predictors."
        )
