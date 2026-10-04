"""Model definitions and pipeline construction for BurnoutLens."""

from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from burnoutlens.preprocessing import build_preprocessor

# Explicit target label mapping (NOT LabelEncoder)
LABEL_TO_INT: Dict[str, int] = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}

INT_TO_LABEL: Dict[int, str] = {
    0: "Low",
    1: "Medium",
    2: "High",
}


def get_models() -> Dict[str, Any]:
    """Return dictionary of models instantiated with identical baseline hyperparameters.

    Hyperparameters are held constant from Phase 1 without tuning:
    - LogisticRegression(max_iter=1000, random_state=42)
    - DecisionTreeClassifier(max_depth=5, random_state=42)
    - RandomForestClassifier(n_estimators=200, random_state=42)
    - XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42,
                    eval_metric="mlogloss") [default multiclass objective]
    - MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                    max_iter=1000, random_state=42, early_stopping=True)

    Returns
    -------
    Dict[str, Any]
        Mapping from model display name to scikit-learn compatible classifier.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss",
        ),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        ),
    }


def make_pipeline(model: Any) -> Pipeline:
    """Wrap preprocessor and classifier in a single scikit-learn Pipeline.

    Ensures that StandardScaler and OneHotEncoder are fitted ONLY on training folds
    and training subsets, completely preventing data leakage into test sets.

    Parameters
    ----------
    model : Any
        Classifier instance.

    Returns
    -------
    Pipeline
        Pipeline containing [('prep', ColumnTransformer), ('clf', model)].
    """
    return Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            ("clf", model),
        ]
    )
