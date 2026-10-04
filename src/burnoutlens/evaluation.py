"""Evaluation protocols, metric utilities, and statistical significance tests."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from burnoutlens.config import LABELS
from burnoutlens.modeling import INT_TO_LABEL, LABEL_TO_INT, make_pipeline


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for multi-class predictions.

    Label ordering is strictly [0, 1, 2] corresponding to ['Low', 'Medium', 'High'].

    Parameters
    ----------
    y_true : np.ndarray
        True integer labels.
    y_pred : np.ndarray
        Predicted integer labels.

    Returns
    -------
    Dict[str, Any]
        Dictionary with accuracy, macro_f1, per_class metrics, and confusion matrix.
    """
    labels_int = [LABEL_TO_INT[lbl] for lbl in LABELS]  # [0, 1, 2]
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_int, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_int)

    per_class = {}
    for i, lbl in enumerate(LABELS):
        per_class[lbl] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(supp[i]),
        }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "label_order": LABELS,
    }


def majority_class_baseline(y: pd.Series) -> Dict[str, float]:
    """Compute majority-class baseline metrics.

    Parameters
    ----------
    y : pd.Series
        Target series.

    Returns
    -------
    Dict[str, float]
        Baseline accuracy and macro_f1.
    """
    majority_val = y.mode()[0]
    y_pred = np.full(len(y), majority_val)
    return {
        "majority_class": INT_TO_LABEL.get(majority_val, str(majority_val)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
    }


def run_protocol_a(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Protocol A: Random Stratified 80/20 Holdout Split.

    Simulates the Phase 1 splitting strategy while using the clean Pipeline.
    Measures the number of test rows with an identical-input twin in the train set.
    """
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups, test_size=test_size, random_state=random_state, stratify=y
    )

    train_groups_set = set(g_train)
    twins_count = int(g_test.isin(train_groups_set).sum())
    twins_pct = float(twins_count / len(g_test) * 100)

    results = {}
    for name, model in models.items():
        pipe = make_pipeline(model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = compute_metrics(y_test.to_numpy(), preds)
        results[name] = {
            "metrics": metrics,
            "predictions": preds,
        }

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "twin_test_rows": twins_count,
        "twin_test_pct": twins_pct,
        "y_test": y_test.to_numpy(),
        "models": results,
    }


def run_protocol_b(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    seeds: List[int] = list(range(10)),
) -> Dict[str, Any]:
    """Protocol B: Grouped Holdout Split (Disjoint dup_groups).

    Uses StratifiedGroupKFold(n_splits=5) fold 0 as an 80/20 holdout across 10 random seeds.
    Enforces and verifies zero group overlap between train and test in every repeat.
    """
    results_by_model: Dict[str, Dict[str, List[float]]] = {
        name: {"accuracy": [], "macro_f1": []} for name in models
    }

    overlap_checks = []

    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, test_idx = next(sgkf.split(X, y, groups))

        # Assert zero group overlap
        g_train = set(groups.iloc[train_idx])
        g_test = set(groups.iloc[test_idx])
        overlap = g_train.intersection(g_test)
        if len(overlap) > 0:
            raise AssertionError(f"Group overlap detected in seed {seed}: {len(overlap)} groups")
        overlap_checks.append(len(overlap))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, model in models.items():
            pipe = make_pipeline(model)
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            results_by_model[name]["accuracy"].append(float(acc))
            results_by_model[name]["macro_f1"].append(float(f1))

    summary = {}
    for name in models:
        accs = np.array(results_by_model[name]["accuracy"])
        f1s = np.array(results_by_model[name]["macro_f1"])
        summary[name] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_macro_f1": float(np.mean(f1s)),
            "std_macro_f1": float(np.std(f1s)),
            "raw_accuracies": accs.tolist(),
            "raw_macro_f1s": f1s.tolist(),
        }

    return {
        "seeds": seeds,
        "all_zero_overlap": all(c == 0 for c in overlap_checks),
        "models": summary,
    }


def run_protocol_c(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    primary_seed: int = 42,
    repeat_seeds: List[int] = list(range(5)),
) -> Dict[str, Any]:
    """Protocol C: Grouped 5-Fold Cross-Validation.

    Evaluates StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
    capturing per-fold metrics and pooled out-of-fold predictions.
    Also repeats across 5 random seeds (0-4) to report mean across repetitions.
    """
    # 1. Primary evaluation with seed 42
    sgkf_primary = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=primary_seed)

    primary_results = {}
    oof_predictions = {}

    for name, model in models.items():
        fold_accs = []
        fold_f1s = []
        oof_preds = np.zeros(len(y), dtype=int)

        for fold_idx, (train_idx, val_idx) in enumerate(sgkf_primary.split(X, y, groups)):
            # Assert zero group overlap
            g_train = set(groups.iloc[train_idx])
            g_val = set(groups.iloc[val_idx])
            if len(g_train.intersection(g_val)) > 0:
                raise AssertionError(f"Fold {fold_idx} has group overlap!")

            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipe = make_pipeline(model)
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)

            oof_preds[val_idx] = preds
            fold_accs.append(float(accuracy_score(y_val, preds)))
            fold_f1s.append(float(f1_score(y_val, preds, average="macro", zero_division=0)))

        primary_metrics = compute_metrics(y.to_numpy(), oof_preds)
        primary_results[name] = {
            "fold_accuracies": fold_accs,
            "fold_macro_f1s": fold_f1s,
            "mean_accuracy": float(np.mean(fold_accs)),
            "std_accuracy": float(np.std(fold_accs)),
            "mean_macro_f1": float(np.mean(fold_f1s)),
            "std_macro_f1": float(np.std(fold_f1s)),
            "pooled_metrics": primary_metrics,
        }
        oof_predictions[name] = oof_preds

    # 2. Repeated CV over repeat_seeds (0 to 4)
    repeated_summary: Dict[str, Dict[str, List[float]]] = {
        name: {"accuracies": [], "macro_f1s": []} for name in models
    }

    for seed in repeat_seeds:
        sgkf_rep = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for name, model in models.items():
            rep_fold_accs = []
            rep_fold_f1s = []
            for train_idx, val_idx in sgkf_rep.split(X, y, groups):
                pipe = make_pipeline(model)
                pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = pipe.predict(X.iloc[val_idx])
                rep_fold_accs.append(accuracy_score(y.iloc[val_idx], preds))
                rep_fold_f1s.append(f1_score(y.iloc[val_idx], preds, average="macro", zero_division=0))
            repeated_summary[name]["accuracies"].append(float(np.mean(rep_fold_accs)))
            repeated_summary[name]["macro_f1s"].append(float(np.mean(rep_fold_f1s)))

    repeated_results = {}
    for name in models:
        rep_acc = np.array(repeated_summary[name]["accuracies"])
        rep_f1 = np.array(repeated_summary[name]["macro_f1s"])
        repeated_results[name] = {
            "repeated_mean_accuracy": float(np.mean(rep_acc)),
            "repeated_std_accuracy": float(np.std(rep_acc)),
            "repeated_mean_macro_f1": float(np.mean(rep_f1)),
            "repeated_std_macro_f1": float(np.std(rep_f1)),
            "seed_accuracies": rep_acc.tolist(),
            "seed_macro_f1s": rep_f1.tolist(),
        }

    return {
        "primary_seed": primary_seed,
        "repeat_seeds": repeat_seeds,
        "primary_results": primary_results,
        "repeated_results": repeated_results,
        "oof_predictions": oof_predictions,
        "y_true": y.to_numpy(),
    }


def run_protocol_d(
    models: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    seeds: List[int] = list(range(5)),
) -> Dict[str, Any]:
    """Protocol D: Random (Non-Grouped) 5-Fold Cross-Validation Control.

    Uses StratifiedKFold(n_splits=5, shuffle=True) across seeds (0-4) with the same
    clean Pipeline and same 5 models as Protocol C.
    The split does not require groups. If groups are provided, measures per-fold contamination:
    count and fraction of validation rows whose dup_group appears in the training fold.
    """
    twin_counts_per_fold: List[int] = []
    twin_pcts_per_fold: List[float] = []

    repeated_summary: Dict[str, Dict[str, List[float]]] = {
        name: {"accuracies": [], "macro_f1s": []} for name in models
    }

    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        # Measure twin contamination across folds if groups provided
        if groups is not None:
            for train_idx, val_idx in skf.split(X, y):
                g_train = set(groups.iloc[train_idx])
                g_val = groups.iloc[val_idx]
                twins = int(g_val.isin(g_train).sum())
                twin_counts_per_fold.append(twins)
                twin_pcts_per_fold.append(float(twins / len(val_idx) * 100))

        for name, model in models.items():
            seed_fold_accs = []
            seed_fold_f1s = []
            for train_idx, val_idx in skf.split(X, y):
                pipe = make_pipeline(model)
                pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = pipe.predict(X.iloc[val_idx])
                seed_fold_accs.append(accuracy_score(y.iloc[val_idx], preds))
                seed_fold_f1s.append(f1_score(y.iloc[val_idx], preds, average="macro", zero_division=0))
            repeated_summary[name]["accuracies"].append(float(np.mean(seed_fold_accs)))
            repeated_summary[name]["macro_f1s"].append(float(np.mean(seed_fold_f1s)))

    models_summary = {}
    for name in models:
        rep_acc = np.array(repeated_summary[name]["accuracies"])
        rep_f1 = np.array(repeated_summary[name]["macro_f1s"])
        models_summary[name] = {
            "mean_accuracy": float(np.mean(rep_acc)),
            "std_accuracy": float(np.std(rep_acc)),
            "mean_macro_f1": float(np.mean(rep_f1)),
            "std_macro_f1": float(np.std(rep_f1)),
            "seed_accuracies": rep_acc.tolist(),
            "seed_macro_f1s": rep_f1.tolist(),
        }

    return {
        "seeds": seeds,
        "avg_twin_rows_per_fold": float(np.mean(twin_counts_per_fold)) if twin_counts_per_fold else 0.0,
        "std_twin_rows_per_fold": float(np.std(twin_counts_per_fold)) if twin_counts_per_fold else 0.0,
        "avg_twin_pct_per_fold": float(np.mean(twin_pcts_per_fold)) if twin_pcts_per_fold else 0.0,
        "std_twin_pct_per_fold": float(np.std(twin_pcts_per_fold)) if twin_pcts_per_fold else 0.0,
        "total_folds_evaluated": len(twin_counts_per_fold),
        "models": models_summary,
    }


def mcnemar_test(
    y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray
) -> Dict[str, Any]:
    """Perform McNemar's exact test using scipy.stats.binomtest on discordant pairs.

    Parameters
    ----------
    y_true : np.ndarray
        True ground truth labels.
    y_pred1 : np.ndarray
        Predictions of Model 1.
    y_pred2 : np.ndarray
        Predictions of Model 2.

    Returns
    -------
    Dict[str, Any]
        Contingency matrix discordant counts (b, c), total discordant n, and two-sided p-value.
    """
    correct1 = y_pred1 == y_true
    correct2 = y_pred2 == y_true

    # b: Model 1 correct, Model 2 incorrect
    b = int(np.sum(correct1 & ~correct2))
    # c: Model 1 incorrect, Model 2 correct
    c = int(np.sum(~correct1 & correct2))
    n_discordant = b + c

    if n_discordant == 0:
        p_val = 1.0
    else:
        test_res = binomtest(b, n=n_discordant, p=0.5, alternative="two-sided")
        p_val = float(test_res.pvalue)

    return {
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "p_value": p_val,
        "significant_at_05": p_val < 0.05,
    }
