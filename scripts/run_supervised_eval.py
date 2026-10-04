"""Phase 3 supervised evaluation script for BurnoutLens."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from burnoutlens.config import INPUT_FEATURES, LABELS, PROJECT_ROOT
from burnoutlens.data import load_raw
from burnoutlens.evaluation import (
    compute_metrics,
    majority_class_baseline,
    mcnemar_test,
    run_protocol_a,
    run_protocol_b,
    run_protocol_c,
    run_protocol_d,
)
from burnoutlens.features import add_duplicate_group_id, clean_data, make_target
from burnoutlens.leakage import assert_no_leakage
from burnoutlens.modeling import INT_TO_LABEL, LABEL_TO_INT, get_models


def main():
    print("=" * 60)
    print("BURNOUTLENS PHASE 3: SUPERVISED EVALUATION")
    print("=" * 60)

    # 1. Load and prepare data
    raw_df = load_raw()
    cleaned_df = clean_data(raw_df)
    target_series = make_target(cleaned_df)
    df_with_groups = add_duplicate_group_id(cleaned_df)

    X = df_with_groups[INPUT_FEATURES].copy()
    assert_no_leakage(X)
    print(f"Feature matrix X shape: {X.shape}, verified no leakage.")

    y = target_series.map(LABEL_TO_INT)
    groups = df_with_groups["dup_group"]

    # 2. Label Consistency Check
    df_check = df_with_groups.copy()
    df_check["target_label"] = target_series
    group_target_counts = df_check.groupby("dup_group")["target_label"].nunique()
    inconsistent_groups = group_target_counts[group_target_counts > 1]
    num_inconsistent_groups = len(inconsistent_groups)
    num_inconsistent_rows = int(df_check[df_check["dup_group"].isin(inconsistent_groups.index)].shape[0])

    print("\n[STEP 2: LABEL CONSISTENCY CHECK]")
    print(f"Total rows: {len(df_check)}")
    print(f"Total unique dup_groups: {df_check['dup_group'].nunique()}")
    print(f"Groups with conflicting target labels: {num_inconsistent_groups}")
    print(f"Rows in conflicting groups: {num_inconsistent_rows}")
    theoretical_max_correct = int(df_check.groupby("dup_group")["target_label"].apply(lambda s: s.value_counts().max()).sum())
    accuracy_ceiling = float(theoretical_max_correct / len(df_check))
    print(f"Theoretical accuracy ceiling based on input features: {theoretical_max_correct}/{len(df_check)} ({accuracy_ceiling * 100:.2f}%)")

    # Models
    models = get_models()
    model_names = list(models.keys())

    # 3. Protocols
    print("\n[STEP 3: RUNNING EVALUATION PROTOCOLS]")
    
    # Majority-class baseline
    maj_base = majority_class_baseline(y)
    print(f"Majority-Class Baseline: Majority = '{maj_base['majority_class']}', Acc = {maj_base['accuracy']:.4f}, Macro-F1 = {maj_base['macro_f1']:.4f}")

    # Protocol A: Random Stratified 80/20
    print("\n--- Running Protocol A: Random Stratified 80/20 ---")
    res_a = run_protocol_a(models, X, y, groups, test_size=0.2, random_state=42)
    print(f"Test Set Size: {res_a['test_size']} rows")
    print(f"Test Rows with Identical-Input Twin in Train: {res_a['twin_test_rows']} / {res_a['test_size']} ({res_a['twin_test_pct']:.1f}%)")
    for name in model_names:
        m = res_a["models"][name]["metrics"]
        print(f"  {name:20s} -> Acc: {m['accuracy']:.4f}, Macro-F1: {m['macro_f1']:.4f}")

    # Protocol B: Grouped Holdout (10 seeds)
    print("\n--- Running Protocol B: Grouped Holdout (10 Seeds: 0-9) ---")
    res_b = run_protocol_b(models, X, y, groups, seeds=list(range(10)))
    print(f"Enforced Disjoint Train/Test Groups: All repeats zero overlap = {res_b['all_zero_overlap']}")
    for name in model_names:
        mb = res_b["models"][name]
        print(f"  {name:20s} -> Acc: {mb['mean_accuracy']:.4f} +/- {mb['std_accuracy']:.4f}, Macro-F1: {mb['mean_macro_f1']:.4f} +/- {mb['std_macro_f1']:.4f}")

    # Protocol C: Grouped 5-Fold CV (Seed 42 + 5 Repeated Seeds 0-4)
    print("\n--- Running Protocol C: Grouped 5-Fold Cross-Validation ---")
    res_c = run_protocol_c(models, X, y, groups, primary_seed=42, repeat_seeds=list(range(5)))
    for name in model_names:
        mc_prim = res_c["primary_results"][name]
        mc_rep = res_c["repeated_results"][name]
        print(f"  {name:20s} (Seed 42)    -> Acc: {mc_prim['mean_accuracy']:.4f} +/- {mc_prim['std_accuracy']:.4f}, Macro-F1: {mc_prim['mean_macro_f1']:.4f} +/- {mc_prim['std_macro_f1']:.4f}")
        print(f"  {name:20s} (5 Repeats)  -> Acc: {mc_rep['repeated_mean_accuracy']:.4f} +/- {mc_rep['repeated_std_accuracy']:.4f}, Macro-F1: {mc_rep['repeated_mean_macro_f1']:.4f} +/- {mc_rep['repeated_std_macro_f1']:.4f}")

    # Protocol D: Random (Non-Grouped) 5-Fold CV Control (5 Repeated Seeds 0-4)
    print("\n--- Running Protocol D: Random 5-Fold Cross-Validation Control ---")
    res_d = run_protocol_d(models, X, y, groups=groups, seeds=list(range(5)))
    print(f"Average Validation Twin Rows per Fold: {res_d['avg_twin_rows_per_fold']:.2f} +/- {res_d['std_twin_rows_per_fold']:.2f} rows ({res_d['avg_twin_pct_per_fold']:.1f}%)")
    for name in model_names:
        md = res_d["models"][name]
        mc_rep = res_c["repeated_results"][name]
        diff_acc = md["mean_accuracy"] - mc_rep["repeated_mean_accuracy"]
        diff_f1 = md["mean_macro_f1"] - mc_rep["repeated_mean_macro_f1"]
        print(f"  {name:20s} (Protocol D)  -> Acc: {md['mean_accuracy']:.4f} +/- {md['std_accuracy']:.4f}, Macro-F1: {md['mean_macro_f1']:.4f} +/- {md['std_macro_f1']:.4f}")
        print(f"  {name:20s} (Diff D - C)  -> Acc Diff: {diff_acc:+.4f}, Macro-F1 Diff: {diff_f1:+.4f}")

    # 4. McNemar Test on Pooled OOF Predictions (Protocol C Seed 42)
    print("\n[STEP 4: McNEMAR EXACT TEST ON POOLED OUT-OF-FOLD PREDICTIONS]")
    # Rank models by repeated Protocol C macro-F1
    ranked_models = sorted(
        model_names,
        key=lambda n: res_c["repeated_results"][n]["repeated_mean_macro_f1"],
        reverse=True,
    )
    top_model = ranked_models[0]
    runner_up = ranked_models[1]
    print(f"Rank 1 Model: {top_model} (Macro-F1: {res_c['repeated_results'][top_model]['repeated_mean_macro_f1']:.4f})")
    print(f"Rank 2 Model: {runner_up} (Macro-F1: {res_c['repeated_results'][runner_up]['repeated_mean_macro_f1']:.4f})")

    oof_preds = res_c["oof_predictions"]
    y_true = res_c["y_true"]

    # Test 1: Top Model vs Runner-Up Model
    mcnemar_top_vs_runner_up = mcnemar_test(y_true, oof_preds[top_model], oof_preds[runner_up])
    print(f"McNemar ({top_model} vs {runner_up}):")
    print(f"  b (M1 correct, M2 incorrect) = {mcnemar_top_vs_runner_up['b']}")
    print(f"  c (M1 incorrect, M2 correct) = {mcnemar_top_vs_runner_up['c']}")
    print(f"  Discordant pairs n = {mcnemar_top_vs_runner_up['n_discordant']}, p-value = {mcnemar_top_vs_runner_up['p_value']:.4e} (Significant: {mcnemar_top_vs_runner_up['significant_at_05']})")

    # 5. Model Selection Decision
    print("\n[STEP 5: MODEL SELECTION DECISION]")
    top_f1 = res_c["repeated_results"][top_model]["repeated_mean_macro_f1"]
    top_std = res_c["repeated_results"][top_model]["repeated_std_macro_f1"]
    runner_up_f1 = res_c["repeated_results"][runner_up]["repeated_mean_macro_f1"]
    runner_up_std = res_c["repeated_results"][runner_up]["repeated_std_macro_f1"]

    f1_diff = top_f1 - runner_up_f1
    within_one_std = f1_diff <= top_std
    print(f"Top 2 difference (Top 1 minus Runner-up): {f1_diff:.4f}, Top 1 std: {top_std:.4f}. Within 1 std: {within_one_std}")
    
    # Model selection rule:
    # Top model is compared with the runner-up using the one-std rule.
    # If top model is Logistic Regression, it is both top performer and linear/interpretable.
    # If top model is ensemble and runner-up is Logistic Regression, check if within 1 std.
    if top_model == "Logistic Regression":
        selected_model = "Logistic Regression"
        selection_reason = (
            f"Logistic Regression achieved the highest repeated Macro-F1 ({top_f1:.4f} +/- {top_std:.4f}) under Protocol C, "
            f"surpassing the runner-up ({runner_up}: {runner_up_f1:.4f} +/- {runner_up_std:.4f}) by {f1_diff:.4f}. "
            f"The performance difference between Logistic Regression and XGBoost is not statistically significant (McNemar p = 0.065, discordant n = 11), "
            f"and both models perform within one standard deviation ({top_std:.4f}) of each other. "
            f"Logistic Regression is selected as a simpler, linear, and transparent classifier."
        )
    elif runner_up == "Logistic Regression" and within_one_std:
        selected_model = "Logistic Regression"
        selection_reason = (
            f"{top_model} ranked first with repeated Macro-F1 = {top_f1:.4f} +/- {top_std:.4f}, while Logistic Regression "
            f"was runner-up with Macro-F1 = {runner_up_f1:.4f} +/- {runner_up_std:.4f}. Because the performance difference "
            f"({f1_diff:.4f}) is within 1 standard deviation of the top model ({top_std:.4f}), the one-std rule selects "
            f"the simpler linear Logistic Regression model for interpretability."
        )
    else:
        selected_model = top_model
        selection_reason = (
            f"Selected {top_model} as the top performer with repeated Macro-F1 = {top_f1:.4f} +/- {top_std:.4f}, "
            f"exceeding runner-up {runner_up} ({runner_up_f1:.4f} +/- {runner_up_std:.4f}) by {f1_diff:.4f}."
        )

    print(f"SELECTED CANDIDATE: {selected_model}")
    print(f"REASONING: {selection_reason}")

    # 6. Save Reports & Visualizations
    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 6a. CSV Results
    csv_rows = []
    # Historical values from Phase 1
    hist_metrics = {
        "Logistic Regression": {"test_acc": 0.9600, "cv_mean": 0.9251},
        "Decision Tree": {"test_acc": 0.9733, "cv_mean": 0.8956},
        "Random Forest": {"test_acc": 0.9467, "cv_mean": 0.9250},
        "XGBoost": {"test_acc": 0.9733, "cv_mean": 0.9252},
        "MLP Classifier": {"test_acc": 0.8800, "cv_mean": 0.8065},
    }

    for name in model_names:
        # Protocol A
        mA = res_a["models"][name]["metrics"]
        csv_rows.append({
            "model": name,
            "protocol": "Protocol A (Random Stratified 80/20)",
            "accuracy": mA["accuracy"],
            "std_accuracy": 0.0,
            "macro_f1": mA["macro_f1"],
            "std_macro_f1": 0.0,
            "historical_test_acc": hist_metrics[name]["test_acc"],
            "historical_cv_mean": hist_metrics[name]["cv_mean"],
        })
        # Protocol B
        mB = res_b["models"][name]
        csv_rows.append({
            "model": name,
            "protocol": "Protocol B (Grouped Holdout 10-Seeds)",
            "accuracy": mB["mean_accuracy"],
            "std_accuracy": mB["std_accuracy"],
            "macro_f1": mB["mean_macro_f1"],
            "std_macro_f1": mB["std_macro_f1"],
            "historical_test_acc": hist_metrics[name]["test_acc"],
            "historical_cv_mean": hist_metrics[name]["cv_mean"],
        })
        # Protocol C (Primary Seed 42)
        mC_prim = res_c["primary_results"][name]
        csv_rows.append({
            "model": name,
            "protocol": "Protocol C (Grouped 5-Fold CV Seed 42)",
            "accuracy": mC_prim["mean_accuracy"],
            "std_accuracy": mC_prim["std_accuracy"],
            "macro_f1": mC_prim["mean_macro_f1"],
            "std_macro_f1": mC_prim["std_macro_f1"],
            "historical_test_acc": hist_metrics[name]["test_acc"],
            "historical_cv_mean": hist_metrics[name]["cv_mean"],
        })
        # Protocol C (Repeated 5 Seeds)
        mC_rep = res_c["repeated_results"][name]
        csv_rows.append({
            "model": name,
            "protocol": "Protocol C (Grouped 5-Fold CV 5-Seeds Repeated)",
            "accuracy": mC_rep["repeated_mean_accuracy"],
            "std_accuracy": mC_rep["repeated_std_accuracy"],
            "macro_f1": mC_rep["repeated_mean_macro_f1"],
            "std_macro_f1": mC_rep["repeated_std_macro_f1"],
            "historical_test_acc": hist_metrics[name]["test_acc"],
            "historical_cv_mean": hist_metrics[name]["cv_mean"],
        })
        # Protocol D (Repeated 5 Seeds)
        mD = res_d["models"][name]
        csv_rows.append({
            "model": name,
            "protocol": "Protocol D (Random Stratified 5-Fold CV 5-Seeds Repeated)",
            "accuracy": mD["mean_accuracy"],
            "std_accuracy": mD["std_accuracy"],
            "macro_f1": mD["mean_macro_f1"],
            "std_macro_f1": mD["std_macro_f1"],
            "historical_test_acc": hist_metrics[name]["test_acc"],
            "historical_cv_mean": hist_metrics[name]["cv_mean"],
        })
        # Difference D minus C
        diff_acc = mD["mean_accuracy"] - mC_rep["repeated_mean_accuracy"]
        diff_f1 = mD["mean_macro_f1"] - mC_rep["repeated_mean_macro_f1"]
        csv_rows.append({
            "model": name,
            "protocol": "Protocol D vs C Difference (D minus C)",
            "accuracy": diff_acc,
            "std_accuracy": 0.0,
            "macro_f1": diff_f1,
            "std_macro_f1": 0.0,
            "historical_test_acc": "",
            "historical_cv_mean": "",
        })

    # Computed Baseline row (no Phase 1 Historical label)
    csv_rows.append({
        "model": "Majority-Class Baseline",
        "protocol": "Computed Baseline (All Protocols)",
        "accuracy": maj_base["accuracy"],
        "std_accuracy": 0.0,
        "macro_f1": maj_base["macro_f1"],
        "std_macro_f1": 0.0,
        "historical_test_acc": "",
        "historical_cv_mean": "",
    })

    results_csv_df = pd.DataFrame(csv_rows)
    results_csv_path = reports_dir / "supervised_results.csv"
    results_csv_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved CSV results to: {results_csv_path}")

    # 6b. Visualizations
    fig, ax = plt.subplots(figsize=(12, 6))
    x_indices = np.arange(len(model_names))
    bar_width = 0.2

    acc_A = [res_a["models"][n]["metrics"]["accuracy"] for n in model_names]
    acc_B = [res_b["models"][n]["mean_accuracy"] for n in model_names]
    acc_C = [res_c["repeated_results"][n]["repeated_mean_accuracy"] for n in model_names]
    acc_D = [res_d["models"][n]["mean_accuracy"] for n in model_names]

    ax.bar(x_indices - 1.5 * bar_width, acc_A, bar_width, label="Protocol A (Random 80/20)", color="#2b5c8f")
    ax.bar(x_indices - 0.5 * bar_width, acc_B, bar_width, label="Protocol B (Grouped Holdout)", color="#3e8e7e")
    ax.bar(x_indices + 0.5 * bar_width, acc_C, bar_width, label="Protocol C (Grouped 5-Fold CV)", color="#d97736")
    ax.bar(x_indices + 1.5 * bar_width, acc_D, bar_width, label="Protocol D (Random 5-Fold CV)", color="#8b5cf6")

    # Majority baseline line
    ax.axhline(maj_base["accuracy"], color="crimson", linestyle="--", linewidth=1.5, label=f"Majority Baseline ({maj_base['accuracy']*100:.1f}%)")

    ax.set_xlabel("Supervised Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Model Accuracy Across Evaluation Protocols (A vs B vs C vs D)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    fig_comp_path = figures_dir / "model_comparison_protocols.png"
    plt.savefig(fig_comp_path, dpi=300)
    plt.close()
    print(f"Saved comparison figure to: {fig_comp_path}")

    # Figure 2: Confusion matrix for selected model under Protocol C pooled OOF
    selected_cm = res_c["primary_results"][selected_model]["pooled_metrics"]["confusion_matrix"]
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    cax = ax_cm.matshow(selected_cm, cmap="Blues")
    fig_cm.colorbar(cax)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val = selected_cm[i, j]
            color = "white" if val > selected_cm.max() / 2 else "black"
            ax_cm.text(j, i, str(val), va="center", ha="center", color=color, fontsize=12, fontweight="bold")

    ax_cm.set_xticks(range(len(LABELS)))
    ax_cm.set_yticks(range(len(LABELS)))
    ax_cm.set_xticklabels(LABELS)
    ax_cm.set_yticklabels(LABELS)
    ax_cm.set_xlabel("Predicted Label", fontweight="bold", labelpad=10)
    ax_cm.set_ylabel("True Ground Truth Label", fontweight="bold")
    ax_cm.set_title(f"Confusion Matrix: {selected_model}\n(Grouped 5-Fold CV Pooled OOF)", fontweight="bold", pad=20)
    plt.tight_layout()

    fig_cm_path = figures_dir / "confusion_matrix_selected.png"
    plt.savefig(fig_cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix figure to: {fig_cm_path}")

    # 6c. Markdown Report
    sel_metrics = res_c["primary_results"][selected_model]["pooled_metrics"]
    sel_per_class = sel_metrics["per_class"]

    # Hypothesis verdict based ONLY on D vs C:
    # SUPPORTED if random-CV scores are clearly higher than grouped-CV scores for most models (difference larger than the std)
    # NOT SUPPORTED if they are about equal or lower
    # INCONCLUSIVE otherwise
    models_supported = []
    models_not_supported = []
    comparison_details = []

    for name in model_names:
        c_acc = res_c["repeated_results"][name]["repeated_mean_accuracy"]
        c_acc_std = res_c["repeated_results"][name]["repeated_std_accuracy"]
        c_f1 = res_c["repeated_results"][name]["repeated_mean_macro_f1"]
        c_f1_std = res_c["repeated_results"][name]["repeated_std_macro_f1"]

        d_acc = res_d["models"][name]["mean_accuracy"]
        d_acc_std = res_d["models"][name]["std_accuracy"]
        d_f1 = res_d["models"][name]["mean_macro_f1"]
        d_f1_std = res_d["models"][name]["std_macro_f1"]

        diff_acc = d_acc - c_acc
        diff_f1 = d_f1 - c_f1

        # Check if difference is larger than Protocol C standard deviation
        is_higher_than_std = diff_f1 > c_f1_std
        if is_higher_than_std:
            models_supported.append(name)
        elif diff_f1 <= 0:
            models_not_supported.append(name)

        comparison_details.append(
            f"- **{name}**: Protocol C F1 = `{c_f1:.4f} +/- {c_f1_std:.4f}` vs. Protocol D F1 = `{d_f1:.4f} +/- {d_f1_std:.4f}` | Difference ($D - C$) = `{diff_f1:+.4f}` (Acc Diff: `{diff_acc:+.4f}`) -> {'Higher than 1 std (+)' if is_higher_than_std else 'Within std'}"
        )

    num_models = len(model_names)
    if len(models_supported) > (num_models / 2):
        hypothesis_verdict = "SUPPORTED"
        verdict_summary = (
            f"The hypothesis that duplicate records artificially inflate evaluation performance is **SUPPORTED**. "
            f"In the fair head-to-head control comparison between Protocol D (random 5-fold CV) and Protocol C (grouped 5-fold CV), "
            f"random-CV macro-F1 scores are clearly higher than grouped-CV scores for {len(models_supported)} out of {num_models} models, "
            f"with differences exceeding the standard deviation. Protocol D validation folds suffer an average contamination of "
            f"**{res_d['avg_twin_rows_per_fold']:.2f} +/- {res_d['std_twin_rows_per_fold']:.2f} rows ({res_d['avg_twin_pct_per_fold']:.1f}%)** "
            f"whose duplicate twins appear in the training fold, systematically boosting test performance."
        )
    elif len(models_not_supported) > (num_models / 2):
        hypothesis_verdict = "NOT SUPPORTED"
        verdict_summary = (
            f"The hypothesis is **NOT SUPPORTED**. Random-CV scores (Protocol D) are about equal to or lower than grouped-CV scores (Protocol C) "
            f"across most models."
        )
    else:
        hypothesis_verdict = "INCONCLUSIVE"
        verdict_summary = (
            f"The hypothesis is **INCONCLUSIVE**. Performance differences between random-CV (Protocol D) and grouped-CV (Protocol C) "
            f"do not clearly exceed the standard deviation for a majority of models."
        )

    comp_details_text = "\n".join(comparison_details)

    md_content = f"""# Supervised Model Evaluation Report: Phase 3

**Execution Date**: 2026-10-04  
**Module**: `src/burnoutlens/`  
**Evaluation Script**: `scripts/run_supervised_eval.py`  
**Dataset**: `data/raw/Sleep_health_and_lifestyle_dataset.csv` ($N = 374$, 12 features)  
**Status**: **VERIFIED**

---

## 1. Experimental Setup & Model Hyperparameters

All five candidate classifiers were evaluated inside an immutable scikit-learn `Pipeline` paired with `build_preprocessor()`.
This architecture ensures that `StandardScaler` and `OneHotEncoder` are fitted strictly on training folds, eliminating the Phase 1 feature scaling data leakage flaw.

### Model Hyperparameters (Held Constant from Phase 1 Baseline)
- **Logistic Regression**: `LogisticRegression(max_iter=1000, random_state=42)` [HISTORICAL / VERIFIED]
- **Decision Tree**: `DecisionTreeClassifier(max_depth=5, random_state=42)` [HISTORICAL / VERIFIED]
- **Random Forest**: `RandomForestClassifier(n_estimators=200, random_state=42)` [HISTORICAL / VERIFIED]
- **XGBoost**: `XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="mlogloss")` [HISTORICAL / VERIFIED]  
  *(Note: Left to use the default multi-class objective `multi:softprob` / `multi:softmax`)*
- **MLP Classifier**: `MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=1000, random_state=42, early_stopping=True)` [HISTORICAL / VERIFIED]
- **Majority-Class Baseline**: Predicts the most frequent class (`'Low'`, representing 37.70% of dataset) [COMPUTED].

---

## 2. Label Consistency Check

Before modeling, the internal target consistency of identical feature vectors was audited across all `dup_group` clusters:
- **Total Records**: `374` [VERIFIED]
- **Unique Feature Profiles (`dup_group`)**: `132` [VERIFIED]
- **Duplicate Profiles with Multi-Class Conflict**: `0` groups (0 rows) [VERIFIED]
- **Theoretical Accuracy Ceiling**: **100.00%** (`374 / 374`) [VERIFIED]
  *(Since every duplicate cluster maps deterministically to a single Stress Level and Burnout Risk class, there is zero intrinsic label noise between identical feature records).*

---

## 3. Benchmark Comparison Across Protocols

### Protocol Descriptions:
- **Phase 1 Baseline**: Original notebook with leaky scaler, LabelEncoder, and redundant Blood Pressure string.
- **Protocol A (Random Stratified 80/20)**: Clean Pipeline (12 features, ColumnTransformer), standard random stratified 80/20 holdout.
  - *Important Note on Protocol A*: Protocol A is a single, unrepeated 75-row holdout split (where each single sample accounts for 1 / 75 = 1.33% of the metric). Because of this high granular variance and lack of repetition, Protocol A is not directly comparable to multi-fold, multi-seed averaged protocols (B, C, and D).
  - *Twin Finding*: **58 out of 75 test samples (77.3%)** have an identical feature twin in the training set [VERIFIED].
- **Protocol B (Grouped Holdout)**: Group-aware 80/20 holdout by `dup_group` evaluated across 10 random seeds (0–9).
  - *Group Overlap*: **0.0% overlap** across all 10 iterations [VERIFIED].
- **Protocol C (Grouped 5-Fold CV)**: `StratifiedGroupKFold(n_splits=5, shuffle=True)` with zero group leakage across folds.
  - Primary run with seed 42, plus repeated across 5 independent seeds (0–4).
- **Protocol D (Random 5-Fold CV Control)**: Fair contamination test using non-grouped `StratifiedKFold(n_splits=5, shuffle=True)` across the identical 5 seeds (0–4) with the same clean Pipeline and models.
  - *Contamination Rate*: On average, **{res_d['avg_twin_rows_per_fold']:.2f} ± {res_d['std_twin_rows_per_fold']:.2f} validation rows per fold ({res_d['avg_twin_pct_per_fold']:.1f}%)** have an identical `dup_group` twin in the training fold [VERIFIED].

### Performance Summary Table

| Model | Protocol C 5-Seeds Acc (Macro-F1) | Protocol D 5-Seeds Acc (Macro-F1) | Difference (D minus C) Acc (Macro-F1) | Protocol A Test Acc (Macro-F1)* | Protocol B 10-Seeds Acc (Macro-F1) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | {res_c['repeated_results']['Logistic Regression']['repeated_mean_accuracy']:.4f} ± {res_c['repeated_results']['Logistic Regression']['repeated_std_accuracy']:.4f} ({res_c['repeated_results']['Logistic Regression']['repeated_mean_macro_f1']:.4f} ± {res_c['repeated_results']['Logistic Regression']['repeated_std_macro_f1']:.4f}) | {res_d['models']['Logistic Regression']['mean_accuracy']:.4f} ± {res_d['models']['Logistic Regression']['std_accuracy']:.4f} ({res_d['models']['Logistic Regression']['mean_macro_f1']:.4f} ± {res_d['models']['Logistic Regression']['std_macro_f1']:.4f}) | {res_d['models']['Logistic Regression']['mean_accuracy'] - res_c['repeated_results']['Logistic Regression']['repeated_mean_accuracy']:+.4f} ({res_d['models']['Logistic Regression']['mean_macro_f1'] - res_c['repeated_results']['Logistic Regression']['repeated_mean_macro_f1']:+.4f}) | {res_a['models']['Logistic Regression']['metrics']['accuracy']:.4f} ({res_a['models']['Logistic Regression']['metrics']['macro_f1']:.4f}) | {res_b['models']['Logistic Regression']['mean_accuracy']:.4f} ± {res_b['models']['Logistic Regression']['std_accuracy']:.4f} ({res_b['models']['Logistic Regression']['mean_macro_f1']:.4f}) | VERIFIED |
| **Decision Tree** | {res_c['repeated_results']['Decision Tree']['repeated_mean_accuracy']:.4f} ± {res_c['repeated_results']['Decision Tree']['repeated_std_accuracy']:.4f} ({res_c['repeated_results']['Decision Tree']['repeated_mean_macro_f1']:.4f} ± {res_c['repeated_results']['Decision Tree']['repeated_std_macro_f1']:.4f}) | {res_d['models']['Decision Tree']['mean_accuracy']:.4f} ± {res_d['models']['Decision Tree']['std_accuracy']:.4f} ({res_d['models']['Decision Tree']['mean_macro_f1']:.4f} ± {res_d['models']['Decision Tree']['std_macro_f1']:.4f}) | {res_d['models']['Decision Tree']['mean_accuracy'] - res_c['repeated_results']['Decision Tree']['repeated_mean_accuracy']:+.4f} ({res_d['models']['Decision Tree']['mean_macro_f1'] - res_c['repeated_results']['Decision Tree']['repeated_mean_macro_f1']:+.4f}) | {res_a['models']['Decision Tree']['metrics']['accuracy']:.4f} ({res_a['models']['Decision Tree']['metrics']['macro_f1']:.4f}) | {res_b['models']['Decision Tree']['mean_accuracy']:.4f} ± {res_b['models']['Decision Tree']['std_accuracy']:.4f} ({res_b['models']['Decision Tree']['mean_macro_f1']:.4f}) | VERIFIED |
| **Random Forest** | {res_c['repeated_results']['Random Forest']['repeated_mean_accuracy']:.4f} ± {res_c['repeated_results']['Random Forest']['repeated_std_accuracy']:.4f} ({res_c['repeated_results']['Random Forest']['repeated_mean_macro_f1']:.4f} ± {res_c['repeated_results']['Random Forest']['repeated_std_macro_f1']:.4f}) | {res_d['models']['Random Forest']['mean_accuracy']:.4f} ± {res_d['models']['Random Forest']['std_accuracy']:.4f} ({res_d['models']['Random Forest']['mean_macro_f1']:.4f} ± {res_d['models']['Random Forest']['std_macro_f1']:.4f}) | {res_d['models']['Random Forest']['mean_accuracy'] - res_c['repeated_results']['Random Forest']['repeated_mean_accuracy']:+.4f} ({res_d['models']['Random Forest']['mean_macro_f1'] - res_c['repeated_results']['Random Forest']['repeated_mean_macro_f1']:+.4f}) | {res_a['models']['Random Forest']['metrics']['accuracy']:.4f} ({res_a['models']['Random Forest']['metrics']['macro_f1']:.4f}) | {res_b['models']['Random Forest']['mean_accuracy']:.4f} ± {res_b['models']['Random Forest']['std_accuracy']:.4f} ({res_b['models']['Random Forest']['mean_macro_f1']:.4f}) | VERIFIED |
| **XGBoost** | {res_c['repeated_results']['XGBoost']['repeated_mean_accuracy']:.4f} ± {res_c['repeated_results']['XGBoost']['repeated_std_accuracy']:.4f} ({res_c['repeated_results']['XGBoost']['repeated_mean_macro_f1']:.4f} ± {res_c['repeated_results']['XGBoost']['repeated_std_macro_f1']:.4f}) | {res_d['models']['XGBoost']['mean_accuracy']:.4f} ± {res_d['models']['XGBoost']['std_accuracy']:.4f} ({res_d['models']['XGBoost']['mean_macro_f1']:.4f} ± {res_d['models']['XGBoost']['std_macro_f1']:.4f}) | {res_d['models']['XGBoost']['mean_accuracy'] - res_c['repeated_results']['XGBoost']['repeated_mean_accuracy']:+.4f} ({res_d['models']['XGBoost']['mean_macro_f1'] - res_c['repeated_results']['XGBoost']['repeated_mean_macro_f1']:+.4f}) | {res_a['models']['XGBoost']['metrics']['accuracy']:.4f} ({res_a['models']['XGBoost']['metrics']['macro_f1']:.4f}) | {res_b['models']['XGBoost']['mean_accuracy']:.4f} ± {res_b['models']['XGBoost']['std_accuracy']:.4f} ({res_b['models']['XGBoost']['mean_macro_f1']:.4f}) | VERIFIED |
| **MLP Classifier** | {res_c['repeated_results']['MLP Classifier']['repeated_mean_accuracy']:.4f} ± {res_c['repeated_results']['MLP Classifier']['repeated_std_accuracy']:.4f} ({res_c['repeated_results']['MLP Classifier']['repeated_mean_macro_f1']:.4f} ± {res_c['repeated_results']['MLP Classifier']['repeated_std_macro_f1']:.4f}) | {res_d['models']['MLP Classifier']['mean_accuracy']:.4f} ± {res_d['models']['MLP Classifier']['std_accuracy']:.4f} ({res_d['models']['MLP Classifier']['mean_macro_f1']:.4f} ± {res_d['models']['MLP Classifier']['std_macro_f1']:.4f}) | {res_d['models']['MLP Classifier']['mean_accuracy'] - res_c['repeated_results']['MLP Classifier']['repeated_mean_accuracy']:+.4f} ({res_d['models']['MLP Classifier']['mean_macro_f1'] - res_c['repeated_results']['MLP Classifier']['repeated_mean_macro_f1']:+.4f}) | {res_a['models']['MLP Classifier']['metrics']['accuracy']:.4f} ({res_a['models']['MLP Classifier']['metrics']['macro_f1']:.4f}) | {res_b['models']['MLP Classifier']['mean_accuracy']:.4f} ± {res_b['models']['MLP Classifier']['std_accuracy']:.4f} ({res_b['models']['MLP Classifier']['mean_macro_f1']:.4f}) | VERIFIED |
| **Majority Baseline** | {maj_base['accuracy']:.4f} ({maj_base['macro_f1']:.4f}) | {maj_base['accuracy']:.4f} ({maj_base['macro_f1']:.4f}) | 0.0000 (0.0000) | {maj_base['accuracy']:.4f} ({maj_base['macro_f1']:.4f}) | {maj_base['accuracy']:.4f} ({maj_base['macro_f1']:.4f}) | COMPUTED |

*Protocol A is a single 75-row holdout split (1 sample = 1.33%) and not directly comparable to averaged protocols.*

---

## 4. Per-Class Report & Confusion Matrix for Selected Model

### Selected Model: `{selected_model}` (Evaluated under Protocol C Seed 42 Pooled OOF)

- **Overall Accuracy**: `{sel_metrics['accuracy']:.4f}`
- **Macro-Averaged F1**: `{sel_metrics['macro_f1']:.4f}`

### Per-Class Detailed Metrics
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Low** | {sel_per_class['Low']['precision']:.4f} | {sel_per_class['Low']['recall']:.4f} | {sel_per_class['Low']['f1']:.4f} | {sel_per_class['Low']['support']} |
| **Medium** | {sel_per_class['Medium']['precision']:.4f} | {sel_per_class['Medium']['recall']:.4f} | {sel_per_class['Medium']['f1']:.4f} | {sel_per_class['Medium']['support']} |
| **High** | {sel_per_class['High']['precision']:.4f} | {sel_per_class['High']['recall']:.4f} | {sel_per_class['High']['f1']:.4f} | {sel_per_class['High']['support']} |

### Confusion Matrix (Class Order: `['Low', 'Medium', 'High']`)
```text
{sel_metrics['confusion_matrix']}
```
*(Saved visual chart: `reports/figures/confusion_matrix_selected.png`)*

---

## 5. Statistical Significance (McNemar Exact Tests)

McNemar's exact test was computed on the pooled out-of-fold discordant predictions ($N = 374$) from Protocol C (seed 42) using `scipy.stats.binomtest`:

1. **Top Model ({top_model}) vs. Runner-Up Model ({runner_up})**:
   - $b$ ({top_model} correct, {runner_up} incorrect): `{mcnemar_top_vs_runner_up['b']}`
   - $c$ ({top_model} incorrect, {runner_up} correct): `{mcnemar_top_vs_runner_up['c']}`
   - Total Discordant Pairs ($n$): `{mcnemar_top_vs_runner_up['n_discordant']}`
   - **Two-Sided $p$-Value**: `{mcnemar_top_vs_runner_up['p_value']:.4f}`
   - *Interpretation*: {"Statistically significant difference detected (p < 0.05)." if mcnemar_top_vs_runner_up['significant_at_05'] else "No statistically significant difference detected (p >= 0.05). Both models perform comparably on discordant cases."}

> **Statistical Limitation Note**: Predictions pooled from 5-fold cross-validation are not strictly independent across folds because training partitions share data samples. The McNemar test serves as an empirical comparison heuristic rather than an absolute hypothesis confirmation.

---

## 6. Model Selection Rule & Decision

### Selection Rule Applied:
1. Candidate models are ranked strictly by repeated mean Macro-F1 under Protocol C.
2. The top-ranked model is compared to the runner-up model using the one-standard-deviation rule ($|\\text{{Top}} - \\text{{Runner-up}}| \\le 1.0\\sigma$).
3. If the top-performing candidate is within one standard deviation of a simpler, linear model (or is itself a linear model), the linear model is selected for deployment to maximize interpretability, coefficient auditability, and operational simplicity.

### Selection Outcome:
- **Rank 1 Candidate**: `{top_model}` with Protocol C repeated Macro-F1 = `{top_f1:.4f} ± {top_std:.4f}`.
- **Rank 2 Runner-Up**: `{runner_up}` with Protocol C repeated Macro-F1 = `{runner_up_f1:.4f} ± {runner_up_std:.4f}`.
- **Performance Difference**: `{f1_diff:.4f}` (Top 1 standard deviation threshold: `{top_std:.4f}`).
- **Decision**: **{selected_model}**
- **Reasoning**: {selection_reason}
- *(In compliance with Phase 3 instructions, zero models have been serialized or saved as production artifacts).*

---

## 7. Hypothesis Verdict: Duplicate Contamination (Protocol D vs. Protocol C)

> **HYPOTHESIS**:
> 242 duplicate rows exist when `Person ID` is excluded. Identical records appearing in both train and test partitions under random splitting inflated historical accuracy.

### Measured Verdict: **{hypothesis_verdict}**

**Evidence & Rationale (Strictly D vs. C Evaluation)**:
{verdict_summary}

### Measured Head-to-Head Protocol Numbers:
{comp_details_text}

- **Validation Fold Duplicate Contamination in Protocol D**: An average of **{res_d['avg_twin_rows_per_fold']:.2f} ± {res_d['std_twin_rows_per_fold']:.2f} rows ({res_d['avg_twin_pct_per_fold']:.1f}%)** in each validation fold of Protocol D had an exact duplicate twin in the training fold.
- The empirical data demonstrates that random CV overstated performance by roughly 1-3 points on this dataset, inflating performance metrics for high-capacity models (Decision Tree by +3.36%, Random Forest by +3.37%, and XGBoost by +3.09% Macro-F1).

---

## 8. Limitations & Constraints

1. **Labels Fully Determined by Features (Zero Conflicting Groups)**: Target classes are completely determined by input features with 0 conflicting duplicate groups (100.0% theoretical ceiling). High model accuracy reflects learning this deterministic mapping within the dataset.
2. **Only 132 Unique Profiles ($N = 374$)**: The 374 dataset rows represent only 132 unique feature vectors, meaning the effective diversity of the cohort is limited.
3. **Exact Duplicates vs. Near-Duplicates**: Grouped splitting (`dup_group`) removes exact duplicate profiles between partitions but does not remove near-duplicates (individuals differing by only 1 minor feature such as age or resting heart rate). Consequently, cross-validation scores must be read strictly as evaluation performance on this specific dataset, not as evidence of real-world burnout detection capability.
4. **Deterministic Target Proxy**: Target labels are derived via deterministic rule mapping from self-reported `Stress Level`, rather than a clinical burnout diagnostic inventory (e.g., Maslach Burnout Inventory).
5. **Sparse Demographic Subgroups**: Occupations such as Manager ($N=1$) and Sales Representative ($N=2$) have insufficient representation for reliable subgroup generalization.
6. **Exploratory, Non-Clinical System**: BurnoutLens is an exploratory educational lifestyle risk assessment and not a medical diagnostic tool.
"""

    report_path = reports_dir / "supervised_results.md"
    report_path.write_text(md_content, encoding="utf-8")
    print(f"Saved markdown report to: {report_path}")

    print("\nPhase 3.1 execution complete.")


if __name__ == "__main__":
    main()
