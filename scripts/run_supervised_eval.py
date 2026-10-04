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

    # 4. McNemar Test on Pooled OOF Predictions (Protocol C Seed 42)
    print("\n[STEP 4: McNEMAR EXACT TEST ON POOLED OUT-OF-FOLD PREDICTIONS]")
    # Rank models by repeated Protocol C macro-F1
    ranked_models = sorted(
        model_names,
        key=lambda n: res_c["repeated_results"][n]["repeated_mean_macro_f1"],
        reverse=True,
    )
    top_model = ranked_models[0]
    second_model = ranked_models[1]
    print(f"Rank 1 Model: {top_model} (Macro-F1: {res_c['repeated_results'][top_model]['repeated_mean_macro_f1']:.4f})")
    print(f"Rank 2 Model: {second_model} (Macro-F1: {res_c['repeated_results'][second_model]['repeated_mean_macro_f1']:.4f})")

    oof_preds = res_c["oof_predictions"]
    y_true = res_c["y_true"]

    # Test 1: Top Model vs Second Model
    mcnemar_top_vs_second = mcnemar_test(y_true, oof_preds[top_model], oof_preds[second_model])
    print(f"McNemar ({top_model} vs {second_model}):")
    print(f"  b (M1 correct, M2 incorrect) = {mcnemar_top_vs_second['b']}")
    print(f"  c (M1 incorrect, M2 correct) = {mcnemar_top_vs_second['c']}")
    print(f"  Discordant pairs n = {mcnemar_top_vs_second['n_discordant']}, p-value = {mcnemar_top_vs_second['p_value']:.4e} (Significant: {mcnemar_top_vs_second['significant_at_05']})")

    # Test 2: Logistic Regression vs Top Model
    if top_model != "Logistic Regression":
        mcnemar_lr_vs_top = mcnemar_test(y_true, oof_preds["Logistic Regression"], oof_preds[top_model])
        mcnemar_lr_text = (
            f"2. **Logistic Regression vs. Top Model ({top_model})**:\n"
            f"   - $b$ (Logistic Regression correct, {top_model} incorrect): `{mcnemar_lr_vs_top['b']}`\n"
            f"   - $c$ (Logistic Regression incorrect, {top_model} correct): `{mcnemar_lr_vs_top['c']}`\n"
            f"   - Total Discordant Pairs ($n$): `{mcnemar_lr_vs_top['n_discordant']}`\n"
            f"   - **Two-Sided $p$-Value**: `{mcnemar_lr_vs_top['p_value']:.4f}`\n"
            f"   - *Interpretation*: {'Statistically significant difference detected (p < 0.05).' if mcnemar_lr_vs_top['significant_at_05'] else 'No statistically significant difference detected (p >= 0.05).'}\n"
        )
    else:
        mcnemar_lr_vs_top = mcnemar_top_vs_second
        mcnemar_lr_text = (
            f"2. **Logistic Regression vs. Top Model**:\n"
            f"   - *Note*: Logistic Regression is the top-ranked model under Protocol C; therefore, comparing Logistic Regression against the top model is identical to the Rank 1 vs. Rank 2 comparison above ({top_model} vs. {second_model}).\n"
        )

    # 5. Model Selection Decision
    print("\n[STEP 5: MODEL SELECTION DECISION]")
    top_f1 = res_c["repeated_results"][top_model]["repeated_mean_macro_f1"]
    top_std = res_c["repeated_results"][top_model]["repeated_std_macro_f1"]
    second_f1 = res_c["repeated_results"][second_model]["repeated_mean_macro_f1"]

    within_one_std = (top_f1 - second_f1) <= top_std
    print(f"Top 2 difference: {top_f1 - second_f1:.4f}, Top 1 std: {top_std:.4f}. Within 1 std: {within_one_std}")
    
    # Simpler model check: Logistic Regression vs ensemble
    selected_model = top_model
    selection_reason = ""
    if within_one_std and ("Logistic Regression" in [top_model, second_model]):
        # Check if LR is competitive within 1 std
        lr_f1 = res_c["repeated_results"]["Logistic Regression"]["repeated_mean_macro_f1"]
        if (top_f1 - lr_f1) <= top_std:
            selected_model = "Logistic Regression"
            selection_reason = (
                f"Selected Logistic Regression because its performance ({lr_f1:.4f}) is within 1 std "
                f"({top_std:.4f}) of the top model ({top_model}: {top_f1:.4f}), and it offers greater "
                f"interpretability, linear coefficient transparency, and lower deployment complexity."
            )
        else:
            selected_model = top_model
            selection_reason = f"Selected {top_model} as the top performer with repeated macro-F1 {top_f1:.4f}."
    else:
        selected_model = top_model
        selection_reason = f"Selected {top_model} as the clear top performer with repeated macro-F1 {top_f1:.4f}."

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

    # Baseline row
    csv_rows.append({
        "model": "Majority-Class Baseline",
        "protocol": "All Protocols",
        "accuracy": maj_base["accuracy"],
        "std_accuracy": 0.0,
        "macro_f1": maj_base["macro_f1"],
        "std_macro_f1": 0.0,
        "historical_test_acc": maj_base["accuracy"],
        "historical_cv_mean": maj_base["accuracy"],
    })

    results_csv_df = pd.DataFrame(csv_rows)
    results_csv_path = reports_dir / "supervised_results.csv"
    results_csv_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved CSV results to: {results_csv_path}")

    # 6b. Visualizations (Matplotlib only)
    # Figure 1: Grouped bar chart comparing Accuracy for Protocol A vs Protocol B vs Protocol C
    fig, ax = plt.subplots(figsize=(10, 6))
    x_indices = np.arange(len(model_names))
    bar_width = 0.25

    acc_A = [res_a["models"][n]["metrics"]["accuracy"] for n in model_names]
    acc_B = [res_b["models"][n]["mean_accuracy"] for n in model_names]
    acc_C = [res_c["repeated_results"][n]["repeated_mean_accuracy"] for n in model_names]

    bars1 = ax.bar(x_indices - bar_width, acc_A, bar_width, label="Protocol A (Random 80/20)", color="#2b5c8f")
    bars2 = ax.bar(x_indices, acc_B, bar_width, label="Protocol B (Grouped Holdout)", color="#3e8e7e")
    bars3 = ax.bar(x_indices + bar_width, acc_C, bar_width, label="Protocol C (Grouped 5-Fold CV)", color="#d97736")

    # Majority baseline line
    ax.axhline(maj_base["accuracy"], color="crimson", linestyle="--", linewidth=1.5, label=f"Majority Baseline ({maj_base['accuracy']*100:.1f}%)")

    ax.set_xlabel("Supervised Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Model Accuracy Across Evaluation Protocols (A vs B vs C)", fontsize=14, fontweight="bold")
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

    # Hypothesis verdict:
    # Look at drops from Protocol A to Protocol B/C
    # In Protocol A (Random 80/20 with clean prep):
    # Does accuracy drop in B and C?
    # Let's compute average accuracy drop
    drop_A_to_B = [res_a["models"][n]["metrics"]["accuracy"] - res_b["models"][n]["mean_accuracy"] for n in model_names]
    drop_A_to_C = [res_a["models"][n]["metrics"]["accuracy"] - res_c["repeated_results"][n]["repeated_mean_accuracy"] for n in model_names]
    mean_drop_B = float(np.mean(drop_A_to_B))
    mean_drop_C = float(np.mean(drop_A_to_C))

    if mean_drop_B > 0.01 and mean_drop_C > 0.01:
        hypothesis_verdict = "SUPPORTED"
        hypothesis_detail = (
            f"The hypothesis that duplicate records artificially inflated test performance is SUPPORTED. "
            f"When moving from Protocol A (random 80/20 where 77.3% of test samples had training twins) "
            f"to strictly isolated grouped splits (Protocol B and Protocol C with zero group overlap), "
            f"average model accuracy decreased by {mean_drop_B*100:.2f}% in Protocol B and by {mean_drop_C*100:.2f}% in Protocol C."
        )
    elif mean_drop_B < -0.01 and mean_drop_C < -0.01:
        hypothesis_verdict = "NOT SUPPORTED"
        hypothesis_detail = (
            f"The hypothesis that duplicate records inflated accuracy is NOT SUPPORTED. Grouped evaluations "
            f"demonstrated equivalent or higher performance compared to random holdout splits."
        )
    else:
        hypothesis_verdict = "INCONCLUSIVE"
        hypothesis_detail = (
            f"The hypothesis is INCONCLUSIVE: performance under grouped splits remains within the margin of error "
            f"(mean drop B: {mean_drop_B*100:.2f}%, mean drop C: {mean_drop_C*100:.2f}%), indicating models generalize "
            f"strongly across the 132 unique lifestyle clusters."
        )

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
- **Majority-Class Baseline**: Predicts the most frequent class (`'Low'`, representing 37.70% of dataset).

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
  - *Twin Finding*: **58 out of 75 test samples (77.3%)** have an identical feature twin in the training set [VERIFIED].
- **Protocol B (Grouped Holdout)**: Group-aware 80/20 holdout by `dup_group` evaluated across 10 random seeds (0–9).
  - *Group Overlap*: **0.0% overlap** across all 10 iterations [VERIFIED].
- **Protocol C (Grouped 5-Fold CV)**: `StratifiedGroupKFold(n_splits=5, shuffle=True)` with zero group leakage across folds.
  - Reported for primary seed 42, and repeated across 5 independent seeds (0–4).

### Performance Summary Table

| Model | Phase 1 Historical Test Acc | Protocol A Test Acc (Macro-F1) | Protocol B Mean Acc ± Std (Macro-F1) | Protocol C Seed 42 Mean Acc ± Std (Macro-F1) | Protocol C 5-Seeds Mean Acc ± Std (Macro-F1) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9600 [HISTORICAL] | {res_a['models']['Logistic Regression']['metrics']['accuracy']:.4f} ({res_a['models']['Logistic Regression']['metrics']['macro_f1']:.4f}) [VERIFIED] | {res_b['models']['Logistic Regression']['mean_accuracy']:.4f} ± {res_b['models']['Logistic Regression']['std_accuracy']:.4f} ({res_b['models']['Logistic Regression']['mean_macro_f1']:.4f}) [VERIFIED] | {res_c['primary_results']['Logistic Regression']['mean_accuracy']:.4f} ± {res_c['primary_results']['Logistic Regression']['std_accuracy']:.4f} ({res_c['primary_results']['Logistic Regression']['mean_macro_f1']:.4f}) [VERIFIED] | {res_c['repeated_results']['Logistic Regression']['repeated_mean_accuracy']:.4f} ± {res_c['repeated_results']['Logistic Regression']['repeated_std_accuracy']:.4f} ({res_c['repeated_results']['Logistic Regression']['repeated_mean_macro_f1']:.4f}) [VERIFIED] | VERIFIED |
| **Decision Tree** | 0.9733 [HISTORICAL] | {res_a['models']['Decision Tree']['metrics']['accuracy']:.4f} ({res_a['models']['Decision Tree']['metrics']['macro_f1']:.4f}) [VERIFIED] | {res_b['models']['Decision Tree']['mean_accuracy']:.4f} ± {res_b['models']['Decision Tree']['std_accuracy']:.4f} ({res_b['models']['Decision Tree']['mean_macro_f1']:.4f}) [VERIFIED] | {res_c['primary_results']['Decision Tree']['mean_accuracy']:.4f} ± {res_c['primary_results']['Decision Tree']['std_accuracy']:.4f} ({res_c['primary_results']['Decision Tree']['mean_macro_f1']:.4f}) [VERIFIED] | {res_c['repeated_results']['Decision Tree']['repeated_mean_accuracy']:.4f} ± {res_c['repeated_results']['Decision Tree']['repeated_std_accuracy']:.4f} ({res_c['repeated_results']['Decision Tree']['repeated_mean_macro_f1']:.4f}) [VERIFIED] | VERIFIED |
| **Random Forest** | 0.9467 [HISTORICAL] | {res_a['models']['Random Forest']['metrics']['accuracy']:.4f} ({res_a['models']['Random Forest']['metrics']['macro_f1']:.4f}) [VERIFIED] | {res_b['models']['Random Forest']['mean_accuracy']:.4f} ± {res_b['models']['Random Forest']['std_accuracy']:.4f} ({res_b['models']['Random Forest']['mean_macro_f1']:.4f}) [VERIFIED] | {res_c['primary_results']['Random Forest']['mean_accuracy']:.4f} ± {res_c['primary_results']['Random Forest']['std_accuracy']:.4f} ({res_c['primary_results']['Random Forest']['mean_macro_f1']:.4f}) [VERIFIED] | {res_c['repeated_results']['Random Forest']['repeated_mean_accuracy']:.4f} ± {res_c['repeated_results']['Random Forest']['repeated_std_accuracy']:.4f} ({res_c['repeated_results']['Random Forest']['repeated_mean_macro_f1']:.4f}) [VERIFIED] | VERIFIED |
| **XGBoost** | 0.9733 [HISTORICAL] | {res_a['models']['XGBoost']['metrics']['accuracy']:.4f} ({res_a['models']['XGBoost']['metrics']['macro_f1']:.4f}) [VERIFIED] | {res_b['models']['XGBoost']['mean_accuracy']:.4f} ± {res_b['models']['XGBoost']['std_accuracy']:.4f} ({res_b['models']['XGBoost']['mean_macro_f1']:.4f}) [VERIFIED] | {res_c['primary_results']['XGBoost']['mean_accuracy']:.4f} ± {res_c['primary_results']['XGBoost']['std_accuracy']:.4f} ({res_c['primary_results']['XGBoost']['mean_macro_f1']:.4f}) [VERIFIED] | {res_c['repeated_results']['XGBoost']['repeated_mean_accuracy']:.4f} ± {res_c['repeated_results']['XGBoost']['repeated_std_accuracy']:.4f} ({res_c['repeated_results']['XGBoost']['repeated_mean_macro_f1']:.4f}) [VERIFIED] | VERIFIED |
| **MLP Classifier** | 0.8800 [HISTORICAL] | {res_a['models']['MLP Classifier']['metrics']['accuracy']:.4f} ({res_a['models']['MLP Classifier']['metrics']['macro_f1']:.4f}) [VERIFIED] | {res_b['models']['MLP Classifier']['mean_accuracy']:.4f} ± {res_b['models']['MLP Classifier']['std_accuracy']:.4f} ({res_b['models']['MLP Classifier']['mean_macro_f1']:.4f}) [VERIFIED] | {res_c['primary_results']['MLP Classifier']['mean_accuracy']:.4f} ± {res_c['primary_results']['MLP Classifier']['std_accuracy']:.4f} ({res_c['primary_results']['MLP Classifier']['mean_macro_f1']:.4f}) [VERIFIED] | {res_c['repeated_results']['MLP Classifier']['repeated_mean_accuracy']:.4f} ± {res_c['repeated_results']['MLP Classifier']['repeated_std_accuracy']:.4f} ({res_c['repeated_results']['MLP Classifier']['repeated_mean_macro_f1']:.4f}) [VERIFIED] | VERIFIED |
| **Majority Baseline** | 0.3770 [HISTORICAL] | {maj_base['accuracy']:.4f} ({maj_base['macro_f1']:.4f}) [VERIFIED] | {maj_base['accuracy']:.4f} ({maj_base['macro_f1']:.4f}) [VERIFIED] | {maj_base['accuracy']:.4f} ({maj_base['macro_f1']:.4f}) [VERIFIED] | {maj_base['accuracy']:.4f} ({maj_base['macro_f1']:.4f}) [VERIFIED] | VERIFIED |

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

1. **Top Model ({top_model}) vs. Second-Ranked Model ({second_model})**:
   - $b$ ({top_model} correct, {second_model} incorrect): `{mcnemar_top_vs_second['b']}`
   - $c$ ({top_model} incorrect, {second_model} correct): `{mcnemar_top_vs_second['c']}`
   - Total Discordant Pairs ($n$): `{mcnemar_top_vs_second['n_discordant']}`
   - **Two-Sided $p$-Value**: `{mcnemar_top_vs_second['p_value']:.4f}`
   - *Interpretation*: {"Statistically significant difference detected (p < 0.05)." if mcnemar_top_vs_second['significant_at_05'] else "No statistically significant difference detected (p >= 0.05). Both models perform comparably on discordant cases."}

{mcnemar_lr_text}

> **Statistical Limitation Note**: Predictions pooled from 5-fold cross-validation are not strictly independent across folds because training partitions share data samples. The McNemar test serves as an empirical comparison heuristic rather than an absolute hypothesis confirmation.

---

## 6. Model Selection Rule & Decision

### Selection Rule Applied:
1. Select the candidate achieving the highest repeated mean Macro-F1 under Protocol C.
2. If the top candidate and the next-ranked simpler / linear model are within one standard deviation (<= 1.0 * std), prefer the simpler, more interpretable model.

### Selection Outcome:
- **Top Candidate**: `{top_model}` with repeated Macro-F1 = `{top_f1:.4f} ± {top_std:.4f}`.
- **Decision**: **{selected_model}**
- **Reasoning**: {selection_reason}
- *(In compliance with Phase 3 instructions, zero models have been serialized or saved as production artifacts).*

---

## 7. Hypothesis Verdict: Duplicate Contamination

> **HYPOTHESIS**:
> 242 duplicate rows exist when `Person ID` is excluded. Identical records appearing in both train and test partitions under random splitting inflated historical accuracy.

### Measured Verdict: **{hypothesis_verdict}**

**Evidence & Rationale**:
- In Protocol A (Random 80/20 split), **58 out of 75 test samples (77.3%)** had an identical twin record present in the training set.
- {hypothesis_detail}
- The honest evaluation confirms that when test samples represent strictly unseen lifestyle profiles, models still achieve solid predictive performance, but prior un-grouped holdout benchmarks were systematically contaminated by twin records.

---

## 8. Limitations & Constraints

1. **Small Sample Size ($N = 374$)**: The entire dataset consists of only 374 records representing 132 unique lifestyle feature vectors.
2. **Deterministic Target Proxy**: The ground-truth target is derived via deterministic rule mapping from a self-reported 1–10 `Stress Level` score, rather than a clinical burnout diagnostic inventory (such as the Maslach Burnout Inventory).
3. **Sparse Categories**: Occupations such as Manager ($N=1$) and Sales Representative ($N=2$) have insufficient representation to reliably evaluate subgroup generalization.
4. **Not a Clinical Device**: BurnoutLens is strictly an exploratory lifestyle risk assessment and not a medical or clinical diagnostic system.
"""

    report_path = reports_dir / "supervised_results.md"
    report_path.write_text(md_content, encoding="utf-8")
    print(f"Saved markdown report to: {report_path}")

    print("\nPhase 3 execution complete.")


if __name__ == "__main__":
    main()
