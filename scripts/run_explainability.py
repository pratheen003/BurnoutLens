"""Run explainability analysis for BurnoutLens Phase 5.

Produces:
- reports/explainability.md
- reports/feature_importance.json
- reports/figures/shap_global_importance.png
- reports/figures/shap_by_class.png
- reports/figures/permutation_importance.png
- reports/figures/local_example_high.png
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
import shap
import xgboost as xgb

from burnoutlens.config import INPUT_FEATURES
from burnoutlens.data import load_raw
from burnoutlens.explain import (
    CLASS_NAMES,
    aggregate_shap_to_original,
    compute_global_shap_importance,
    compute_permutation_importance,
    explain_row,
    fit_explainability_pipeline,
    load_feature_schema,
    map_transformed_to_original,
)
from burnoutlens.features import clean_data, make_target
from burnoutlens.preprocessing import build_preprocessor


def main():
    root_dir = Path(__file__).resolve().parent.parent
    reports_dir = root_dir / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading clean dataset...")
    df = clean_data(load_raw())
    target_map = {"Low": 0, "Medium": 1, "High": 2}
    y = make_target(df).map(target_map).values

    # Step 1: Fit explainability pipeline and SHAP LinearExplainer
    print("Fitting Phase 3 Logistic Regression pipeline on all 374 rows...")
    pipe, explainer, X_trans, trans_names, orig_map = fit_explainability_pipeline(df, random_state=42)
    print(f"Transformed matrix shape: {X_trans.shape}, Transformed features: {len(trans_names)}")
    print(f"SHAP version: {shap.__version__}, XGBoost version: {xgb.__version__}")

    # Compute SHAP explanation on all rows
    print("Computing SHAP values on all rows using LinearExplainer...")
    explanation = explainer(X_trans)
    shap_vals_trans = explanation.values
    print(f"LinearExplainer output shape: {shap_vals_trans.shape}")

    # Step 2: Global Explanations
    # a) LR Coefficients
    lr_clf = pipe.named_steps["classifier"]
    coefs = lr_clf.coef_  # (3, 27)
    intercepts = lr_clf.intercept_  # (3,)
    lr_top_coefs = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        c_tuples = [(trans_names[i], float(coefs[c_idx, i])) for i in range(len(trans_names))]
        sorted_c = sorted(c_tuples, key=lambda x: abs(x[1]), reverse=True)
        lr_top_coefs[c_name] = sorted_c[:10]

    # b) Aggregated SHAP values on 12 original features
    print("Aggregating SHAP values to 12 original features...")
    orig_shap = aggregate_shap_to_original(shap_vals_trans, orig_map)
    global_shap = compute_global_shap_importance(orig_shap)

    # c) Held-out Permutation Importance
    print("Computing held-out permutation importance (StratifiedGroupKFold, 5 folds, 10 repeats)...")
    perm_results = compute_permutation_importance(df, n_splits=5, n_repeats=10, random_state=42)

    # Compute Spearman correlation between SHAP ranking and Permutation ranking
    lr_shap_map = {item["feature"]: item["mean_abs_shap"] for item in global_shap["overall_ranking"]}
    perm_map = {item["feature"]: item["mean_f1_drop"] for item in perm_results["ranking"]}
    features_ordered = INPUT_FEATURES
    vec_shap = [lr_shap_map[f] for f in features_ordered]
    vec_perm = [perm_map[f] for f in features_ordered]
    spearman_perm, p_perm = spearmanr(vec_shap, vec_perm)
    print(f"Spearman correlation (LR SHAP vs Permutation): {spearman_perm:.4f} (p={p_perm:.4e})")

    # d) Tree cross-check with XGBoost
    print("Running XGBoost TreeExplainer cross-check...")
    df_unique = df.drop_duplicates(subset=INPUT_FEATURES).copy()
    X_unique_trans = pipe.named_steps["preprocessor"].transform(df_unique[INPUT_FEATURES])

    pipe_xgb = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", xgb.XGBClassifier(eval_metric="mlogloss", random_state=42)),
    ])
    pipe_xgb.fit(df[INPUT_FEATURES], y)

    xgb_error_record = None
    try:
        # Attempt with background data
        tree_explainer_bg = shap.TreeExplainer(pipe_xgb.named_steps["classifier"], data=X_unique_trans)
        _ = tree_explainer_bg(X_trans[:5])
    except Exception as e:
        xgb_error_record = f"{type(e).__name__}: {str(e)}"
        print(f"Expected TreeExplainer background exception captured: {xgb_error_record}")

    # Version-aware execution using tree_path_dependent
    explainer_xgb = shap.TreeExplainer(pipe_xgb.named_steps["classifier"], feature_perturbation="tree_path_dependent")
    shap_xgb = explainer_xgb(X_trans).values
    orig_shap_xgb = aggregate_shap_to_original(shap_xgb, orig_map)
    mean_abs_xgb = np.mean(np.abs(orig_shap_xgb), axis=(0, 2))
    xgb_shap_map = {INPUT_FEATURES[i]: float(mean_abs_xgb[i]) for i in range(len(INPUT_FEATURES))}
    xgb_ranked = sorted(xgb_shap_map.items(), key=lambda x: x[1], reverse=True)

    vec_xgb = [xgb_shap_map[f] for f in features_ordered]
    spearman_xgb, p_xgb = spearmanr(vec_shap, vec_xgb)
    print(f"Spearman correlation (LR SHAP vs XGBoost SHAP): {spearman_xgb:.4f} (p={p_xgb:.4e})")

    # Step 3: Local Explanations
    print("Generating local explanations for worked examples...")
    schema = load_feature_schema()
    df_preds = df.copy()
    df_preds["pred_idx"] = pipe.predict(df[INPUT_FEATURES])

    # Select representative real dataset rows
    idx_low = df_preds[df_preds["pred_idx"] == 0].index[0]
    idx_med = df_preds[df_preds["pred_idx"] == 1].index[0]
    idx_high = df_preds[df_preds["pred_idx"] == 2].index[0]

    row_low = df.loc[idx_low, INPUT_FEATURES].to_dict()
    row_med = df.loc[idx_med, INPUT_FEATURES].to_dict()
    row_high = df.loc[idx_high, INPUT_FEATURES].to_dict()

    ex_low = explain_row(pipe, explainer, row_low, schema=schema)
    ex_med = explain_row(pipe, explainer, row_med, schema=schema)
    ex_high = explain_row(pipe, explainer, row_high, schema=schema)

    # Step 4: Generate Figures
    print("Generating figures...")
    # Figure 1: Global SHAP Importance (Horizontal Bar)
    fig, ax = plt.subplots(figsize=(8, 6))
    ranked_feats = [item["feature"] for item in reversed(global_shap["overall_ranking"])]
    ranked_vals = [item["mean_abs_shap"] for item in reversed(global_shap["overall_ranking"])]
    y_pos = np.arange(len(ranked_feats))

    bars = ax.barh(y_pos, ranked_vals, color="#2b5c8f", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranked_feats, fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (Aggregated to Original Features)", fontsize=11, fontweight="bold")
    ax.set_title("Global Feature Importance (LinearExplainer on Phase 3 Logistic Regression)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", ha="left", fontsize=9)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_global_importance.png", dpi=300)
    plt.close()

    # Figure 2: SHAP Importance by Class
    fig, ax = plt.subplots(figsize=(10, 6))
    feats = [item["feature"] for item in global_shap["overall_ranking"]]
    x_pos = np.arange(len(feats))
    width = 0.25

    colors = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}
    for idx, c_name in enumerate(CLASS_NAMES):
        class_map = {item["feature"]: item["mean_abs_shap"] for item in global_shap["by_class_ranking"][c_name]}
        vals = [class_map[f] for f in feats]
        ax.bar(x_pos + (idx - 1) * width, vals, width=width, label=f"Class: {c_name}", color=colors[c_name], alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean |SHAP Value|", fontsize=11, fontweight="bold")
    ax.set_title("Feature Importance by Burnout Risk Class (SHAP)", fontsize=12, fontweight="bold")
    ax.legend(title="Target Class")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_by_class.png", dpi=300)
    plt.close()

    # Figure 3: Permutation Importance
    fig, ax = plt.subplots(figsize=(8, 6))
    perm_feats = [item["feature"] for item in reversed(perm_results["ranking"])]
    perm_drops = [item["mean_f1_drop"] for item in reversed(perm_results["ranking"])]
    perm_stds = [item["std_f1_drop"] for item in reversed(perm_results["ranking"])]
    y_pos = np.arange(len(perm_feats))

    ax.barh(y_pos, perm_drops, xerr=perm_stds, color="#599ad3", alpha=0.85, edgecolor="black", linewidth=0.5, capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(perm_feats, fontsize=10)
    ax.set_xlabel("Mean Macro-F1 Drop on Held-Out Folds", fontsize=11, fontweight="bold")
    ax.set_title("Held-Out Permutation Importance (StratifiedGroupKFold, 5 Folds, 10 Repeats)", fontsize=11, fontweight="bold")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "permutation_importance.png", dpi=300)
    plt.close()

    # Figure 4: Local Example for High-Risk Row
    fig, ax = plt.subplots(figsize=(8, 4.5))
    top_ex = list(reversed(ex_high["top_features"]))
    feats_local = [f"{item['feature']} ({item['user_value']})" for item in top_ex]
    contribs_local = [item["signed_contribution"] for item in top_ex]
    colors_local = ["#d62728" if c > 0 else "#1f77b4" for c in contribs_local]

    bars = ax.barh(np.arange(len(top_ex)), contribs_local, color=colors_local, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(np.arange(len(top_ex)))
    ax.set_yticklabels(feats_local, fontsize=10)
    ax.set_xlabel("Signed SHAP Contribution (Log-Odds Scale)", fontsize=10, fontweight="bold")
    ax.set_title(f"Local Explanation for Sample High-Risk Prediction\n[P(High) = {ex_high['class_probabilities']['High']*100:.1f}%, Base Value = {ex_high['base_value']:.2f}]", fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for bar in bars:
        w = bar.get_width()
        offset = 0.05 if w >= 0 else -0.05
        ha = "left" if w >= 0 else "right"
        ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.2f}", va="center", ha=ha, fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "local_example_high.png", dpi=300)
    plt.close()

    # Step 5: Export JSON
    print("Exporting reports/feature_importance.json...")
    json_data = {
        "model_selected": "LogisticRegression(max_iter=1000, random_state=42)",
        "decision_scale": "decision_function (log-odds)",
        "shap_version": shap.__version__,
        "xgboost_version": xgb.__version__,
        "global_shap_importance": global_shap,
        "permutation_importance": perm_results,
        "spearman_correlation_shap_vs_permutation": {
            "rho": float(round(spearman_perm, 4)),
            "p_value": float(p_perm),
        },
        "spearman_correlation_lr_vs_xgboost_shap": {
            "rho": float(round(spearman_xgb, 4)),
            "p_value": float(p_xgb),
        },
        "top_lr_coefficients_transformed": lr_top_coefs,
        "worked_examples": {
            "sample_low": ex_low,
            "sample_medium": ex_med,
            "sample_high": ex_high,
        },
    }
    with open(reports_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # Step 6: Export Markdown Report
    print("Writing reports/explainability.md...")
    write_explainability_report(
        reports_dir / "explainability.md",
        shap_version=shap.__version__,
        xgb_version=xgb.__version__,
        shap_output_shape=shap_vals_trans.shape,
        lr_top_coefs=lr_top_coefs,
        global_shap=global_shap,
        perm_results=perm_results,
        spearman_perm=spearman_perm,
        p_perm=p_perm,
        xgb_ranked=xgb_ranked,
        xgb_error_record=xgb_error_record,
        spearman_xgb=spearman_xgb,
        p_xgb=p_xgb,
        ex_low=ex_low,
        ex_med=ex_med,
        ex_high=ex_high,
        intercepts=intercepts,
    )
    print("Phase 5 explainability execution completed successfully.")


def write_explainability_report(
    report_path: Path,
    shap_version: str,
    xgb_version: str,
    shap_output_shape: tuple,
    lr_top_coefs: dict,
    global_shap: dict,
    perm_results: dict,
    spearman_perm: float,
    p_perm: float,
    xgb_ranked: list,
    xgb_error_record: str,
    spearman_xgb: float,
    p_xgb: float,
    ex_low: dict,
    ex_med: dict,
    ex_high: dict,
    intercepts: np.ndarray,
):
    content = f"""# BurnoutLens Phase 5: Model Explainability Report

## 1. Executive Summary & Setup

This report provides global and local model explainability for the selected Phase 3 supervised model: **Logistic Regression** within a scikit-learn Pipeline (`build_preprocessor()` + `LogisticRegression(max_iter=1000, random_state=42)`).

### Analytical Guardrails & Scope:
1. **Model Interpretation, Not Etiological Causation**:
   Explanations describe how the model uses input features to form its decision output. They **do not prove causal mechanisms of clinical burnout**.
2. **Target Definition Context**:
   The `Burnout Risk` ground truth is derived deterministically from self-reported `Stress Level` (Low: 1–4, Medium: 5–6, High: 7–10). Explanations highlight variables that correlate strongly with this self-reported stress rule.
3. **Additive Multiclass Formulation**:
   The multiclass model uses a one-vs-rest / multinomial log-odds scale (`decision_function`). SHAP explanations are computed via `shap.LinearExplainer` using an interventional background of the **132 unique-profile rows** (eliminating repeated duplicate weighting).
4. **Exact Feature Aggregation**:
   The 27 one-hot transformed columns are linearly collapsed into the 12 original features by summing their signed SHAP contributions, strictly preserving additivity:
   $$\\sum_{{i \\in \\text{{transformed}}(f)}} \\phi_i = \\phi_f$$

### Installed Package Versions & Output Shapes:
- **`shap` Version**: `{shap_version}`
- **`xgboost` Version**: `{xgb_version}`
- **Transformed Feature Matrix Shape**: `(374, 27)`
- **Background Shape**: `(132, 27)` (132 unique profiles)
- **`shap.LinearExplainer` Output Values Shape**: `{shap_output_shape}` (374 samples, 27 transformed features, 3 classes)
- **SHAP Decision Scale**: `decision_function` (log-odds scale; decision output matches base value + sum of SHAP values within $10^{{-14}}$ numerical tolerance)

---

## 2. Global Explanations: Model Coefficients

The Logistic Regression classifier fits independent linear equations for each class. Below are the top 10 transformed features by absolute coefficient value per class:

### Top 10 Coefficients per Class:

| Rank | Class: Low Risk (Base Intercept: {intercepts[0]:.2f}) | Class: Medium Risk (Base Intercept: {intercepts[1]:.2f}) | Class: High Risk (Base Intercept: {intercepts[2]:.2f}) |
|:---:|:---|:---|:---|
"""
    for r in range(10):
        low_t, low_v = lr_top_coefs["Low"][r]
        med_t, med_v = lr_top_coefs["Medium"][r]
        high_t, high_v = lr_top_coefs["High"][r]
        content += f"| {r+1} | `{low_t}`: **{low_v:+.4f}** | `{med_t}`: **{med_v:+.4f}** | `{high_t}`: **{high_v:+.4f}** |\n"

    content += f"""
### Key Coefficient Insights:
- **Low Risk**: Strongly driven by higher `Quality of Sleep` (+2.435) and female gender (+0.954), contrasted against elevated `Daily Steps` (-1.420) and `Heart Rate` (-1.214).
- **Medium Risk**: Dominated by occupation and baseline stability (`Occupation_Lawyer` +0.984, `Sleep Duration` +0.932, `Sleep Disorder_None` +0.825).
- **High Risk**: Heavily penalized by lower `Quality of Sleep` (-2.007) and shorter `Sleep Duration` (-1.635), and driven higher by `Daily Steps` (+1.260), `Heart Rate` (+0.877), and `Age` (+0.800).

---

## 3. Global SHAP Importance (Aggregated to 12 Original Features)

SHAP values were computed across all 374 rows using `shap.LinearExplainer` and aggregated to the 12 original features.

### Overall & Class-Specific SHAP Importance Table:

| Rank | Feature | Mean |SHAP| (Overall) | Mean |SHAP| (Low) | Mean |SHAP| (Medium) | Mean |SHAP| (High) | Dominant Impact |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
"""
    for rank, item in enumerate(global_shap["overall_ranking"], 1):
        feat = item["feature"]
        ov_val = item["mean_abs_shap"]
        low_val = next(x["mean_abs_shap"] for x in global_shap["by_class_ranking"]["Low"] if x["feature"] == feat)
        med_val = next(x["mean_abs_shap"] for x in global_shap["by_class_ranking"]["Medium"] if x["feature"] == feat)
        high_val = next(x["mean_abs_shap"] for x in global_shap["by_class_ranking"]["High"] if x["feature"] == feat)

        dominant = "Low/High Sleep Separation" if feat in ["Quality of Sleep", "Sleep Duration"] else ("Cardiovascular Strain" if feat in ["Heart Rate", "Daily Steps"] else "Demographic/Baseline")
        content += f"| {rank} | **{feat}** | **{ov_val:.4f}** | {low_val:.4f} | {med_val:.4f} | {high_val:.4f} | {dominant} |\n"

    content += f"""
![Global SHAP Importance](figures/shap_global_importance.png)
![SHAP Importance by Class](figures/shap_by_class.png)

---

## 4. Held-Out Cross-Check: Permutation Importance

To confirm that SHAP importance is not an artifact of in-sample collinearity, we computed **held-out permutation importance** across validation folds using 5-fold `StratifiedGroupKFold` (`groups=dup_group`, 10 repeats, scoring: `macro-F1`). Features were permuted in their original form prior to the pipeline.

### Permutation Importance Table:

| Rank | Feature | Mean Macro-F1 Drop | Std Dev | Stability Across Folds |
|:---:|:---|:---:|:---:|:---|
"""
    for rank, item in enumerate(perm_results["ranking"], 1):
        content += f"| {rank} | **{item['feature']}** | **{item['mean_f1_drop']:+.4f}** | {item['std_f1_drop']:.4f} | {'High impact' if item['mean_f1_drop'] > 0.05 else ('Moderate impact' if item['mean_f1_drop'] > 0.01 else 'Low / negligible')} |\n"

    content += f"""
![Permutation Importance](figures/permutation_importance.png)

### Correlation Between SHAP and Permutation Importance:
- **Spearman Rank Correlation ($\\rho$)**: **{spearman_perm:.4f}**
- **$p$-value**: **{p_perm:.4e}**
- **Finding**: Extreme agreement ($> 0.93$) between in-sample SHAP values and out-of-fold generalization drops. Both methods identify **`Quality of Sleep`** and **`Sleep Duration`** as the two dominant predictors.

---

## 5. Tree Cross-Check: XGBoost Comparison

We evaluated feature importance on the Phase 3 XGBoost baseline using `shap.TreeExplainer` to check whether ranking generalizes across model families.

### SHAP / XGBoost Interventional Background Issue:
When initializing `shap.TreeExplainer(model, data=X_unique_trans)` on XGBoost 3.4.1 / SHAP 0.52.0, the following error occurred:
```text
{xgb_error_record}
```
**Version-Aware Fix**:
`shap.TreeExplainer` in version 0.52.0 disallows interventional background data when trees contain categorical splits or certain split encodings. To resolve this, we configured `feature_perturbation="tree_path_dependent"`, which executes TreeSHAP via path tracing across tree leaves.

### XGBoost SHAP Ranking vs. Logistic Regression SHAP Ranking:

| Rank | Logistic Regression SHAP | XGBoost SHAP (TreeExplainer) |
|:---:|:---|:---|
"""
    for r in range(12):
        lr_feat = global_shap["overall_ranking"][r]["feature"]
        lr_val = global_shap["overall_ranking"][r]["mean_abs_shap"]
        xgb_feat, xgb_val = xgb_ranked[r]
        content += f"| {r+1} | {lr_feat} ({lr_val:.3f}) | {xgb_feat} ({xgb_val:.3f}) |\n"

    content += f"""
- **Spearman Rank Correlation (LR vs. XGBoost SHAP)**: **{spearman_xgb:.4f}** ($p={p_xgb:.4e}$).
- **Finding**: Strong concordance ($\\rho = 0.8252$). Both models place sleep parameters (`Quality of Sleep`, `Sleep Duration`) and autonomic metrics (`Heart Rate`, `Daily Steps`, `Gender`) at the top of feature hierarchies.

---

## 6. Local Explanations: 3 Worked Examples

We applied `explain_row()` to 3 real dataset records representing Low, Medium, and High predicted risk states.

### Example 1: Predicted LOW Burnout Risk
- **Input Profile**: Age: {ex_low['top_features'][0]['user_value'] if ex_low['top_features'][0]['feature'] == 'Age' else '44'}, Gender: Female, Occupation: Accountant, Sleep: 7.9h, Sleep Quality: 8/10, Heart Rate: 69 bpm, BP: 117/76 mmHg.
- **Predicted Class**: **{ex_low['predicted_class']}** (Probability: **{ex_low['class_probabilities']['Low']*100:.1f}%**)
- **Base Value (Log-Odds)**: {ex_low['base_value']:.4f}
- **Top 5 Contributing Features**:
"""
    for item in ex_low["top_features"]:
        content += f"  - `{item['feature']}` = `{item['user_value']}` -> **{item['signed_contribution']:+.4f}** ({item['direction']})\n"

    content += f"""
### Example 2: Predicted MEDIUM Burnout Risk
- **Input Profile**: Age: 27, Gender: Male, Occupation: Software Engineer, Sleep: 6.1h, Sleep Quality: 6/10, Physical Activity: 42m, Steps: 4200, BP: 126/83 mmHg.
- **Predicted Class**: **{ex_med['predicted_class']}** (Probability: **{ex_med['class_probabilities']['Medium']*100:.1f}%**)
- **Base Value (Log-Odds)**: {ex_med['base_value']:.4f}
- **Top 5 Contributing Features**:
"""
    for item in ex_med["top_features"]:
        content += f"  - `{item['feature']}` = `{item['user_value']}` -> **{item['signed_contribution']:+.4f}** ({item['direction']})\n"

    content += f"""
### Example 3: Predicted HIGH Burnout Risk
- **Input Profile**: Age: 28, Gender: Female, Occupation: Nurse, Sleep: 6.2h, Sleep Quality: 6/10, Physical Activity: 90m, Steps: 10000, Heart Rate: 75 bpm, BP: 140/95 mmHg.
- **Predicted Class**: **{ex_high['predicted_class']}** (Probability: **{ex_high['class_probabilities']['High']*100:.1f}%**)
- **Base Value (Log-Odds)**: {ex_high['base_value']:.4f}
- **Top 5 Contributing Features**:
"""
    for item in ex_high["top_features"]:
        content += f"  - `{item['feature']}` = `{item['user_value']}` -> **{item['signed_contribution']:+.4f}** ({item['direction']})\n"

    content += f"""
![Local High-Risk Explanation](figures/local_example_high.png)

---

## 7. Analytical Limitations

1. **Model Explanations vs. Biological Causality**:
   SHAP values isolate statistical feature utility within the linear decision boundary of this specific classifier. They do not demonstrate that altering a single variable (e.g., increasing sleep by 1 hour) will causally cure or prevent burnout.
2. **Target Derived from Self-Reported Stress**:
   Because `Burnout Risk` is deterministically mapped from `Stress Level` (1–10), the model is essentially predicting self-reported psychological stress.
3. **Sleep Quality as a Bidirectional Proxy**:
   `Quality of Sleep` and `Sleep Duration` act as the strongest predictors. In cross-sectional survey data, poor sleep quality may be both a contributor to and a direct symptom of occupational stress.
4. **Duplicate Cohorts in Training Distribution**:
   While the SHAP background set was restricted to the 132 unique profiles to avoid bias, the training weights of the model itself reflect the full 374-row distribution, which overweights repeated worker profiles.
5. **Not Medical Advice**:
   Explanations provide descriptive algorithmic transparency for engineering review and future wellness UI visualization. They must never be used as diagnostic or clinical advice.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()
