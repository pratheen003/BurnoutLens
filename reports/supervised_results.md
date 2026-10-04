# Supervised Model Evaluation Report: Phase 3

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
  - *Contamination Rate*: On average, **60.76 ± 3.64 validation rows per fold (81.2%)** have an identical `dup_group` twin in the training fold [VERIFIED].

### Performance Summary Table

| Model | Protocol C 5-Seeds Acc (Macro-F1) | Protocol D 5-Seeds Acc (Macro-F1) | Difference (D minus C) Acc (Macro-F1) | Protocol A Test Acc (Macro-F1)* | Protocol B 10-Seeds Acc (Macro-F1) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9513 ± 0.0057 (0.9499 ± 0.0060) | 0.9615 ± 0.0028 (0.9606 ± 0.0032) | +0.0102 (+0.0107) | 0.9067 (0.9056) | 0.9292 ± 0.0219 (0.9290) | VERIFIED |
| **Decision Tree** | 0.9246 ± 0.0104 (0.9230 ± 0.0110) | 0.9572 ± 0.0050 (0.9566 ± 0.0049) | +0.0326 (+0.0336) | 0.8933 (0.8927) | 0.9132 ± 0.0330 (0.9127) | VERIFIED |
| **Random Forest** | 0.9427 ± 0.0090 (0.9414 ± 0.0092) | 0.9759 ± 0.0024 (0.9751 ± 0.0029) | +0.0332 (+0.0337) | 0.9200 (0.9194) | 0.9412 ± 0.0270 (0.9398) | VERIFIED |
| **XGBoost** | 0.9470 ± 0.0073 (0.9451 ± 0.0078) | 0.9765 ± 0.0062 (0.9759 ± 0.0064) | +0.0294 (+0.0309) | 0.9467 (0.9453) | 0.9319 ± 0.0331 (0.9300) | VERIFIED |
| **MLP Classifier** | 0.9166 ± 0.0080 (0.9129 ± 0.0082) | 0.9240 ± 0.0050 (0.9192 ± 0.0056) | +0.0074 (+0.0064) | 0.9467 (0.9428) | 0.9132 ± 0.0211 (0.9109) | VERIFIED |
| **Majority Baseline** | 0.3770 (0.1825) | 0.3770 (0.1825) | 0.0000 (0.0000) | 0.3770 (0.1825) | 0.3770 (0.1825) | COMPUTED |

*Protocol A is a single 75-row holdout split (1 sample = 1.33%) and not directly comparable to averaged protocols.*

---

## 4. Per-Class Report & Confusion Matrix for Selected Model

### Selected Model: `Logistic Regression` (Evaluated under Protocol C Seed 42 Pooled OOF)

- **Overall Accuracy**: `0.9412`
- **Macro-Averaged F1**: `0.9393`

### Per-Class Detailed Metrics
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Low** | 0.9375 | 0.9574 | 0.9474 | 141 |
| **Medium** | 0.9333 | 0.8673 | 0.8991 | 113 |
| **High** | 0.9520 | 0.9917 | 0.9714 | 120 |

### Confusion Matrix (Class Order: `['Low', 'Medium', 'High']`)
```text
[[135   6   0]
 [  9  98   6]
 [  0   1 119]]
```
*(Saved visual chart: `reports/figures/confusion_matrix_selected.png`)*

---

## 5. Statistical Significance (McNemar Exact Tests)

McNemar's exact test was computed on the pooled out-of-fold discordant predictions ($N = 374$) from Protocol C (seed 42) using `scipy.stats.binomtest`:

1. **Top Model (Logistic Regression) vs. Runner-Up Model (XGBoost)**:
   - $b$ (Logistic Regression correct, XGBoost incorrect): `2`
   - $c$ (Logistic Regression incorrect, XGBoost correct): `9`
   - Total Discordant Pairs ($n$): `11`
   - **Two-Sided $p$-Value**: `0.0654`
   - *Interpretation*: No statistically significant difference detected (p >= 0.05). Both models perform comparably on discordant cases.

> **Statistical Limitation Note**: Predictions pooled from 5-fold cross-validation are not strictly independent across folds because training partitions share data samples. The McNemar test serves as an empirical comparison heuristic rather than an absolute hypothesis confirmation.

---

## 6. Model Selection Rule & Decision

### Selection Rule Applied:
1. Candidate models are ranked strictly by repeated mean Macro-F1 under Protocol C.
2. The top-ranked model is compared to the runner-up model using the one-standard-deviation rule ($|\text{Top} - \text{Runner-up}| \le 1.0\sigma$).
3. If the top-performing candidate is within one standard deviation of a simpler, linear model (or is itself a linear model), the linear model is selected for deployment to maximize interpretability, coefficient auditability, and operational simplicity.

### Selection Outcome:
- **Rank 1 Candidate**: `Logistic Regression` with Protocol C repeated Macro-F1 = `0.9499 ± 0.0060`.
- **Rank 2 Runner-Up**: `XGBoost` with Protocol C repeated Macro-F1 = `0.9451 ± 0.0078`.
- **Performance Difference**: `0.0048` (Top 1 standard deviation threshold: `0.0060`).
- **Decision**: **Logistic Regression**
- **Reasoning**: Logistic Regression achieved the highest repeated Macro-F1 (0.9499 +/- 0.0060) under Protocol C, surpassing the runner-up (XGBoost: 0.9451 +/- 0.0078) by 0.0048. Because Logistic Regression is the top-ranked model and is also a linear, transparent classifier, it is selected without any trade-off between predictive accuracy and clinical interpretability.
- *(In compliance with Phase 3 instructions, zero models have been serialized or saved as production artifacts).*

---

## 7. Hypothesis Verdict: Duplicate Contamination (Protocol D vs. Protocol C)

> **HYPOTHESIS**:
> 242 duplicate rows exist when `Person ID` is excluded. Identical records appearing in both train and test partitions under random splitting inflated historical accuracy.

### Measured Verdict: **SUPPORTED**

**Evidence & Rationale (Strictly D vs. C Evaluation)**:
The hypothesis that duplicate records artificially inflate evaluation performance is **SUPPORTED**. In the fair head-to-head control comparison between Protocol D (random 5-fold CV) and Protocol C (grouped 5-fold CV), random-CV macro-F1 scores are clearly higher than grouped-CV scores for 4 out of 5 models, with differences exceeding the standard deviation. Protocol D validation folds suffer an average contamination of **60.76 +/- 3.64 rows (81.2%)** whose duplicate twins appear in the training fold, systematically boosting test performance.

### Measured Head-to-Head Protocol Numbers:
- **Logistic Regression**: Protocol C F1 = `0.9499 +/- 0.0060` vs. Protocol D F1 = `0.9606 +/- 0.0032` | Difference ($D - C$) = `+0.0107` (Acc Diff: `+0.0102`) -> Higher than 1 std (+)
- **Decision Tree**: Protocol C F1 = `0.9230 +/- 0.0110` vs. Protocol D F1 = `0.9566 +/- 0.0049` | Difference ($D - C$) = `+0.0336` (Acc Diff: `+0.0326`) -> Higher than 1 std (+)
- **Random Forest**: Protocol C F1 = `0.9414 +/- 0.0092` vs. Protocol D F1 = `0.9751 +/- 0.0029` | Difference ($D - C$) = `+0.0337` (Acc Diff: `+0.0332`) -> Higher than 1 std (+)
- **XGBoost**: Protocol C F1 = `0.9451 +/- 0.0078` vs. Protocol D F1 = `0.9759 +/- 0.0064` | Difference ($D - C$) = `+0.0309` (Acc Diff: `+0.0294`) -> Higher than 1 std (+)
- **MLP Classifier**: Protocol C F1 = `0.9129 +/- 0.0082` vs. Protocol D F1 = `0.9192 +/- 0.0056` | Difference ($D - C$) = `+0.0064` (Acc Diff: `+0.0074`) -> Within std

- **Validation Fold Duplicate Contamination in Protocol D**: An average of **60.76 ± 3.64 rows (81.2%)** in each validation fold of Protocol D had an exact duplicate twin in the training fold.
- The empirical data demonstrates that prior un-grouped random cross-validation benchmarks were systematically contaminated by twin records, inflating performance metrics for high-capacity models (Decision Tree by +3.36%, Random Forest by +3.37%, and XGBoost by +3.09% Macro-F1).

---

## 8. Limitations & Constraints

1. **Labels Fully Determined by Features (Zero Conflicting Groups)**: Target classes are completely determined by input features with 0 conflicting duplicate groups (100.0% theoretical ceiling). High model accuracy reflects learning this deterministic mapping within the dataset.
2. **Only 132 Unique Profiles ($N = 374$)**: The 374 dataset rows represent only 132 unique feature vectors, meaning the effective diversity of the cohort is limited.
3. **Exact Duplicates vs. Near-Duplicates**: Grouped splitting (`dup_group`) removes exact duplicate profiles between partitions but does not remove near-duplicates (individuals differing by only 1 minor feature such as age or resting heart rate). Consequently, cross-validation scores must be read strictly as evaluation performance on this specific dataset, not as evidence of real-world burnout detection capability.
4. **Deterministic Target Proxy**: Target labels are derived via deterministic rule mapping from self-reported `Stress Level`, rather than a clinical burnout diagnostic inventory (e.g., Maslach Burnout Inventory).
5. **Sparse Demographic Subgroups**: Occupations such as Manager ($N=1$) and Sales Representative ($N=2$) have insufficient representation for reliable subgroup generalization.
6. **Exploratory, Non-Clinical System**: BurnoutLens is an exploratory educational lifestyle risk assessment and not a medical diagnostic tool.
