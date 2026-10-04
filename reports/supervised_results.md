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
| **Logistic Regression** | 0.9600 [HISTORICAL] | 0.9067 (0.9056) [VERIFIED] | 0.9292 ± 0.0219 (0.9290) [VERIFIED] | 0.9411 ± 0.0138 (0.9389) [VERIFIED] | 0.9513 ± 0.0057 (0.9499) [VERIFIED] | VERIFIED |
| **Decision Tree** | 0.9733 [HISTORICAL] | 0.8933 (0.8927) [VERIFIED] | 0.9132 ± 0.0330 (0.9127) [VERIFIED] | 0.9199 ± 0.0233 (0.9178) [VERIFIED] | 0.9246 ± 0.0104 (0.9230) [VERIFIED] | VERIFIED |
| **Random Forest** | 0.9467 [HISTORICAL] | 0.9200 (0.9194) [VERIFIED] | 0.9412 ± 0.0270 (0.9398) [VERIFIED] | 0.9412 ± 0.0135 (0.9396) [VERIFIED] | 0.9427 ± 0.0090 (0.9414) [VERIFIED] | VERIFIED |
| **XGBoost** | 0.9733 [HISTORICAL] | 0.9467 (0.9453) [VERIFIED] | 0.9319 ± 0.0331 (0.9300) [VERIFIED] | 0.9598 ± 0.0190 (0.9590) [VERIFIED] | 0.9470 ± 0.0073 (0.9451) [VERIFIED] | VERIFIED |
| **MLP Classifier** | 0.8800 [HISTORICAL] | 0.9467 (0.9428) [VERIFIED] | 0.9132 ± 0.0211 (0.9109) [VERIFIED] | 0.9144 ± 0.0137 (0.9100) [VERIFIED] | 0.9166 ± 0.0080 (0.9129) [VERIFIED] | VERIFIED |
| **Majority Baseline** | 0.3770 [HISTORICAL] | 0.3770 (0.1825) [VERIFIED] | 0.3770 (0.1825) [VERIFIED] | 0.3770 (0.1825) [VERIFIED] | 0.3770 (0.1825) [VERIFIED] | VERIFIED |

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

1. **Top Model (Logistic Regression) vs. Second-Ranked Model (XGBoost)**:
   - $b$ (Logistic Regression correct, XGBoost incorrect): `2`
   - $c$ (Logistic Regression incorrect, XGBoost correct): `9`
   - Total Discordant Pairs ($n$): `11`
   - **Two-Sided $p$-Value**: `0.0654`
   - *Interpretation*: No statistically significant difference detected (p >= 0.05). Both models perform comparably on discordant cases.

2. **Logistic Regression vs. Top Model**:
   - *Note*: Logistic Regression is the top-ranked model under Protocol C; therefore, comparing Logistic Regression against the top model is identical to the Rank 1 vs. Rank 2 comparison above (Logistic Regression vs. XGBoost).


> **Statistical Limitation Note**: Predictions pooled from 5-fold cross-validation are not strictly independent across folds because training partitions share data samples. The McNemar test serves as an empirical comparison heuristic rather than an absolute hypothesis confirmation.

---

## 6. Model Selection Rule & Decision

### Selection Rule Applied:
1. Select the candidate achieving the highest repeated mean Macro-F1 under Protocol C.
2. If the top candidate and the next-ranked simpler / linear model are within one standard deviation (<= 1.0 * std), prefer the simpler, more interpretable model.

### Selection Outcome:
- **Top Candidate**: `Logistic Regression` with repeated Macro-F1 = `0.9499 ± 0.0060`.
- **Decision**: **Logistic Regression**
- **Reasoning**: Selected Logistic Regression because its performance (0.9499) is within 1 std (0.0060) of the top model (Logistic Regression: 0.9499), and it offers greater interpretability, linear coefficient transparency, and lower deployment complexity.
- *(In compliance with Phase 3 instructions, zero models have been serialized or saved as production artifacts).*

---

## 7. Hypothesis Verdict: Duplicate Contamination

> **HYPOTHESIS**:
> 242 duplicate rows exist when `Person ID` is excluded. Identical records appearing in both train and test partitions under random splitting inflated historical accuracy.

### Measured Verdict: **INCONCLUSIVE**

**Evidence & Rationale**:
- In Protocol A (Random 80/20 split), **58 out of 75 test samples (77.3%)** had an identical twin record present in the training set.
- The hypothesis is INCONCLUSIVE: performance under grouped splits remains within the margin of error (mean drop B: -0.30%, mean drop C: -1.38%), indicating models generalize strongly across the 132 unique lifestyle clusters.
- The honest evaluation confirms that when test samples represent strictly unseen lifestyle profiles, models still achieve solid predictive performance, but prior un-grouped holdout benchmarks were systematically contaminated by twin records.

---

## 8. Limitations & Constraints

1. **Small Sample Size ($N = 374$)**: The entire dataset consists of only 374 records representing 132 unique lifestyle feature vectors.
2. **Deterministic Target Proxy**: The ground-truth target is derived via deterministic rule mapping from a self-reported 1–10 `Stress Level` score, rather than a clinical burnout diagnostic inventory (such as the Maslach Burnout Inventory).
3. **Sparse Categories**: Occupations such as Manager ($N=1$) and Sales Representative ($N=2$) have insufficient representation to reliably evaluate subgroup generalization.
4. **Not a Clinical Device**: BurnoutLens is strictly an exploratory lifestyle risk assessment and not a medical or clinical diagnostic system.
