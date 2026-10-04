# Baseline Reproduction Report: BurnoutLens Project

**Script**: `src/recovery/reproduce_notebook.py`  
**Reference Source**: `notebooks/data_understanding.ipynb`  
**Dataset**: `data/raw/Sleep_health_and_lifestyle_dataset.csv`  
**Status**: **VERIFIED - EXACT REPRODUCTION ACROSS ALL METRICS**

---

## 1. Reproduction Summary Table

| Model | Historical Test Acc | Reproduced Test Acc | Historical CV Mean | Reproduced CV Mean | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9600 [HISTORICAL] | 0.9600 [VERIFIED] | 0.9251 [HISTORICAL] | 0.9251 [VERIFIED] | VERIFIED (Exact Match) |
| **Decision Tree** | 0.9733 [HISTORICAL] | 0.9733 [VERIFIED] | 0.8956 [HISTORICAL] | 0.8956 [VERIFIED] | VERIFIED (Exact Match) |
| **Random Forest** | 0.9467 [HISTORICAL] | 0.9467 [VERIFIED] | 0.9250 [HISTORICAL] | 0.9250 [VERIFIED] | VERIFIED (Exact Match) |
| **XGBoost** | 0.9733 [HISTORICAL] | 0.9733 [VERIFIED] | 0.9252 [HISTORICAL] | 0.9252 [VERIFIED] | VERIFIED (Exact Match) |
| **MLP Classifier** | 0.8800 [HISTORICAL] | 0.8800 [VERIFIED] | 0.8065 [HISTORICAL] | 0.8065 [VERIFIED] | VERIFIED (Exact Match) |

> **Note on MLP Stability**: In historical cross-validation, the 5th CV fold for MLP was unstable at `~0.446`. The reproduced 5-fold CV scores are `[0.8267, 0.9067, 0.9333, 0.9200, 0.4459]` with a mean of `0.8065`, exactly reproducing the instability on fold 5 [VERIFIED].

---

## 2. Environment & Dependencies

- **Python Version**: `Python 3.14.6` [VERIFIED]
- **Key Packages**:
  - `pandas`: `3.0.6` [VERIFIED]
  - `numpy`: `2.5.3` [VERIFIED]
  - `scikit-learn`: `1.9.1` [VERIFIED]
  - `xgboost`: `3.4.1` [VERIFIED]
  - `shap`: `0.52.0` [VERIFIED]
  - `matplotlib`: `3.11.2` [VERIFIED]
  - `seaborn`: `0.13.2` [VERIFIED]
  - `joblib`: `1.6.0` [VERIFIED]
  - `fastapi`: `0.142.2` [VERIFIED]
  - `uvicorn`: `0.54.0` [VERIFIED]
  - `pytest`: `9.1.1` [VERIFIED]

---

## 3. Faithful Reproduction Preprocessing Pipeline

The original notebook pipeline was replicated exactly without modifying or "fixing" its methodological flaws:

1. **Drop Person ID**: Column `Person ID` dropped from dataset [VERIFIED].
2. **Missing Value Imputation**: `Sleep Disorder` null values imputed with string `"None"` [VERIFIED].
3. **Blood Pressure Handling**:
   - `Blood Pressure` string split into `Systolic BP` and `Diastolic BP` as integers [VERIFIED].
   - **Flaw Preserved**: The original `Blood Pressure` string column was **retained** in `X` rather than dropped, resulting in redundant categorical + numeric representations of blood pressure [VERIFIED].
4. **Target Derivation**:
   - `Stress Level <= 4` -> `Low`
   - `Stress Level 5 - 6` -> `Medium`
   - `Stress Level >= 7` -> `High` [VERIFIED]
5. **Intermediate Feature Generation**:
   - `Lifestyle Score`, `Lifestyle Category`, and `Burnout Index` computed using the exact notebook functions [VERIFIED].
   - Dropped from `X` alongside `Burnout Risk` and `Stress Level` prior to modeling [VERIFIED].
6. **Feature Matrix ($X$)**:
   - Exact 13 columns: `['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps', 'Sleep Disorder', 'Systolic BP', 'Diastolic BP']` [VERIFIED].
   - Shape: `(374, 13)` [VERIFIED].
7. **Categorical Encoding**:
   - Scikit-learn `LabelEncoder` applied to every object/string column in $X$ [VERIFIED].
   - Target variable $y$ encoded using `LabelEncoder`: `{0: 'High', 1: 'Low', 2: 'Medium'}` [VERIFIED].
8. **Data Leakage Flaw (Faithfully Preserved)**:
   - `StandardScaler().fit_transform(X)` executed on the **entire** 374-row feature matrix **before** splitting into train and test sets [VERIFIED].
9. **Data Splitting**:
   - `train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)` [VERIFIED].
   - Training Set: `299` samples x `13` features [VERIFIED].
   - Testing Set: `75` samples x `13` features [VERIFIED].

---

## 4. Class Distribution & Confusion Matrix

### Target Counts
- **Full Dataset ($N=374$)**:
  - `Low`: 141 (37.70%) [VERIFIED]
  - `High`: 120 (32.09%) [VERIFIED]
  - `Medium`: 113 (30.21%) [VERIFIED]
- **Test Set ($N=75$)**:
  - `Low`: 28 (37.33%) [VERIFIED]
  - `High`: 24 (32.00%) [VERIFIED]
  - `Medium`: 23 (30.67%) [VERIFIED]
- **Train Set ($N=299$)**:
  - `Low`: 113 (37.79%) [VERIFIED]
  - `High`: 96 (32.11%) [VERIFIED]
  - `Medium`: 90 (30.10%) [VERIFIED]

### Logistic Regression Confusion Matrix (Class Order: `['High', 'Low', 'Medium']`)
- **Historical**:
  ```text
  [[24,  0,  0],
   [ 0, 27,  1],
   [ 2,  0, 21]]
  ```
- **Reproduced**:
  ```text
  [[24,  0,  0],
   [ 0, 27,  1],
   [ 2,  0, 21]]
  ```
- **Status**: **EXACT MATCH [VERIFIED]**

---

## 5. Mismatches and Explanations

- **Observed Metric Mismatches**: **None (0 mismatches)**.
- All test accuracies, cross-validation fold scores, cross-validation means, and confusion matrices match the historical notebook outputs to 4 decimal places.
- Reproducibility confirmed across modern scikit-learn, XGBoost, and Python runtimes when matching the exact random seed (`42`), stratified split, and preprocessing sequence.

---

## 6. HYPOTHESIS - TO BE TESTED IN PHASE 3

> **HYPOTHESIS - TO BE TESTED IN PHASE 3**:
> 242 duplicate rows exist when `Person ID` is excluded (`132` unique feature rows out of `374`). Identical records may appear in both train and test sets due to random splitting, which may artificially inflate test accuracy and cross-validation scores. This will be formally evaluated and tested under de-duplicated and strictly isolated splits during Phase 3.
