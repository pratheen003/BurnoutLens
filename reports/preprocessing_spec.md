# Preprocessing Specification & Design: Phase 2

**Module**: `src/burnoutlens/`  
**Tests**: `tests/test_preprocessing.py`  
**Schema Artifact**: `reports/feature_schema.json`  
**Status**: **VERIFIED - ALL 10 UNIT & INTEGRATION TESTS PASSED**

---

## 1. Overview & Scope

Phase 2 replaces the ad-hoc and flawed preprocessing from the original exploratory notebook (`data_understanding.ipynb` / Phase 1 baseline) with a modular, importable, leak-free preprocessing architecture. 

In accordance with Phase 2 instructions:
- **Zero model training or metric reporting** is performed.
- **Zero duplicate rows are deleted**; duplicate rows are tracked deterministically via `dup_group` to prepare for grouped validation in Phase 3.
- All Phase 1 source files in `src/recovery/` remain frozen and untouched.

---

## 2. Preprocessing Architecture: Phase 1 Baseline vs. Phase 2 Design

| Aspect | Phase 1 Baseline (Flawed Original) | Phase 2 Clean Design (Corrected) | Status |
| :--- | :--- | :--- | :--- |
| **Scaler Fit Location** | Fitted on the **entire** 374-row dataset before train/test split (**Data Leakage**) | Preprocessor is constructed **unfitted**; fitted strictly on the training partition within cross-validation or split | VERIFIED |
| **Categorical Encoding** | `LabelEncoder` applied to nominal categorical features (imposing spurious numerical ordering) | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` preserving nominal independence | VERIFIED |
| **Blood Pressure Feature** | Split into `Systolic BP` and `Diastolic BP`, but the original string `Blood Pressure` was **retained** in $X$ | `Blood Pressure` string column is **dropped**; only integer `Systolic BP` and `Diastolic BP` are retained | VERIFIED |
| **BMI Category Normalization** | Preserved raw split between `"Normal"` (195) and `"Normal Weight"` (21) | Merged `"Normal Weight"` into `"Normal"` (yielding 216 `"Normal"`, 148 `"Overweight"`, 10 `"Obese"`) | VERIFIED |
| **Duplicate Row Handling** | Ignored; 242 duplicates existed across train and test | **Zero rows removed**; deterministic hash assigned to `dup_group` (132 unique groups) for Phase 3 grouped splitting | VERIFIED |
| **Model Feature Count** | **13 features** (included redundant `Blood Pressure` string) | **12 features** (redundant `Blood Pressure` string dropped) | VERIFIED |
| **Derived Features Isolation** | `Lifestyle Score`, `Lifestyle Category`, `Burnout Index` computed inline | Isolated to analytics-only utilities; guarded against model inclusion via `assert_no_leakage()` | VERIFIED |

---

## 3. Cleaning Decisions and Rationale

1. **Whitespace Normalization**:
   - *Action*: Stripped leading/trailing whitespace across all text columns.
   - *Rationale*: Prevents silent category fragmentation caused by trailing spaces in user inputs or CSV records.
2. **Blood Pressure Decomposition and Removal**:
   - *Action*: Parsed `"Blood Pressure"` (`"126/83"`) into integer `"Systolic BP"` (`126`) and `"Diastolic BP"` (`83`), and dropped the raw string.
   - *Rationale*: The raw slash-delimited string is non-numeric and redundant once decomposed. Dropping it reduces feature count from 13 to 12 and eliminates duplicate representation.
3. **BMI Category Consolidation**:
   - *Action*: Mapped `"Normal Weight"` -> `"Normal"`.
   - *Rationale*: `"Normal Weight"` and `"Normal"` represent the identical clinical category in the underlying dataset taxonomy. Leaving them separate created an artificial category of only 21 samples.
4. **Identifier Elimination**:
   - *Action*: Dropped `"Person ID"`.
   - *Rationale*: Sequential row index with zero predictive value that would lead to spurious correlation if encoded.
5. **Missing Value Preservation in Sleep Disorder**:
   - *Action*: Preserved literal string `"None"` in `"Sleep Disorder"`.
   - *Rationale*: `"None"` denotes the clinical absence of a sleep disorder, not an unobserved missing value.
6. **Data Leakage Guards**:
   - *Action*: Implemented `assert_no_leakage(X)` checking against `FORBIDDEN_COLUMNS`.
   - *Rationale*: Prevents target (`Burnout Risk`), source target (`Stress Level`), derived index (`Burnout Index`), or auxiliary metrics (`Lifestyle Score`, `Lifestyle Category`, `Cluster`, `Person ID`) from entering model feature matrices.

---

## 4. Exact Feature Lists

### Input Features ($N = 12$)
1. `Gender` (Categorical)
2. `Age` (Numeric)
3. `Occupation` (Categorical)
4. `Sleep Duration` (Numeric)
5. `Quality of Sleep` (Numeric)
6. `Physical Activity Level` (Numeric)
7. `BMI Category` (Categorical)
8. `Heart Rate` (Numeric)
9. `Daily Steps` (Numeric)
10. `Sleep Disorder` (Categorical)
11. `Systolic BP` (Numeric)
12. `Diastolic BP` (Numeric)

### Numeric Features ($N = 8$)
`["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Heart Rate", "Daily Steps", "Systolic BP", "Diastolic BP"]`

### Categorical Features ($N = 4$)
`["Gender", "Occupation", "BMI Category", "Sleep Disorder"]`

### Forbidden Columns ($N = 7$)
`["Burnout Risk", "Stress Level", "Burnout Index", "Lifestyle Score", "Lifestyle Category", "Cluster", "Person ID"]`

---

## 5. Feature Schema Summary (`reports/feature_schema.json`)

Derived empirically from the 374 raw records without fabrication:

### Categorical Features Summary
- **`Gender`**: Allowed: `["Female", "Male"]` (Male: 189, Female: 185)
- **`Occupation`** (11 categories): Allowed: `["Accountant", "Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Sales Representative", "Salesperson", "Scientist", "Software Engineer", "Teacher"]`  
  - Counts: Nurse (73), Doctor (71), Engineer (63), Lawyer (47), Teacher (40), Accountant (37), Salesperson (32), Software Engineer (4), Scientist (4), Sales Representative (2), Manager (1).
- **`BMI Category`**: Allowed: `["Normal", "Obese", "Overweight"]` (Normal: 216, Overweight: 148, Obese: 10)
- **`Sleep Disorder`**: Allowed: `["Insomnia", "None", "Sleep Apnea"]` (None: 219, Sleep Apnea: 78, Insomnia: 77)

### Numeric Features Summary
| Feature | Dtype | Min | Max | Median |
| :--- | :--- | :--- | :--- | :--- |
| **`Age`** | `int64` | 27.0 | 59.0 | 43.0 |
| **`Sleep Duration`** | `float64` | 5.8 | 8.5 | 7.2 |
| **`Quality of Sleep`** | `int64` | 4.0 | 9.0 | 7.0 |
| **`Physical Activity Level`** | `int64` | 30.0 | 90.0 | 60.0 |
| **`Heart Rate`** | `int64` | 65.0 | 86.0 | 70.0 |
| **`Daily Steps`** | `int64` | 3,000.0 | 10,000.0 | 7,000.0 |
| **`Systolic BP`** | `int64` | 115.0 | 142.0 | 130.0 |
| **`Diastolic BP`** | `int64` | 75.0 | 95.0 | 85.0 |

---

## 6. Duplicate Analysis and Group Tracking

- **Total Rows**: `374` [VERIFIED]
- **Duplicate Rows**: `242` duplicate rows exist across the 12 input features [VERIFIED].
- **Unique Duplicate Groups**: `132` distinct lifestyle profiles (`dup_group` hash values) [VERIFIED].
- **Policy in Phase 2**: No duplicate rows are dropped. Each record is assigned its deterministic `dup_group` hash via `add_duplicate_group_id()`.

---

## 7. Status Classifications

- **[VERIFIED]**:
  - All 10 unit and integration tests passed cleanly in pytest (`test_clean_data_shape_and_columns`, `test_target_counts`, `test_bmi_normalization`, `test_sleep_disorder_values`, `test_blood_pressure_split_and_drop`, `test_assert_no_leakage`, `test_immutability_and_raw_integrity`, `test_preprocessor_unfitted_and_transform`, `test_duplicate_group_id`, `test_compute_lifestyle_score_hand_checkable`).
  - Raw dataset SHA-256 hash remained unchanged: `1EFE7B6F781FF88078D08D81FE136CFF98B1B0C32560F35F8650CA5984B77841`.
  - Feature count reduced from 13 to 12.
  - Zero leakage verified with `assert_no_leakage`.
- **[HYPOTHESIS - TO BE TESTED IN PHASE 3]**:
  - The presence of 242 duplicate rows across the 374 total rows may artificially inflate model accuracy and cross-validation scores when randomly split. Phase 3 will test this hypothesis using `GroupKFold` / `GroupShuffleSplit` on `dup_group` vs standard random stratified splits.
- **[FUTURE - PHASE 3 / PHASE 4]**:
  - Train clean benchmark models using the Phase 2 pipeline.
  - Evaluate impact of GroupKFold on test accuracy.
  - Deploy API endpoints validating incoming payloads against `reports/feature_schema.json`.
