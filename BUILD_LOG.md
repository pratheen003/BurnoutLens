# BurnoutLens Build & Recovery Log

## 2026-10-04 - Phase 1 Setup & Initialization

### Environment & System Information
- **OS**: Windows 11 / x86_64
- **Date/Time**: 2026-10-04T10:19:06+05:30 to 2026-10-04T10:50:00+05:30
- **Python**: 3.14.6 (tags/v3.14.6:c63aec6, Jun 10 2026, 10:26:10) [MSC v.1944 64 bit (AMD64)]
- **Installed Package Versions**:
  - `pandas`: 3.0.6
  - `numpy`: 2.5.3
  - `scikit-learn`: 1.9.1
  - `xgboost`: 3.4.1
  - `shap`: 0.52.0
  - `matplotlib`: 3.11.2
  - `seaborn`: 0.13.2
  - `joblib`: 1.6.0
  - `fastapi`: 0.142.2
  - `uvicorn`: 0.54.0
  - `pytest`: 9.1.1

---

### Step 0: Inspection & Git Setup
- Command: `Get-ChildItem; git status; git remote -v; git branch`
  - Output: `git` was not initially found in PATH.
  - Source files present:
    - `BurnoutLens AI - review 2.pptx` (2,343,492 bytes)
    - `BurnoutLens_Recovery_Context.docx` (43,570 bytes)
    - `data_understanding.ipynb` (1,420,684 bytes)
    - `Sleep_health_and_lifestyle_dataset.csv` (24,137 bytes)
- MinGit Installation:
  - Downloaded `MinGit-2.44.0-64-bit.zip` from official git-for-windows releases.
  - Extracted to `C:\Users\acer\AppData\Local\Programs\Git`.
  - Added `C:\Users\acer\AppData\Local\Programs\Git\cmd` to user PATH and session PATH.
  - Verified `git --version`: `git version 2.44.0.windows.1`.
- Repository Initialization:
  - `git init` -> Initialized empty Git repository in `D:/BurnoutLens/.git/`
  - `git branch -M main`
  - `git remote add origin https://github.com/pratheen003/BurnoutLens`
  - Identified `user.name` and `user.email` unset; stopped per Step 0 rules.
  - User configured `git config --global user.name "Pratheen K"` and `git config --global user.email "pratheen003@gmail.com"`.

---

### Step 1: Python Virtual Environment & Dependencies
- Command: `python -m venv venv`
- Upgraded pip: `.\venv\Scripts\python.exe -m pip install --upgrade pip` -> Upgraded to pip 26.2.1
- Package Installation: `.\venv\Scripts\pip.exe install pandas numpy scikit-learn xgboost shap matplotlib seaborn joblib fastapi uvicorn pytest`
  - Error encountered during parallel install: `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: ...`
  - Resolution: Stopped conflicting python background tasks (`Stop-Process -Name python -Force`), re-ran `pip install`. All packages installed successfully.
- Import Verification:
  - Executed import test script in `.\venv\Scripts\python.exe`.
  - All 11 libraries imported with status OK.
- Pinned Requirements:
  - Executed `pip freeze > requirements.txt` to capture exact reproducible dependencies.

---

### Step 2: Project Structure Setup
- Created target directory tree:
  - `data/raw/`
  - `notebooks/`
  - `src/recovery/`
  - `reports/`
  - `models/`
  - `tests/`
  - `docs/`
- Added `.gitkeep` to all created directories.
- Created `.gitignore` excluding `venv/`, `__pycache__/`, `*.pyc`, `.env`, `.ipynb_checkpoints/`, `.DS_Store`, `.vscode/`, `.idea/`, `data/raw/*.csv`, `/Sleep_health_and_lifestyle_dataset.csv`, `/*.docx`, `/*.pptx`, `/data_understanding.ipynb`.
- Created `README.md` with explicit medical disclaimers, Kaggle dataset origin notes, and project architecture overview.

---

### Step 3: Source Preservation & Integrity Checks
- Copied (did not move):
  - `Sleep_health_and_lifestyle_dataset.csv` -> `data/raw/Sleep_health_and_lifestyle_dataset.csv`
  - `data_understanding.ipynb` -> `notebooks/data_understanding.ipynb`
- SHA-256 Checksums:
  - CSV Original: `1EFE7B6F781FF88078D08D81FE136CFF98B1B0C32560F35F8650CA5984B77841`
  - CSV Copy:     `1EFE7B6F781FF88078D08D81FE136CFF98B1B0C32560F35F8650CA5984B77841`
  - CSV Verification: **MATCH (VERIFIED)**
  - Notebook Original: `BC54DBEBC1197918AEBD25BE5555A8C490BC35222B7CC508F690DD24FE87F4ED`
  - Notebook Copy:     `BC54DBEBC1197918AEBD25BE5555A8C490BC35222B7CC508F690DD24FE87F4ED`
  - Notebook Verification: **MATCH (VERIFIED)**
- Attribute: `data/raw/Sleep_health_and_lifestyle_dataset.csv` set to Read-Only (`IsReadOnly = True`).
- Original files in workspace root remain untouched.

---

### Commit 1
- **Hash**: `e0091f0`
- **Message**: `"chore: initialize BurnoutLens project"`
- **Staged files**: `.gitignore`, `BUILD_LOG.md`, `README.md`, `requirements.txt`, `notebooks/data_understanding.ipynb`, and `.gitkeep` files in `data/raw/`, `docs/`, `models/`, `notebooks/`, `reports/`, `src/recovery/`, `tests/`.

---

### Step 4: Data Verification (`src/recovery/verify_data.py`)
- Created `src/recovery/verify_data.py`.
- Command: `.\venv\Scripts\python.exe src/recovery/verify_data.py`
- Generated: `reports/data_audit.md`
- Results:
  - Dataset Shape: `(374, 13)` [VERIFIED - Expected 374x13]
  - Column Names: All 13 columns present and ordered correctly [VERIFIED]
  - Missing Values: 0 missing values across all columns under `keep_default_na=False`; under standard `pd.read_csv()`, `"None"` in `Sleep Disorder` becomes `NaN` (219 rows) [VERIFIED]
  - Duplicate Records:
    - Including `Person ID`: 0 duplicates [VERIFIED - Expected 0]
    - Excluding `Person ID`: 242 duplicates [VERIFIED - Expected 242] (Measured only; no records dropped per Phase 1 rules)
  - Target Distribution (`Burnout Risk` mapped from `Stress Level`):
    - `Low` (Stress <= 4): 141 [VERIFIED - Expected 141]
    - `Medium` (Stress 5 - 6): 113 [VERIFIED - Expected 113]
    - `High` (Stress >= 7): 120 [VERIFIED - Expected 120]
  - Audit Mismatches: **0 mismatches** [VERIFIED]

---

### Step 5: Faithful Notebook Baseline Reproduction (`src/recovery/reproduce_notebook.py`)
- Created `src/recovery/reproduce_notebook.py` replicating `notebooks/data_understanding.ipynb` logic verbatim:
  - Dropped `Person ID`
  - Imputed `Sleep Disorder` with `"None"`
  - Split `Blood Pressure` into `Systolic BP` and `Diastolic BP` while retaining original `Blood Pressure` column in $X$ (13 total features)
  - Derived `Burnout Risk`
  - Replicated exact `calculate_lifestyle_score`, `lifestyle_category`, and `Burnout Index`
  - Dropped derived/target columns, retaining `Blood Pressure`
  - Categorical encoding with `LabelEncoder`
  - Replicated scaling flaw: `StandardScaler().fit_transform(X)` on entire dataset before split
  - `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` -> 299 train, 75 test
  - Trained and evaluated all 5 models + 5-fold cross-validation
- Command: `.\venv\Scripts\python.exe src/recovery/reproduce_notebook.py`
- Output:
  - Logistic Regression: Test Acc `0.9600` (CV Mean: `0.9251`)
  - Decision Tree: Test Acc `0.9733` (CV Mean: `0.8956`)
  - Random Forest: Test Acc `0.9467` (CV Mean: `0.9250`)
  - XGBoost: Test Acc `0.9733` (CV Mean: `0.9252`)
  - MLP Classifier: Test Acc `0.8800` (CV Mean: `0.8065`, Fold 5: `0.4459`)
  - Confusion Matrix (LR): `[[24, 0, 0], [0, 27, 1], [2, 0, 21]]` for classes `['High', 'Low', 'Medium']`
  - Exact match across every single model and CV score [VERIFIED]

---

### Step 6: Baseline Reproduction Report
- Created `reports/reproduction.md` containing:
  - Full comparison table between historical and reproduced results (all exact matches)
  - Environment and package versions
  - Detailed preprocessing pipeline documentation
  - Class distribution and confusion matrices
  - "HYPOTHESIS - TO BE TESTED IN PHASE 3" regarding the 242 duplicate rows and potential data contamination across train/test splits.

---

### Commit 2
- **Hash**: `fba7fb6`
- **Message**: `"feat: recover dataset and preprocessing pipeline"`
- **Staged files**: `BUILD_LOG.md`, `reports/data_audit.md`, `reports/reproduction.md`, `src/recovery/reproduce_notebook.py`, `src/recovery/verify_data.py`.

---

### Step 7: Push Attempt & Authentication Status
- **Remote Origin**: `https://github.com/pratheen003/BurnoutLens`
- **Command**: `git push -u origin main`
- **Output / Result**:
  ```text
  fatal: could not read Username for 'https://github.com': terminal prompts disabled
  ```
- **Error Analysis**: Authentication requires interactive GitHub login credentials (Personal Access Token or GitHub CLI / SSH key). Execution stopped per Step 7 instructions.

---

## 2026-10-04 - Phase 2: Modular Preprocessing & Feature Engineering

### Scope & Constraints
- Implement clean, reusable, importable preprocessing pipeline without model training or metric reporting.
- Phase 1 baseline files in `src/recovery/` kept frozen.
- Raw CSV and notebook kept untouched.
- Duplicate rows measured and tracked via `dup_group`; zero rows deleted in Phase 2.

### Step 0: Inspection & Clean Tree Check
- Command: `git status; git branch` -> Branch `main`, working tree clean.

### Step 1: Package Creation (`src/burnoutlens/`)
- Created package `src/burnoutlens/` with `__init__.py`:
  - `config.py`: Defined `RAW_CSV_PATH`, `TARGET_COL="Burnout Risk"`, `LABELS=["Low","Medium","High"]`, target thresholds (4 and 6), `NUMERIC_FEATURES` (8 features), `CATEGORICAL_FEATURES` (4 features), `INPUT_FEATURES` (12 features), `FORBIDDEN_COLUMNS` (7 forbidden columns). Explicitly corrected feature count from 13 to 12 by dropping redundant raw `Blood Pressure` string.
  - `data.py`: `load_raw()` loading raw CSV with `keep_default_na=False` so `"None"` in `Sleep Disorder` is preserved. Never mutates or writes to disk.
  - `features.py`:
    - `clean_data(df)`: Immutably strips whitespace, decomposes and drops `Blood Pressure`, normalizes `"Normal Weight"` to `"Normal"`, drops `"Person ID"`.
    - `make_target(df)`: Generates target series (`"Low"`, `"Medium"`, `"High"`).
    - `compute_lifestyle_score(df)`: Replicates exact notebook lifestyle formula for analytics only.
    - `add_duplicate_group_id(df)`: Assigns deterministic sha256 hash to `dup_group` across input features (yields 132 unique groups out of 374 rows; zero rows deleted).
  - `preprocessing.py`: `build_preprocessor()` returning unfitted `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` for categorical features. Supports `get_feature_names_out()`.
  - `leakage.py`: `assert_no_leakage(X)` raising `ValueError` if any `FORBIDDEN_COLUMNS` are present.

### Step 2: Input Schema Generation (`src/burnoutlens/schema.py`)
- Created `src/burnoutlens/schema.py` and generated `reports/feature_schema.json` directly from the raw dataset:
  - 8 numeric features with empirically derived min, max, and median values.
  - 4 categorical features with allowed categories and empirical category counts:
    - `Gender`: `["Female", "Male"]`
    - `Occupation`: 11 categories
    - `BMI Category`: `["Normal", "Obese", "Overweight"]`
    - `Sleep Disorder`: `["Insomnia", "None", "Sleep Apnea"]`

### Step 3: Unit & Integration Tests (`tests/test_preprocessing.py`)
- Created `tests/test_preprocessing.py` and `pytest.ini`.
- Command: `.\venv\Scripts\pytest.exe -v`
- Output:
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0 -- D:\BurnoutLens\venv\Scripts\python.exe
  cachedir: .pytest_cache
  rootdir: D:\BurnoutLens
  configfile: pytest.ini
  testpaths: tests
  plugins: anyio-4.15.1
  collecting ... collected 10 items

  tests/test_preprocessing.py::test_clean_data_shape_and_columns PASSED    [ 10%]
  tests/test_preprocessing.py::test_target_counts PASSED                   [ 20%]
  tests/test_preprocessing.py::test_bmi_normalization PASSED               [ 30%]
  tests/test_preprocessing.py::test_sleep_disorder_values PASSED           [ 40%]
  tests/test_preprocessing.py::test_blood_pressure_split_and_drop PASSED   [ 50%]
  tests/test_preprocessing.py::test_assert_no_leakage PASSED               [ 60%]
  tests/test_preprocessing.py::test_immutability_and_raw_integrity PASSED  [ 70%]
  tests/test_preprocessing.py::test_preprocessor_unfitted_and_transform PASSED [ 80%]
  tests/test_preprocessing.py::test_duplicate_group_id PASSED              [ 90%]
  tests/test_preprocessing.py::test_compute_lifestyle_score_hand_checkable PASSED [100%]

  ============================= 10 passed in 3.12s ==============================
  ```

### Step 4: Preprocessing Specification
- Authored `reports/preprocessing_spec.md` with:
  - Phase 1 baseline vs. Phase 2 design comparison table.
  - Explicit documentation of the 13 -> 12 feature count reduction.
  - Full schema summary and hand-checkable calculations.
  - Statements labeled [VERIFIED], [HYPOTHESIS], and [FUTURE].

---

## 2026-10-04 - Phase 3: Supervised Model Training & Honest Evaluation

### Scope & Constraints
- Evaluate models honestly using Phase 2 package with strict leakage prevention.
- Same 5 baseline models without hyperparameter tuning: Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP Classifier.
- No duplicate rows deleted; evaluate under random stratified vs. group-isolated protocols.
- Zero final models serialized/saved to disk.
- Manual push policy respected (zero git push commands executed).

### Step 0: Verification & Health Check
- `git status; git branch` -> Branch `main`, working tree clean.
- `pytest -v` -> 10 passed in 4.12s.

### Step 1: Implementation of Modeling and Evaluation Code
- Created `src/burnoutlens/modeling.py`:
  - `LABEL_TO_INT = {"Low": 0, "Medium": 1, "High": 2}` and `INT_TO_LABEL`.
  - `get_models()` returning exact baseline hyperparameters.
  - `make_pipeline(model)` wrapping `build_preprocessor()` and model.
- Created `src/burnoutlens/evaluation.py`:
  - `compute_metrics()`: Accuracy, macro-F1, per-class metrics, confusion matrix.
  - `majority_class_baseline()`: Evaluates majority-class predictor ('Low' at 37.70% accuracy).
  - `run_protocol_a()`: Random stratified 80/20 holdout + train twin measurement.
  - `run_protocol_b()`: Grouped holdout across 10 random seeds with verified 0% group overlap.
  - `run_protocol_c()`: Grouped 5-fold cross-validation (seed 42 + 5 repeated seeds 0-4) with pooled OOF predictions.
  - `mcnemar_test()`: Exact McNemar test using `scipy.stats.binomtest` on discordant pairs.
- Created `scripts/run_supervised_eval.py`:
  - Automated execution pipeline for all protocols, significance tests, model selection, and reporting.

### Step 2: Label Consistency Findings
- Total records: 374 across 132 unique `dup_group` profiles.
- Groups with conflicting target labels: **0 groups (0 rows)**.
- Theoretical accuracy ceiling: **100.00% (374 / 374)**.

### Step 3: Evaluation Protocols Execution
- **Protocol A (Random Stratified 80/20)**:
  - Test set size: 75 samples.
  - Identical-input twin records in train set: **58 out of 75 (77.3%)**.
  - Accuracies: LR: 0.9067, DT: 0.8933, RF: 0.9200, XGB: 0.9467, MLP: 0.9467.
- **Protocol B (Grouped Holdout, 10 Seeds 0–9)**:
  - All repeats verified with 0 group overlap between train and test.
  - Accuracies (Mean ± Std): LR: 0.9292 ± 0.0219, DT: 0.9132 ± 0.0330, RF: 0.9412 ± 0.0270, XGB: 0.9319 ± 0.0331, MLP: 0.9132 ± 0.0211.
- **Protocol C (Grouped 5-Fold CV)**:
  - Seed 42: LR: 0.9411 ± 0.0138, DT: 0.9199 ± 0.0233, RF: 0.9412 ± 0.0135, XGB: 0.9598 ± 0.0190, MLP: 0.9144 ± 0.0137.
  - 5 Repeated Seeds (0–4) Macro-F1:
    - **Logistic Regression**: `0.9499 ± 0.0060` (Mean Acc: `0.9513 ± 0.0057`)
    - **XGBoost**: `0.9451 ± 0.0078` (Mean Acc: `0.9470 ± 0.0073`)
    - **Random Forest**: `0.9414 ± 0.0092` (Mean Acc: `0.9427 ± 0.0090`)
    - **Decision Tree**: `0.9230 ± 0.0110` (Mean Acc: `0.9246 ± 0.0104`)
    - **MLP Classifier**: `0.9129 ± 0.0082` (Mean Acc: `0.9166 ± 0.0080`)
  - Majority-class baseline: Accuracy = 0.3770, Macro-F1 = 0.1825.

### Step 4: McNemar Statistical Significance
- Top 2 models (Logistic Regression vs. XGBoost):
  - $b$ (LR correct, XGB incorrect): 2
  - $c$ (LR incorrect, XGB correct): 9
  - Discordant pairs $n = 11$, two-sided $p$-value = `0.0654` (Not significant at $\alpha = 0.05$).

### Step 5: Model Selection Decision
- **Selected Candidate**: **Logistic Regression**
- **Reasoning**: Achieved the highest repeated Macro-F1 (0.9499) under Protocol C, with performance statistically indistinguishable from XGBoost (0.9451, within 1 standard deviation $\le 0.0060$, McNemar $p = 0.0654$). Chosen for maximum interpretability, direct feature weight explainability, and minimal deployment overhead.

### Step 6: Artifacts Generated
- `reports/supervised_results.csv`: Complete metrics matrix across models and protocols.
- `reports/supervised_results.md`: Full evaluation report with tables, per-class metrics, confusion matrices, and limitations.
- `reports/figures/model_comparison_protocols.png`: Grouped bar chart comparing model accuracy across Protocols A, B, and C.
- `reports/figures/confusion_matrix_selected.png`: Confusion matrix for Logistic Regression under Protocol C pooled OOF predictions.

### Step 7: Tests Execution
- Created `tests/test_supervised.py` verifying:
  - Zero group overlap in Protocol B and C.
  - Preprocessor strictly fitted on train subset without data leakage.
  - Explicit target integer encoding `{"Low": 0, "Medium": 1, "High": 2}`.
  - Zero forbidden feature leakage with `assert_no_leakage`.
- Command: `.\venv\Scripts\pytest.exe -v` -> **14 passed in 3.38s**.




