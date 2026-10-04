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

---

## 2026-10-04 - Phase 3.1: Reproducibility Patch & Random-CV Control (Protocol D)

### Step 1: Investigation & Explanation of Phase 3 Metric Change
- **Context**: In Phase 3, the initial output stream of `scripts/run_supervised_eval.py` reported 5-repeat results of XGBoost Acc 0.9556 / F1 0.9547, RF F1 0.9366, DT F1 0.9213, selecting XGBoost. In the clean re-run across seeds 0–4, the numbers became XGBoost Acc 0.9470 / F1 0.9451, RF 0.9414, DT 0.9230, and Logistic Regression achieved 0.9499 Macro-F1 (Rank 1), selecting Logistic Regression.
- **Analysis**:
  - The script and evaluation module on disk deterministically produce XGBoost Acc 0.9470 / F1 0.9451, RF 0.9414, and DT 0.9230 when evaluated using `StratifiedGroupKFold(n_splits=5, shuffle=True)` across seeds 0–4.
  - In the initial task stream, Logistic Regression was omitted from the repeat loop or an alternate seed list/aggregation scheme was evaluated during script drafting.
  - Because the exact seed configuration and aggregation script state that generated the initial 0.9556/0.9547 console stream was not preserved in Git or persistent files on disk, the exact mechanism is formally classified as **UNVERIFIED**.

### Step 2: Determinism Check
- Executed `scripts/run_supervised_eval.py` twice consecutively in the environment.
- Computed SHA-256 hash of `reports/supervised_results.csv` after each run:
  - **Run 1 Hash**: `91C4915403415AF9918EF4A945B3A5E0CD5431E102AB16801904C3E8D2DF8DDB`
  - **Run 2 Hash**: `91C4915403415AF9918EF4A945B3A5E0CD5431E102AB16801904C3E8D2DF8DDB`
  - **Identical**: **True** (Bit-for-bit determinism verified).

### Step 3: Protocol D (Fair Random-CV Contamination Test) Implementation
- Implemented `run_protocol_d()` in `src/burnoutlens/evaluation.py`:
  - Uses non-grouped `StratifiedKFold(n_splits=5, shuffle=True)` across identical seeds 0–4.
  - Uses the same clean Pipeline (fitted on train folds only) and same 5 baseline models.
  - Splitting does not require groups; tracks per-fold duplicate contamination rate if groups are provided.
- Measured Contamination Rate:
  - **Average Twin Rows per Fold**: `60.76 ± 3.64` rows out of 74–75 validation samples (**81.2% contamination**).
- Head-to-Head Comparison (Protocol D vs. Protocol C):
  - **Logistic Regression**: Protocol C F1 `0.9499 ± 0.0060` vs. Protocol D F1 `0.9606 ± 0.0032` (Diff: `+0.0107`, > 1 std)
  - **Decision Tree**: Protocol C F1 `0.9230 ± 0.0110` vs. Protocol D F1 `0.9566 ± 0.0049` (Diff: `+0.0336`, > 3 std)
  - **Random Forest**: Protocol C F1 `0.9414 ± 0.0092` vs. Protocol D F1 `0.9751 ± 0.0029` (Diff: `+0.0337`, > 3 std)
  - **XGBoost**: Protocol C F1 `0.9451 ± 0.0078` vs. Protocol D F1 `0.9759 ± 0.0064` (Diff: `+0.0309`, > 3 std)
  - **MLP Classifier**: Protocol C F1 `0.9129 ± 0.0082` vs. Protocol D F1 `0.9192 ± 0.0056` (Diff: `+0.0064`, within std)

### Step 4: Report Updates (`reports/supervised_results.md` & CSV)
- **Hypothesis Verdict**: Re-evaluated strictly based on D vs. C. Verdict is **SUPPORTED** because random-CV scores are higher than grouped-CV scores across all 5 models, and the difference exceeds 1 standard deviation for 4 out of 5 models (DT by +3.36%, RF by +3.37%, XGBoost by +3.09%, LR by +1.07%).
- **Protocol A Disclaimer**: Explicitly noted that Protocol A is a single, unrepeated 75-row split (1 sample = 1.33%) and not directly comparable to averaged protocols.
- **Model Selection Text**: Top model is `Logistic Regression` (`0.9499 ± 0.0060`) and runner-up is `XGBoost` (`0.9451 ± 0.0078`). Difference is `0.0048`, which is within the 1-std bound (`0.0060`). Logistic Regression is selected without self-comparison.
- **Baseline Fix**: Removed "Phase 1 Historical" label from the computed majority-class baseline.
- **Expanded Limitations**: Added notes on 0 conflicting groups (100% theoretical ceiling), cohort limited to 132 unique profiles, and inability of exact-duplicate grouping to eliminate near-duplicates.

### Step 5: Unit Tests
- Added `test_protocol_d_split_and_pipeline` in `tests/test_supervised.py`:
  - Verifies `run_protocol_d` operates without `groups`.
  - Verifies `StratifiedKFold` splits do not require groups while pipeline preprocessor fits strictly on train folds.
- Executed `pytest -v`: **15 passed in 4.31s**.

---

## 2026-10-04 - Phase 4: Unsupervised Behavioral Analytics (K-Means & PCA)

### Step 0: Pre-Flight Verification
- Confirmed git status clean, branch `main`.
- Active virtual environment verified.
- Confirmed all 15 existing tests pass (`pytest -v`).

### Step 1: Implementation of Analytics Module & Script
- Created `src/burnoutlens/analytics.py` providing:
  - `prepare_clustering_data()`: Extracts 12 `INPUT_FEATURES`, verifies zero leakage via `assert_no_leakage()`, fits clean `ColumnTransformer` (OneHotEncoder + StandardScaler), yielding dense (374, 27) matrix.
  - `evaluate_kmeans_k_range()`: Computes inertia and silhouette scores for $k \in [2, 8]$ using `random_state=42` and `n_init=10`. Implements pre-specified selection rule.
  - `evaluate_cluster_stability()`: Evaluates sensitivity to duplicate rows (132 unique rows vs full data ARI) and seed stability across 10 random seeds (mean pairwise ARI).
  - `fit_pca()`: Evaluates first 10 principal components, cumulative variance, top 5 positive and negative feature loadings for PC1 and PC2, and 2-D coordinates.
  - `compute_cluster_profiles()`: Computes descriptive numeric statistics (mean, median, std) and categorical modes/shares for measured features; strictly post-hoc burnout/lifestyle metrics; crosstabulation and chi-square test of independence.
- Created `scripts/run_clustering_pca.py` orchestrating end-to-end execution and report generation.

### Step 2: Choosing $k$ (Inertia & Silhouette Evaluation)
- Evaluated $k \in [2, 8]$ on dense 27-dimensional feature space:
  - $k=2$: Inertia 2881.59, Silhouette 0.2911, Sizes: 156 / 218
  - $k=3$: Inertia 2153.01, Silhouette 0.3559, Sizes: 128 / 67 / 179 (Historical $k$)
  - $k=4$: Inertia 1756.15, Silhouette 0.4108, Sizes: 149 / 32 / 67 / 126
  - $k=5$: Inertia 1417.87, Silhouette 0.4750, Sizes: 35 / 126 / 149 / 32 / 32
  - $k=6$: Inertia 1211.25, Silhouette 0.4692, Sizes: 93 / 35 / 32 / 121 / 32 / 61
  - $k=7$: Inertia 971.78, Silhouette 0.5067, Sizes: 32 / 81 / 53 / 65 / 35 / 76 / 32
  - $k=8$: Inertia 804.16, Silhouette 0.5454, Sizes: 21 / 106 / 32 / 33 / 69 / 32 / 39 / 42
- **Pre-specified Rule Application**: Global best silhouette is $k=8$ (`0.5454`). Historical $k=3$ silhouette is `0.3559`. Difference is `0.1895` (> 0.02 continuity threshold).
- **Selected $k$**: **$k=8$**.

### Step 3: Sensitivity & Cluster Stability
- **Sensitivity to Duplicate Rows**:
  - $k=8$: ARI between full-data clustering and 132 unique-profile clustering is **0.8795**.
  - $k=3$: ARI between full-data clustering and 132 unique-profile clustering is **0.9730**.
- **Algorithmic Seed Stability (10 Random Seeds, 0 to 9)**:
  - $k=8$: Mean pairwise ARI across 45 seed combinations is **0.7634 ± 0.1337** (range: [0.5878, 1.0000]). Verdict: **Stable**.
  - $k=3$: Mean pairwise ARI is **1.0000 ± 0.0000**.

### Step 4: Principal Component Analysis (PCA)
- **Variance Capture**:
  - PC1: `32.95%`
  - PC2: `27.10%`
  - Combined 2-PC: `60.04%` (PC3: 16.44%, PC4: 6.91%, PC5: 4.70%, ..., PC10: 0.89%; Cumulative 10-PC: 96.44%).
  - **Lossy Projection Caveat**: The 2-D plot omits `39.96%` of the total variance; Euclidean distances in 2-D do not represent full 27-D geometric distances.
- **Top Feature Loadings**:
  - **PC1 Positive**: `Diastolic BP` (+0.521), `Systolic BP` (+0.503), `Age` (+0.331), `Physical Activity Level` (+0.238), `BMI Category_Overweight` (+0.205)
  - **PC1 Negative**: `BMI Category_Normal` (-0.217), `Sleep Disorder_None` (-0.201), `Sleep Duration` (-0.137), `Quality of Sleep` (-0.101), `Gender_Male` (-0.088)
  - **PC2 Positive**: `Quality of Sleep` (+0.564), `Sleep Duration` (+0.515), `Age` (+0.371), `Physical Activity Level` (+0.190), `Gender_Female` (+0.120)
  - **PC2 Negative**: `Heart Rate` (-0.409), `Gender_Male` (-0.120), `Occupation_Doctor` (-0.078), `Sleep Disorder_Insomnia` (-0.074), `Occupation_Salesperson` (-0.052)

### Step 5: Cluster Profiles & Post-Hoc Significance
- Derived neutral descriptive human labels based strictly on measured values (e.g., "short sleep, high activity, elevated BP, overweight BMI, frequent sleep apnea").
- Post-hoc contingency analysis of Cluster vs. `Burnout Risk`:
  - $\chi^2 = 502.14$, $\text{df} = 14$, $p = 3.26 \times 10^{-98}$ ($p < 0.05$).
  - Caveat explicitly noted: nominal p-value is artificially inflated because duplicate rows violate sample independence.

### Step 6: Generated Artifacts & Reports
- `reports/clustering_pca.md`: Full analytics report covering setup, $k$ selection, stability, PCA, profiles, crosstab, historical comparison, and limitations.
- `reports/cluster_profiles.json`: Lightweight export for future application serving (cluster id, size, human label, key profile stats, post-hoc metrics; zero serialized models).
- `reports/figures/`:
  - `elbow_silhouette.png`
  - `pca_scree.png`
  - `pca_clusters.png`
  - `pca_by_risk.png`

### Step 7: Tests Execution
- Created `tests/test_analytics.py`:
  - `test_clustering_input_no_forbidden_columns`: Confirms no target/forbidden leakage.
  - `test_cluster_sizes_and_labels`: Confirms clusters sum to 374 and labels span `range(k)`.
  - `test_pca_variance_properties`: Confirms explained variance $\le 1.0$ and cumulative variance is monotonic.
  - `test_clustering_reproducibility`: Confirms identical seed reproduces identical assignments.
- Executed `pytest -v`: **19 passed in 7.42s** (15 existing + 4 new).

---

## 2026-10-04 - Phase 4.1: Clustering Report Patch & Behavioral-Only Variant (Variant B)

### Step 1: Report Corrections & Demographic Confounding Identification
- **Removed "~0.35" Historical Silhouette**: The original exploratory notebook (`data_understanding.ipynb`) never computed or reported silhouette scores; classified formally as **UNVERIFIED**.
- **Hypothesis Labeling**: Labeled the explanation that "LabelEncoder caused the cluster shift" as **HYPOTHESIS (Untested)**.
- **Terminology Alignment**:
  - Replaced "physiological states" with "measured lifestyle and health variables".
  - Replaced "robust cluster boundaries" with the measured algorithmic seed ARI: **mean 0.7634, min 0.5878**.
  - Replaced claims of nominal chi-square hypothesis confirmation with the explicit caveat that the chi-square test is **descriptive only** due to duplicated non-independent rows violating i.i.d. assumptions.
- **Demographic Confounding Plain Finding**:
  - Documented that 6 of 8 clusters (75.0%) in Variant A are $\ge 90\%$ dominated by a single gender or occupation.
  - Specifically, **Cluster 2 and Cluster 3 share near-identical demographic and health status profiles** (100% Female Nurses, Overweight BMI, high prevalence of Sleep Apnea, average Blood Pressure 140/95 mmHg) but have **diametrically opposed burnout risk** (Cluster 2 is 100% High Risk; Cluster 3 is 100% Low Risk).
  - Identified that the risk divergence is driven purely by sleep duration and quality (Cluster 2 averages 6.07h sleep at quality 6.0; Cluster 3 averages 8.09h sleep at quality 9.0), proving that Variant A conflates demographic dataset structure with lifestyle behaviors.

### Step 2: Dual Silhouette Evaluation on Unique Rows (Variant A)
- Evaluated K-Means for $k \in [2, 8]$ on both the full 374 rows and the 132 unique-profile rows:
  - $k=2$: Full Silhouette = `0.2911`, Unique Silhouette = `0.2510`
  - $k=3$: Full Silhouette = `0.3559`, Unique Silhouette = `0.3151`
  - $k=4$: Full Silhouette = `0.4108`, Unique Silhouette = `0.3452`
  - $k=5$: Full Silhouette = `0.4750`, Unique Silhouette = `0.3838`
  - $k=6$: Full Silhouette = `0.4692`, Unique Silhouette = `0.3931`
  - $k=7$: Full Silhouette = `0.5067`, Unique Silhouette = `0.4061`
  - $k=8$: Full Silhouette = `0.5454`, Unique Silhouette = `0.4059`

### Step 3: Variant B (Behavioral-Only Clustering)
- **Feature Set**: Exactly 5 continuous features (`Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Daily Steps`, `Heart Rate`) scaled via `StandardScaler`. All demographic, occupational, and health-status variables (`Gender`, `Occupation`, `Age`, `BMI Category`, `Sleep Disorder`, `Blood Pressure`) strictly excluded.
- **Zero Leakage**: Verified with `assert_no_leakage`.
- **Dual Silhouette Evaluation**:
  - $k=2$: Full = `0.4159`, Unique = `0.3929`, Inertia (Full: 1165.86, Unique: 448.64)
  - $k=3$: Full = `0.4887`, Unique = `0.4710`, Inertia (Full: 812.33, Unique: 315.98)
  - $k=4$: Full = `0.5278`, Unique = `0.4519`, Inertia (Full: 571.05, Unique: 236.46)
  - $k=5$: Full = `0.5689`, Unique = `0.5146`, Inertia (Full: 413.25, Unique: 168.36) **[CHOSEN B]**
  - $k=6$: Full = `0.5636`, Unique = `0.4381`, Inertia (Full: 330.19, Unique: 145.62)
  - $k=7$: Full = `0.5546`, Unique = `0.4683`, Inertia (Full: 260.37, Unique: 114.05)
  - $k=8$: Full = `0.6338`, Unique = `0.4912`, Inertia (Full: 206.25, Unique: 95.64)
- **Selection Decision**: Evaluated on 132-unique silhouette. $k=5$ achieves the global best unique silhouette (`0.5146`). $k=3$ unique silhouette is `0.4710` (diff `0.0436` > 0.02 continuity threshold). **$k=5$ chosen**.
- **Metrics for Chosen $k=5$**:
  - Cluster Sizes (374 Full Rows): `[178, 34, 22, 108, 32]`
  - Unique-vs-Full ARI: **1.0000**
  - Seed Stability (Seeds 0–9): Mean pairwise ARI = **0.8224 ± 0.1654** (min: `0.5147`, max: `1.0000`).
  - PCA Variance: PC1 = `48.33%`, PC2 = `35.37%`, Cumulative 2-PC = **83.70%** (PC1 loadings: Quality of Sleep +0.616, Sleep Duration +0.585, Heart Rate -0.485; PC2 loadings: Physical Activity +0.689, Daily Steps +0.686).
  - Demographic Purity: Only 2 of 5 clusters (40.0%) are $\ge 90\%$ single gender or occupation (vs 75.0% in Variant A).
  - Post-hoc Contingency: $\chi^2 = 290.73$, $\text{df} = 8$, $p = 3.86 \times 10^{-58}$ (descriptive only).

### Step 4: Comparative Synthesis (Variant A vs. Variant B)
- Documented full comparison table in `reports/clustering_pca.md` contrasting the 27-D occupational archetypes against the 5-D behavioral cohorts.
- Updated `reports/cluster_profiles.json` to store both `"variant_a_full_features"` and `"variant_b_behavioral_only"` profiles under dedicated keys.

- Executed `pytest -v`: **22 passed in 8.25s** (19 existing + 3 new).

---

## 2026-10-04 - Phase 5: Model Explainability (SHAP & Permutation Importance)

### Step 0: Pre-Flight Verification
- Verified git status clean, branch `main`.
- Active virtual environment verified.
- Confirmed all 22 existing tests pass (`pytest -v`).

### Step 1: Implementation of Explainability Module & Pipeline
- Created `src/burnoutlens/explain.py`:
  - `map_transformed_to_original()`: Maps all 27 one-hot and scaled features from `preprocessor.get_feature_names_out()` to the 12 original `INPUT_FEATURES`, verifying that every transformed column maps to exactly one original feature.
  - `fit_explainability_pipeline()`: Fits the selected Phase 3 Logistic Regression pipeline (`build_preprocessor()` + `LogisticRegression(max_iter=1000, random_state=42)`) on all 374 rows in memory (`assert_no_leakage` verified). Sets SHAP background to the 132 unique-profile rows (transformed) with `max_samples=132` to avoid subsampling.
  - `aggregate_shap_to_original()`: Linearly sums one-hot SHAP contributions to collapse from 27 transformed features back to the 12 original features while strictly preserving local and global additivity.
  - `compute_global_shap_importance()`: Computes global mean |SHAP| overall and per class (Low, Medium, High).
  - `compute_permutation_importance()`: Evaluates held-out generalization importance using `StratifiedGroupKFold` (`groups=dup_group`, 5 folds, 10 repeats, scoring: `macro-F1`) by permuting original columns before the pipeline.
  - `explain_row()`: Validates single input records against `reports/feature_schema.json` (raising `ValueError` for unseen categories or out-of-range numeric values), infers class and probabilities, and outputs top 5 original features by |contribution| for the predicted class with signed values and direction (`pushes toward <class>` / `pushes away`).
- Created `scripts/run_explainability.py` orchestrating end-to-end execution, figure generation, and report compilation.

### Step 2: Global Explainability Findings
- **Installed Library Versions**: `shap` = 0.52.0, `xgboost` = 3.4.1, `scikit-learn` = 1.9.1.
- **Output Shapes**:
  - Transformed matrix: `(374, 27)`
  - SHAP LinearExplainer values: `(374, 27, 3)`
  - Aggregated original SHAP values: `(374, 12, 3)`
  - Base values: `(374, 3)`
- **Additivity Verification**:
  - Decision scale: `decision_function` (log-odds scale for multiclass Logistic Regression).
  - Base value + $\sum \phi_i$ matches `decision_function` output across all classes within $3.55 \times 10^{-15}$ tolerance.
- **Top 5 Global Features by SHAP**:
  1. `Quality of Sleep`: Mean |SHAP| = `1.4057`
  2. `Sleep Duration`: Mean |SHAP| = `0.9553`
  3. `Daily Steps`: Mean |SHAP| = `0.7911`
  4. `Heart Rate`: Mean |SHAP| = `0.6573`
  5. `Gender`: Mean |SHAP| = `0.6325`
- **Class-Specific Drivers**:
  - **Low Risk**: Driven by higher `Quality of Sleep` (2.1086), lower `Daily Steps` (1.1867), lower `Heart Rate` (0.9860), and `Gender` (0.9487).
  - **Medium Risk**: Driven by `Sleep Duration` (0.8173), `Sleep Disorder` (0.6073), `Physical Activity Level` (0.5306), `Systolic BP` (0.3834).
  - **High Risk**: Heavily penalized by poor `Quality of Sleep` (1.7379), reduced `Sleep Duration` (1.4330), elevated `Daily Steps` (1.0530), `Physical Activity Level` (0.7669), and elevated `Heart Rate` (0.7123).
- **Held-Out Permutation Importance (Macro-F1 Drops)**:
  - Top features: `Quality of Sleep` (+0.1719 ± 0.0473), `Sleep Duration` (+0.1421 ± 0.0256), `Gender` (+0.0578 ± 0.0531), `Heart Rate` (+0.0558 ± 0.0362), `Daily Steps` (+0.0433 ± 0.0314).
  - **Spearman Rank Correlation (LR SHAP vs. Permutation)**: **$\rho = 0.9371$** ($p = 6.99 \times 10^{-6}$).
- **Tree Cross-Check (XGBoost)**:
  - When initializing `TreeExplainer(model, data=X_unique_trans)` with background data, SHAP 0.52.0 raised:
    `NotImplementedError: Categorical split is not yet supported. You can still use TreeExplainer with feature_perturbation="tree_path_dependent".`
  - Version-aware execution using `feature_perturbation="tree_path_dependent"` succeeded, ranking `Quality of Sleep` (1.243), `Heart Rate` (0.939), `Sleep Duration` (0.766), `Gender` (0.359), and `Daily Steps` (0.317) as top 5.
  - **Spearman Rank Correlation (LR SHAP vs. XGBoost SHAP)**: **$\rho = 0.8252$** ($p = 9.51 \times 10^{-4}$).

### Step 3: Local Explanations
- Generated worked local explanations for 3 real dataset rows:
  - **Low Risk Example** (Accountant, 44, F, Sleep 7.9h, Quality 8): $P(\text{Low}) = 92.3\%$. Pushed toward Low by Quality of Sleep (+1.728), Systolic BP (+1.075), Gender (+0.963).
  - **Medium Risk Example** (Software Engineer, 27, M, Sleep 6.1h, Quality 6): $P(\text{Medium}) = 69.2\%$. Pushed away by short sleep (-1.153), pushed toward Medium by Age (+0.726) and Occupation (+0.622).
  - **High Risk Example** (Nurse, 28, F, Sleep 6.2h, Quality 6, Steps 10000, BP 140/95): $P(\text{High}) = 97.8\%$. Pushed toward High by Daily Steps (+2.621), Quality of Sleep (+1.933), Sleep Duration (+1.816).

### Step 4: Generated Artifacts & Figures
- `reports/explainability.md`: Full explainability report.
- `reports/feature_importance.json`: Global SHAP, permutation ranking, correlations, and worked local examples.
- `reports/figures/shap_global_importance.png`: Horizontal bar chart of global mean |SHAP| values.
- `reports/figures/shap_by_class.png`: Grouped bar chart showing feature importance across Low, Medium, and High classes.
- `reports/figures/permutation_importance.png`: Bar chart with error bars showing held-out macro-F1 drops.
- `reports/figures/local_example_high.png`: Local explanation bar chart for sample high-risk prediction.

### Step 5: Unit Tests Execution
- Created `tests/test_explain.py` with 5 tests:
  - `test_every_transformed_column_maps_to_exactly_one_original_feature`
  - `test_shap_additivity` (verifies additivity on decision_function scale within $10^{-6}$)
  - `test_explain_row_output_contains_only_input_features_no_forbidden`
  - `test_explain_row_raises_on_unseen_category_and_out_of_range_numeric`
  - `test_explainability_determinism`
- Executed `pytest -v`: **27 passed in 12.50s** (22 existing + 5 new).

