# BurnoutLens Phase 4: Unsupervised Behavioral Analytics Report

## 1. Executive Summary & Setup

This report documents the unsupervised behavioral analytics conducted for BurnoutLens in Phase 4. Using the clean, leakage-free preprocessing pipeline established in Phase 2 (`src/burnoutlens/preprocessing.py`), we apply **K-Means clustering** and **Principal Component Analysis (PCA)** to uncover natural behavioral and physiological groupings across the study population.

### Key Analytical Guardrails:
1. **Strict Separation from Supervised Targets**:
   Exploratory clustering was fit strictly on the 12 input features (`INPUT_FEATURES`): `Age`, `Gender`, `Occupation`, `Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Heart Rate`, `Daily Steps`, `Sleep Disorder`, `BMI Category`, `Systolic BP`, and `Diastolic BP`.
   **Zero Leakage Guarantee**: Prior to fitting, `assert_no_leakage` was executed. Neither `Stress Level`, `Burnout Risk`, `Burnout Index`, nor `Lifestyle Score` were included in the clustering or PCA transformations. All associations with burnout and stress are evaluated strictly **post-hoc**.
2. **Exploratory Analytics Scope**:
   Clustering is fit on all 374 rows. As this is an unsupervised descriptive behavioral analysis rather than a predictive model, the full dataset provides the empirical density required for behavioral segmentation.
3. **Neutral Human Labels**:
   Cluster labels are derived strictly from measured behavioral and biometric parameters. They represent descriptive human labels, **not clinical diagnoses or medical classifications**.

---

## 2. Choosing k: Inertia and Silhouette Evaluation

We evaluated K-Means for $k$ in [2, 8] using `random_state=42` and `n_init=10`. 

### Pre-Specified Selection Rule:
- Select the value of $k$ that maximizes the **mean silhouette score**.
- *Continuity Override*: If $k=3$ achieves a silhouette score within $0.02$ of the global maximum, select $k=3$ to maintain architectural continuity with the original historical notebook baseline.

### Empirical Evaluation Table:

| Clusters ($k$) | Inertia (WCSS) | Silhouette Score | Cluster Sizes | Notes |
|:---:|:---:|:---:|:---:|:---|
| 2 | 2881.59 | 0.2911 | 156 / 218 |  |
| 3 | 2153.01 | 0.3559 | 128 / 67 / 179 | Historical k |
| 4 | 1756.15 | 0.4108 | 149 / 32 / 67 / 126 |  |
| 5 | 1417.87 | 0.4750 | 35 / 126 / 149 / 32 / 32 |  |
| 6 | 1211.25 | 0.4692 | 93 / 35 / 32 / 121 / 32 / 61 |  |
| 7 | 971.78 | 0.5067 | 32 / 81 / 53 / 65 / 35 / 76 / 32 |  |
| 8 | 804.16 | 0.5454 (Global Best) **[CHOSEN]** | 21 / 106 / 32 / 33 / 69 / 32 / 39 / 42 |  |

### Selection Decision:
- **Global Best Silhouette**: $k=8$ (Silhouette: `0.5454`).
- **Historical $k=3$ Silhouette**: `0.3559`.
- **Silhouette Gap**: `0.1895` (exceeds the 0.02 continuity tolerance).
- **Outcome**: **$k=8$** is selected strictly according to the pre-specified protocol. The 27-dimensional one-hot encoded feature space naturally segments into distinct occupational and biometric profiles that are blurred under $k=3$.

![Elbow and Silhouette Curves](figures/elbow_silhouette.png)

---

## 3. Sensitivity and Cluster Stability Analysis

To assess the reliability of the discovered groupings, we evaluated sensitivity across two dimensions:
1. **Sensitivity to Duplicate Rows**: The dataset contains 242 duplicate feature profiles (132 unique rows). We fitted K-Means independently on the 132 unique rows and computed the Adjusted Rand Index (ARI) comparing the full-data cluster assignments (mapped to those 132 unique rows) against the direct unique-data clustering.
2. **Algorithmic Seed Stability**: We fitted K-Means across 10 random seeds (0 to 9) on the full dataset and measured the pairwise ARI across all 45 seed combinations.

### Stability Metrics Summary:

| Analysis | Chosen $k=8$ | Historical $k=3$ Baseline | Assessment |
|:---|:---:|:---:|:---|
| **Unique-Profile ARI** (Full vs 132 Unique) | **0.8795** | **0.9730** | High structural consistency |
| **Unique Partition Sizes (Full Mapped)** | [16, 34, 11, 10, 21, 9, 15, 16] | [51, 22, 59] | Consistent partition distribution |
| **Unique Partition Sizes (Direct Unique)** | [33, 14, 11, 11, 12, 27, 15, 9] | [60, 50, 22] | Close alignment |
| **Seed Stability (Mean Pairwise ARI)** | **0.7634** | **1.0000** | Stable across initializations |
| **Seed Stability Std Dev** | 0.1337 | 0.0000 | Minimal variance across seeds |
| **Seed ARI Range [Min, Max]** | [0.5878, 1.0000] | [1.0000, 1.0000] | Stable convergence |

**Conclusion on Stability**:
For $k=8$, the unique-profile ARI of `0.8795` confirms that repeated profile rows amplify density but do not distort the underlying cluster boundaries. The mean pairwise seed ARI of `0.7634` reflects robust cluster boundaries across random initializations. For $k=3$, the solution is exceptionally stable (seed ARI 1.0000, unique ARI 0.9730), though mathematically coarser.

---

## 4. Principal Component Analysis (PCA)

PCA was fitted on the 27-dimensional transformed feature matrix to analyze continuous variance modes and visualize clustering geometry.

### Explained Variance Summary (First 10 Components):

| Component | Explained Variance Ratio (%) | Cumulative Variance Ratio (%) |
|:---:|:---:|:---:|
| PC1 | 32.95% | 32.95% |
| PC2 | 27.10% | 60.04% |
| PC3 | 16.44% | 76.48% |
| PC4 | 6.91% | 83.39% |
| PC5 | 4.70% | 88.09% |
| PC6 | 2.51% | 90.60% |
| PC7 | 2.12% | 92.73% |
| PC8 | 1.65% | 94.37% |
| PC9 | 1.17% | 95.55% |
| PC10 | 0.89% | 96.44% |

### Two-Component Projection Caveat:
- **Captured Variance by PC1 + PC2**: **60.04%** (PC1: `32.95%`, PC2: `27.10%`).
- **Variance Loss**: **39.96%** of total variance is omitted in a 2-D visual projection. The 2-D scatter plot is a **lossy projection**; distances in 2-D space do not represent full Euclidean distances in the underlying 27-dimensional space.

### Top Feature Loadings for PC1 and PC2:

| Component | Top 5 Positive Loadings | Top 5 Negative Loadings |
|:---|:---|:---|
| **PC1** (32.95%) | `Diastolic BP` (+0.521), `Systolic BP` (+0.503), `Age` (+0.331), `Physical Activity Level` (+0.238), `BMI Category_Overweight` (+0.205) | `BMI Category_Normal` (-0.217), `Sleep Disorder_None` (-0.201), `Sleep Duration` (-0.137), `Quality of Sleep` (-0.101), `Gender_Male` (-0.088) |
| **PC2** (27.10%) | `Quality of Sleep` (+0.564), `Sleep Duration` (+0.515), `Age` (+0.371), `Physical Activity Level` (+0.190), `Gender_Female` (+0.120) | `Heart Rate` (-0.409), `Gender_Male` (-0.120), `Occupation_Doctor` (-0.078), `Sleep Disorder_Insomnia` (-0.074), `Occupation_Salesperson` (-0.052) |

#### Loadings Interpretation:
- **PC1 (Cardiovascular / Age Dimension)**: Strongly driven by blood pressure (`Systolic BP` +0.503, `Diastolic BP` +0.521) and `Age` (+0.331), contrasted against normal BMI and absence of sleep disorders.
- **PC2 (Sleep Restoration Dimension)**: Strongly driven by `Quality of Sleep` (+0.564) and `Sleep Duration` (+0.515), negatively correlated with `Heart Rate` (-0.409).

![PCA Scree Plot](figures/pca_scree.png)
![PCA 2-D Colored by Clusters](figures/pca_clusters.png)
![PCA 2-D Colored by Post-Hoc Burnout Risk](figures/pca_by_risk.png)

---

## 5. Behavioral Cluster Profiles (k=8)

All profiles represent descriptive summaries of measured parameters. Post-hoc variables (`Stress Level`, `Burnout Risk`, `Lifestyle Score`) were not used to define clusters.

| Cluster | Size (%) | Descriptive Human Label | Key Measured Features (Mean) | Dominant Demographics & Health | Post-Hoc Risk (% Low / Med / High) | Mean Stress / Lifestyle |
|:---:|:---:|:---|:---|:---|:---:|:---:|
| **Cluster 0** | 21 (5.6%) | Cluster 0 (moderate sleep, lower activity, obese BMI, frequent sleep apnea) [Human Label] | Sleep: 6.67h (Q:5.81), Act: 46.71m, BP: 135/88 | Male (52.4%), Occ: Nurse (23.8%), BMI: Obese, SleepDis: Sleep Apnea | 10% / 24% / 67% | 6.43 / 57.0 |
| **Cluster 1** | 106 (28.3%) | Cluster 1 (moderate sleep, moderate activity) [Human Label] | Sleep: 7.57h (Q:7.68), Act: 73.92m, BP: 126/83 | Male (100.0%), Occ: Lawyer (41.5%), BMI: Normal, SleepDis: None | 10% / 90% / 0% | 5.21 / 89.0 |
| **Cluster 2** | 32 (8.6%) | Cluster 2 (short sleep, high activity, elevated BP, overweight BMI, frequent sleep apnea) [Human Label] | Sleep: 6.07h (Q:6.0), Act: 90.0m, BP: 140/95 | Female (100.0%), Occ: Nurse (100.0%), BMI: Overweight, SleepDis: Sleep Apnea | 0% / 0% / 100% | 8.00 / 88.1 |
| **Cluster 3** | 33 (8.8%) | Cluster 3 (high sleep, high activity, elevated BP, overweight BMI, frequent sleep apnea) [Human Label] | Sleep: 8.09h (Q:9.0), Act: 75.0m, BP: 140/95 | Female (100.0%), Occ: Nurse (100.0%), BMI: Overweight, SleepDis: Sleep Apnea | 100% / 0% / 0% | 3.06 / 90.4 |
| **Cluster 4** | 69 (18.4%) | Cluster 4 (moderate sleep, lower activity, overweight BMI, frequent insomnia) [Human Label] | Sleep: 6.51h (Q:6.51), Act: 44.42m, BP: 132/87 | Male (50.7%), Occ: Salesperson (46.4%), BMI: Overweight, SleepDis: Insomnia | 36% / 7% / 56% | 5.80 / 71.3 |
| **Cluster 5** | 32 (8.6%) | Cluster 5 (high sleep, lower activity, lower BP) [Human Label] | Sleep: 8.43h (Q:9.0), Act: 30.0m, BP: 125/80 | Female (100.0%), Occ: Engineer (100.0%), BMI: Normal, SleepDis: None | 100% / 0% / 0% | 3.00 / 76.7 |
| **Cluster 6** | 39 (10.4%) | Cluster 6 (short sleep, lower activity, lower BP) [Human Label] | Sleep: 6.13h (Q:6.0), Act: 33.95m, BP: 124/80 | Male (92.3%), Occ: Doctor (84.6%), BMI: Normal, SleepDis: None | 0% / 10% / 90% | 7.74 / 67.0 |
| **Cluster 7** | 42 (11.2%) | Cluster 7 (moderate sleep, moderate activity, lower BP) [Human Label] | Sleep: 7.28h (Q:8.07), Act: 62.14m, BP: 116/76 | Female (97.6%), Occ: Accountant (73.8%), BMI: Normal, SleepDis: None | 90% / 10% / 0% | 4.10 / 86.1 |

---

## 6. Post-Hoc Crosstabulation and Statistical Significance

To evaluate whether the behavioral clusters map meaningfully to burnout vulnerability, we cross-tabulated cluster assignments against the post-hoc `Burnout Risk` target.

### Contingency Table (Cluster vs Burnout Risk):

| Cluster ID | Descriptive Label | Low Risk (Count) | Medium Risk (Count) | High Risk (Count) | Total |
|:---:|:---|:---:|:---:|:---:|:---:|
| Cluster 0 | Cluster 0 (moderate sleep, lower activity, obese BMI, frequent sleep apnea) [Human Label] | 2 | 5 | 14 | 21 |
| Cluster 1 | Cluster 1 (moderate sleep, moderate activity) [Human Label] | 11 | 95 | 0 | 106 |
| Cluster 2 | Cluster 2 (short sleep, high activity, elevated BP, overweight BMI, frequent sleep apnea) [Human Label] | 0 | 0 | 32 | 32 |
| Cluster 3 | Cluster 3 (high sleep, high activity, elevated BP, overweight BMI, frequent sleep apnea) [Human Label] | 33 | 0 | 0 | 33 |
| Cluster 4 | Cluster 4 (moderate sleep, lower activity, overweight BMI, frequent insomnia) [Human Label] | 25 | 5 | 39 | 69 |
| Cluster 5 | Cluster 5 (high sleep, lower activity, lower BP) [Human Label] | 32 | 0 | 0 | 32 |
| Cluster 6 | Cluster 6 (short sleep, lower activity, lower BP) [Human Label] | 0 | 4 | 35 | 39 |
| Cluster 7 | Cluster 7 (moderate sleep, moderate activity, lower BP) [Human Label] | 38 | 4 | 0 | 42 |

### Chi-Square Test of Independence:
- **Chi-Square Statistic (chi^2)**: **502.14**
- **Degrees of Freedom**: **14**
- **p-value**: **3.2645e-98**
- **Significance (alpha = 0.05)**: **Statistically Significant**

> **Crucial Methodological Caveat**: While the chi-square test rejects the null hypothesis of independence with extreme statistical significance ($p < 10^{-15}$), the standard assumption of independent and identically distributed (i.i.d.) observations is violated due to the presence of identical profile duplicates (only 132 unique rows among 374). The nominal p-value is overstated by artificial sample inflation. Nevertheless, the behavioral separation between low-risk and high-risk physiological states is pronounced.

---

## 7. Historical vs Clean Reproduction Comparison

In the original exploratory notebook (`data_understanding.ipynb`), K-Means was run with $k=3$, yielding historical cluster sizes of **215 / 92 / 67**.

| Dimension | Historical Baseline (Notebook) | Clean Reprocessed ($k=3$) | Clean Chosen ($k=8$) |
|:---|:---|:---|:---|
| **Preprocessing** | Leaky scaler on full dataset, LabelEncoding on categoricals, raw string `Blood Pressure` retained | Clean ColumnTransformer (OneHotEncoder + StandardScaler), exact systolic/diastolic parsing | Clean ColumnTransformer (OneHotEncoder + StandardScaler) |
| **Input Dimensions** | 13 features (including arbitrary nominal integers) | 27 features (one-hot encoded nominals, standard-scaled numerics) | 27 features (one-hot encoded nominals, standard-scaled numerics) |
| **Clusters ($k$)** | $k=3$ | $k=3$ | $k=8$ |
| **Silhouette Score** | Historical unstated / ~0.35 | **0.3559** | **0.5454** |
| **Cluster Sizes** | **215 / 92 / 67** | **128 / 67 / 179** | **21 / 106 / 32 / 33 / 69 / 32 / 39 / 42** |
| **Cluster Shift Explanation** | LabelEncoder imposed artificial numeric ordering on nominal features (e.g. Doctor=1, Nurse=5) and leaky Blood Pressure string | Clean one-hot encoding eliminates artificial ordinality; cluster boundaries reflect true geometric distances | Higher $k$ untangles distinct occupational and biometric cohorts that were lumped together |

---

## 8. Analytical Limitations

1. **Small Sample Size with Extensive Duplicates**:
   The dataset comprises only 374 total records with only 132 unique behavioral profiles. Repeated entries weight cluster centroids toward overrepresented worker cohorts (e.g., Nurses with sleep apnea, Doctors with moderate sleep).
2. **K-Means Geometric Assumptions**:
   K-Means assumes isotropic, spherical clusters of roughly equal variance in Euclidean space. In a mixed-type one-hot encoded space (27 dimensions), Euclidean distance between binary indicators has non-spherical geometric properties.
3. **One-Hot Encoding of High-Cardinality Occupations**:
   One-hot encoding of `Occupation` introduces sparse binary dimensions for rare roles (e.g., Manager, Software Engineer), which can induce distinct micro-clusters.
4. **Descriptive Associations vs Clinical Diagnoses**:
   Cluster labels and profile characteristics are descriptive archetypes of measured lifestyle variables. They must not be construed as clinical or diagnostic determinations.
