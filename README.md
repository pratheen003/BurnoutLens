# BurnoutLens - Lifestyle-based burnout risk assessment (not a medical diagnosis)

> **IMPORTANT DISCLAIMER**: BurnoutLens is a lifestyle-based risk assessment tool, **NOT a medical diagnostic system**. It provides exploratory analytics based on self-reported lifestyle and sleep habits and does not provide clinical diagnosis, medical advice, or treatment plans.

## Project Status

This repository is currently under **recovery and verification** (Phase 1). Historical results referenced from earlier project deliverables are baseline historical benchmarks and are **not final verified results**. All preprocessing steps, models, and evaluation figures are being rigorously audited and verified from the primary source code.

## Dataset Information

- **Dataset Source**: Sleep Health and Lifestyle Dataset (from Kaggle).
- **Redistribution Terms**: The raw CSV dataset is not committed to this repository pending license and redistribution verification.
- **Data Setup**: Download `Sleep_health_and_lifestyle_dataset.csv` from Kaggle and place it into `data/raw/` (or copy from the verified project archive). The file will be treated as read-only.

## Project Structure

```
BurnoutLens/
├── data/
│   └── raw/              # Raw data files (uncommitted / read-only)
├── notebooks/            # Exploratory and historical research notebooks
├── src/
│   └── recovery/         # Data audit and faithful baseline reproduction scripts
├── reports/              # Audit reports and reproduction findings
├── models/               # Model artifacts (post-verification)
├── tests/                # Unit and integration tests
├── docs/                 # Extended documentation
├── requirements.txt      # Pinned environment dependencies
└── BUILD_LOG.md          # Chronological execution log
```
