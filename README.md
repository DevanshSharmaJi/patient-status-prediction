# Patient Risk Stratification — ML Pipeline

A production-grade machine learning pipeline for predicting patient enrollment status (Active vs Inactive) from clinical data. Built for healthcare organizations to identify at-risk patients before they disengage from care.

---

## Problem

In community health settings, patients disengage from care without warning. By the time a clinic identifies an inactive patient, it may be too late for timely outreach. This pipeline uses historical visit behavior and insurance data to flag patients likely to become inactive — enabling proactive intervention.

---

## Results (44,413 patients)

| Model | Active Recall | Inactive Recall | Notes |
|---|---|---|---|
| Baseline (threshold 0.50) | 0.96 | 0.61 | Strong Active, weak Inactive |
| Threshold 0.65 ✅ Deployed | 0.94 | 0.78 | Best clinical balance |

**Primary objective met:** 94% of Active patients correctly identified.  
**Secondary gain:** Inactive Recall improved from 61% to 78% via threshold tuning — 246 additional at-risk patients flagged per test cycle.

---

## Key Findings

| Feature | Importance | Clinical Meaning |
|---|---|---|
| number_of_visit | 0.54 | Visit frequency is the strongest predictor |
| days_since_visit | 0.17 | Recency matters almost as much as frequency |
| lob_missing | 0.08 | Missing insurance data is itself a risk signal |
| overdue (180+ days) | 0.07 | Overdue flag adds signal beyond raw days |
| Age | 0.04 | Age matters but less than visit behavior |

---

## Data

Data is not included in this repository due to HIPAA compliance.

The pipeline expects a pandas DataFrame with the following columns:

```
patient_id       int64       — dropped before modeling
last_visit_date  datetime    — extracted to days_since_visit
number_of_visit  int64       — used directly
Status           bool        — target (True=Active, False=Inactive)
LOB              object      — line of business (nullable)
Program          object      — program name (nullable, ~91% missing)
date_of_birth    object      — extracted to Age
```

---

## Project Structure

```
ml-patient-risk/
├── README.md          — this file
├── requirements.txt   — dependencies
├── config.py          — all settings and thresholds
├── pipeline.py        — main training pipeline
└── experiments.py     — threshold and class weight experiments
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Configure paths and settings
# Edit config.py with your data source and parameters

# Run the full pipeline
python pipeline.py

# Run threshold experiments
python experiments.py
```

---

## Pipeline Steps

1. **Feature Engineering** — extract Age from DOB, Days Since Visit from last visit date, flag missing values, one-hot encode LOB and Program
2. **Class Imbalance Check** — dataset is 84% Active / 16% Inactive
3. **Stratified Train/Test Split** — 80/20, class ratio preserved
4. **Cross Validation** — 5-fold Stratified K-Fold, scoring F1
5. **Hyperparameter Tuning** — RandomizedSearchCV (20 iterations)
6. **Threshold Tuning** — raised from 0.50 to 0.65 for better Inactive Recall
7. **Feature Importance** — identify clinical drivers of patient status

---

## Why Threshold 0.65?

Default ML classifiers use a 0.50 probability threshold. For this clinical use case:

- **Missing an Active patient (FN)** = patient loses care = dangerous
- **False alarm on Inactive patient (FP)** = unnecessary outreach = manageable

Raising the threshold to 0.65 means the model must be 65%+ confident a patient is Active before labeling them so. This catches more truly Active patients at the cost of some false alarms — the right tradeoff for healthcare.

---

## Tech Stack

- Python 3.10+
- scikit-learn — modeling, evaluation, tuning
- pandas / numpy — data processing
- imbalanced-learn — SMOTE (available but not used in final model)

---

## Author

Data Engineer at a healthcare technology company.  
Built for clinical data pipelines serving community health centers.
