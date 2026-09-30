"""
Patient Risk Stratification Pipeline
-------------------------------------
Predicts Active vs Inactive patient status from clinical data.
Designed for community health organizations working with EHR data.

Usage:
    python pipeline.py

Output:
    - Cross validation F1 score and std deviation
    - Best hyperparameters found
    - Classification report on test set
    - Top 10 features driving patient status
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from config import (
    DATA_PATH, COLUMNS, OVERDUE_DAYS,
    DEFAULT_LOB, DEFAULT_PROGRAM,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS, CV_SCORING,
    PARAM_DIST, N_ITER, DECISION_THRESHOLD, TARGET_NAMES,
)


# =====================================================================
# STEP 1 — LOAD DATA
# =====================================================================
def load_data(path: str) -> pd.DataFrame:
    """Load patient data from the configured path."""
    print("📂 Loading data...")
    df = pd.read_csv(path)
    print(f"   Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


# =====================================================================
# STEP 2 — FEATURE ENGINEERING
# =====================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw patient data into model-ready features.

    Steps:
        1. Drop patient ID (no predictive signal)
        2. Convert date columns
        3. Flag missing values (missingness is a clinical signal)
        4. Extract Age from date of birth
        5. Extract Days Since Last Visit + Overdue flag
        6. Encode target (True/False → 1/0)
        7. Fill missing categorical values
        8. One-hot encode LOB and Program
    """
    print("\n🔧 Engineering features...")
    df = df.copy()
    col = COLUMNS

    # 1. Drop ID
    df.drop(columns=[col["patient_id"]], inplace=True, errors="ignore")

    # 2. Convert dates
    df[col["last_visit_date"]] = pd.to_datetime(df[col["last_visit_date"]], errors="coerce")
    df[col["dob"]]             = pd.to_datetime(df[col["dob"]],             errors="coerce")

    # 3. Flag missing values BEFORE filling
    df["lob_missing"] = df[col["lob"]].isna().astype(int)
    df["dob_missing"] = df[col["dob"]].isna().astype(int)

    # 4. Extract Age
    df["Age"] = (pd.Timestamp.now() - df[col["dob"]]).dt.days // 365
    df["Age"] = df["Age"].fillna(df["Age"].median()).astype(int)
    df.drop(columns=[col["dob"]], inplace=True)

    # 5. Days since last visit + overdue flag
    df["days_since_visit"] = (pd.Timestamp.now() - df[col["last_visit_date"]]).dt.days
    df["overdue"]          = (df["days_since_visit"] > OVERDUE_DAYS).astype(int)
    df.drop(columns=[col["last_visit_date"]], inplace=True)

    # 6. Encode target
    df[col["status"]] = df[col["status"]].astype(int)

    # 7. Fill missing categoricals
    df[col["lob"]]     = df[col["lob"]].fillna(DEFAULT_LOB).str.strip()
    df[col["program"]] = df[col["program"]].fillna(DEFAULT_PROGRAM).str.strip()

    # 8. One-hot encode
    df = pd.get_dummies(df, columns=[col["lob"], col["program"]], drop_first=False)

    print(f"   Shape after engineering: {df.shape}")
    return df


# =====================================================================
# STEP 3 — CHECK CLASS BALANCE
# =====================================================================
def check_class_balance(y: pd.Series) -> None:
    """Print class distribution to assess imbalance."""
    print("\n📊 Class Distribution:")
    counts = y.value_counts()
    ratios = y.value_counts(normalize=True).round(2)
    for label, name in enumerate(TARGET_NAMES):
        print(f"   {name}: {counts[label]:,} ({ratios[label]*100:.0f}%)")


# =====================================================================
# STEP 4 — TRAIN / EVALUATE
# =====================================================================
def train_and_evaluate(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Full training pipeline:
        - Stratified train/test split
        - Cross validation baseline
        - Hyperparameter tuning
        - Threshold-adjusted evaluation
        - Feature importance report
    """
    col = COLUMNS
    X   = df.drop(columns=[col["status"]])
    y   = df[col["status"]]

    check_class_balance(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n✅ Split: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test")

    # Cross validation
    print("\n📊 Running Cross Validation...")
    skf      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    baseline = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    scores   = cross_val_score(baseline, X, y, cv=skf, scoring=CV_SCORING)
    print(f"   CV F1 per fold: {scores.round(2)}")
    print(f"   Mean F1:        {scores.mean():.2f}")
    print(f"   Std Dev:        {scores.std():.2f}")

    # Hyperparameter tuning
    print("\n🔍 Running Hyperparameter Search...")
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        cv=skf,
        scoring=CV_SCORING,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"   Best Parameters: {search.best_params_}")
    print(f"   Best CV F1:      {search.best_score_:.2f}")

    # Threshold-adjusted evaluation
    print(f"\n📋 Classification Report (threshold={DECISION_THRESHOLD})")
    y_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= DECISION_THRESHOLD).astype(int)
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

    # Feature importance
    importance = pd.DataFrame({
        "Feature":    X.columns,
        "Importance": search.best_estimator_.feature_importances_,
    }).sort_values("Importance", ascending=False)
    print("🏆 Top 10 Features:")
    print(importance.head(10).to_string(index=False))

    return search.best_estimator_


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print("🚀 Patient Risk Stratification Pipeline")
    print("=" * 60)

    df    = load_data(DATA_PATH)
    df    = engineer_features(df)
    model = train_and_evaluate(df)

    print("\n🎉 Pipeline complete.")
    print(f"   Deploy with threshold: {DECISION_THRESHOLD}")
    print("   Use model.predict_proba(X)[:,1] >= threshold for predictions.")
