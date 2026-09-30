# =====================================================================
# PIPELINE CONFIGURATION
# =====================================================================
# All settings in one place — edit here, nothing else needs changing.

# ── Data Source ───────────────────────────────────────────────────────
# Replace with your actual data path (S3, local, etc.)
DATA_PATH = "path/to/your/patient_data.csv"

# ── Column Names ──────────────────────────────────────────────────────
# Update these if your column names differ
COLUMNS = {
    "patient_id":      "patient_id",
    "last_visit_date": "last_visit_date",
    "visit_count":     "number_of_visit",
    "status":          "Status",
    "lob":             "LOB",
    "program":         "Program",
    "dob":             "date_of_birth",
}

# ── Feature Engineering ───────────────────────────────────────────────
OVERDUE_DAYS       = 180        # days since last visit to flag as overdue
DEFAULT_LOB        = "Unknown"  # fill value for missing LOB
DEFAULT_PROGRAM    = "Unknown"  # fill value for missing Program

# ── Model Settings ────────────────────────────────────────────────────
TEST_SIZE          = 0.2        # 80/20 train/test split
RANDOM_STATE       = 42         # reproducibility seed
CV_FOLDS           = 5          # stratified k-fold splits
CV_SCORING         = "f1"       # metric to optimize

# ── Hyperparameter Search Space ───────────────────────────────────────
PARAM_DIST = {
    "n_estimators":      [50, 100, 200, 300],
    "max_depth":         [3, 4, 5, 6],
    "min_samples_split": [2, 5, 10],
    "class_weight":      ["balanced", None],
}
N_ITER = 20                     # random combinations to try

# ── Threshold ─────────────────────────────────────────────────────────
# Raised from 0.50 to 0.65 for better Inactive Recall
# See experiments.py for full threshold analysis
DECISION_THRESHOLD = 0.65

# ── Target Labels ─────────────────────────────────────────────────────
TARGET_NAMES = ["Inactive", "Active"]   # 0=Inactive, 1=Active
