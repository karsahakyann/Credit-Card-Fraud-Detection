"""Central configuration: paths, random seed, split and CV settings.

Every experiment imports from here so that results are reproducible and
comparable across notebooks and phases.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_CSV = RESULTS_DIR / "metrics.csv"

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED = 42

# --- Splits ------------------------------------------------------------------
TEST_SIZE = 0.2          # held-out fraction for both split strategies
N_CV_FOLDS = 5           # stratified k-fold used for cross-validated metrics

# --- Dataset expectations (used for sanity checks) ---------------------------
EXPECTED_ROWS = 284_807
EXPECTED_COLUMNS = 31    # Time, V1..V28, Amount, Class
EXPECTED_FRAUDS = 492

TARGET = "Class"
# Only Time and Amount are on their original scale; V1-V28 are already
# PCA-transformed and roughly standardised.
UNSCALED_FEATURES = ["Time", "Amount"]
