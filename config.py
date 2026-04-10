"""
Central configuration for Cross-Asset OFI Research Project.
All tuneable parameters live here so notebooks and modules stay clean.
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
FIG_DIR      = OUTPUT_DIR / "figures"
TABLE_DIR    = OUTPUT_DIR / "tables"

# ── Assets ────────────────────────────────────────────────────────────
TICKERS        = ["NIFTY", "BANKNIFTY", "HDFCBANK", "RELIANCE", "INFY"]
SIGNAL_ASSET   = "NIFTY"                       # cross-asset signal source
TARGET_ASSETS  = ["BANKNIFTY", "HDFCBANK", "RELIANCE", "INFY"]

# ── OFI parameters ────────────────────────────────────────────────────
OFI_HORIZONS           = [1, 5, 15, 30, 60]    # trailing bars
FORWARD_RETURN_HORIZONS = [1, 5, 15, 30, 60]   # prediction horizons
DEFAULT_FWD_HORIZON    = 5                      # primary target horizon (bars)

# ── Walk-forward validation ───────────────────────────────────────────
WALK_FORWARD_TRAIN_MONTHS = 1        # shorter window for ~60-day data
WALK_FORWARD_TEST_MONTHS  = 1

# ── Model hyper-parameters ────────────────────────────────────────────
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
XGB_PARAMS   = dict(max_depth=3, n_estimators=100, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)

# ── Market hours (IST) ───────────────────────────────────────────────
MARKET_OPEN  = "09:15"
MARKET_CLOSE = "15:30"

# ── Misc ──────────────────────────────────────────────────────────────
OUTLIER_SIGMA = 5        # winsorisation threshold
RANDOM_SEED   = 42
