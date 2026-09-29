"""Shared defaults. CLI arguments can override every runtime path/tolerance."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "enveda-CASMI26-molecule-id-mass-spectra"
TRAIN_PATH = DEFAULT_DATA_DIR / "train.parquet"
TEST_PATH = DEFAULT_DATA_DIR / "test.parquet"
SAMPLE_SUBMISSION_PATH = DEFAULT_DATA_DIR / "sample_submission.csv"

SEED = 20260929
N_FOLDS = 5
MAX_PPM = 30.0
FALLBACK_PPM = 250.0
PEAK_BIN_WIDTH_DA = 0.02
MIN_RELATIVE_INTENSITY = 0.005
MAX_PEAKS = 256
TOP_K = 25
