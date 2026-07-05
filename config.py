"""
config.py  —  EDIT THIS FILE before running
─────────────────────────────────────────────────────────────────
Set DATASET_NAME to choose which dataset to run on, and set the
matching path below. Each loader accepts a single CSV/JSON file
or a folder.
─────────────────────────────────────────────────────────────────
"""

# ── REQUIRED: which dataset to run on ────────────────────────────────────
# Options: "veremi_extension", "mosaic_replay_bogus", "kaggle_maliciousnode"
DATASET_NAME = "veremi_extension"

# ── REQUIRED: path to the dataset for the selected DATASET_NAME ─────────
#
# Examples:
#   veremi_extension (CSV or JSON, file or folder):
#     r"C:\Users\hansi\OneDrive\Documents\veremi_combined.csv"
#   mosaic_replay_bogus (CSV, file or folder — vehicle-update logs from
#     Iqbal et al. 2022, "Simulating Malicious Attacks on VANETs"):
#     r"C:\Users\hansi\Downloads\mosaic_vanet"
#   kaggle_maliciousnode (single CSV — Kaggle "VANET-MaliciousNode Dataset"):
#     r"C:\Users\hansi\Downloads\vanet_maliciousnode.csv"

DATA_PATH = "/Users/hansikaaaggarwal/.cache/kagglehub/datasets/ivarprudnikov/veremi-extension-data-1-21-gb/versions/1"


# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = "results"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Vehicle setup ─────────────────────────────────────────────────────────────
MAX_VEHICLES       = 200
N_VEHICLES         = 100
ATTACKER_FRACTION  = 0.20

# ── FL training ───────────────────────────────────────────────────────────────
N_ROUNDS           = 20
BEACONS_PER_ROUND  = 50
LOCAL_EPOCHS       = 2
LR                 = 0.01

# ── Privacy parameters ────────────────────────────────────────────────────────
EPS_TOTAL          = 1.00
EPS_MIN            = 0.05
EPS_MAX            = 1.50
CLIP_C             = 1.0
DELTA              = 1e-5

# ── RSU-DT ────────────────────────────────────────────────────────────────────
MIN_TWIN_HISTORY   = 5
