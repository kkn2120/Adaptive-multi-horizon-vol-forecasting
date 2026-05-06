"""
Central configuration for the Volatility Intelligence project.
"""
import os
import torch
import numpy as np
import random

# Random Seed for testing purposes
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497
IBKR_CLIENT_ID = 1
IBKR_TIMEOUT = 30


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

for d in [RAW_DIR, PROC_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)


START_DATE = "2000-01-01"   # extended back to 2000 for more training variety
END_DATE = "2025-12-31"
SPLIT_DATE = "2020-01-01"
TARGET_WINDOW = 22
SPIKE_PERCENTILE = 90

IBKR_BAR_SIZE = "5 mins"
IBKR_DURATION = "1 Y"

CROSS_ASSETS = {
    "TLT": "20Y Treasury", "IEF": "7-10Y Treasury", "SHY": "1-3Y Treasury",
    "HYG": "High Yield", "LQD": "Inv Grade", "JNK": "Junk Bonds",
    "GLD": "Gold", "SLV": "Silver", "USO": "Oil", "UNG": "Nat Gas",
    "UUP": "USD Index", "FXE": "EUR/USD", "FXB": "GBP/USD", "FXY": "JPY/USD",
    "EEM": "EM Equity", "EWJ": "Japan", "FXI": "China", "EWZ": "Brazil",
    "VNQ": "Real Estate", "XLF": "Financials", "XLK": "Tech", "XLE": "Energy",
    "XLV": "Healthcare", "XLU": "Utilities", "DBA": "Agriculture", "COPX": "Copper",
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# P1: VAE
P1_SEQ_LEN = 30
P1_LATENT_DIM = 8
P1_HIDDEN_DIM = 32
P1_DROPOUT = 0.1
P1_EPOCHS = 60
P1_LR = 1e-3
P1_KL_WEIGHT = 0.01
P1_VOL_WEIGHT = 0.1


P2_SEQ_LEN = 60
P2_D_MODEL = 64
P2_HEADS = 4
P2_LAYERS = 2
P2_DROPOUT = 0.15
P2_OUTPUT_DIM = 32
P2_EPOCHS = 40
P2_LR = 5e-4


P4_HIDDEN = 16
P4_OUTPUT = 8
P4_HEADS = 4
P4_DROPOUT = 0.15
P4_EPOCHS = 50
P4_LR = 1e-3
P4_CORR_WINDOW = 60
P4_PATIENCE = 10


ENSEMBLE_HAR_WEIGHT_CALM = 0.30     # less HAR-RV during calm 
ENSEMBLE_HAR_WEIGHT_NORMAL = 0.40   # balanced
ENSEMBLE_HAR_WEIGHT_STRESS = 0.60   # more HAR-RV during stress (more reliable)
ENSEMBLE_HAR_WEIGHT_CRISIS = 0.70   # heavily HAR-RV during crisis

# XGBoost
XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=0.3, reg_lambda=3.0,
    random_state=SEED, verbosity=0,
)

print(f"Config loaded. Device: {DEVICE}, Seed: {SEED}")
