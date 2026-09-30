"""Every protocol number lives here. Change only BEFORE the protocol commit."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("ELLIPTIC_DIR", "data/elliptic"))
OUT_DIR = Path("results")
SCORES_DIR = OUT_DIR / "scores"          # per-node test scores (not committed, large)
INDUCTIVE_DIR = OUT_DIR / "inductive"
RAW_DIR = OUT_DIR / "raw"                # CSVs (committed, handed to Ryan)
PLOTS_DIR = OUT_DIR / "plots"

SEEDS = list(range(10))
TRAIN_STEPS = list(range(1, 25))
VAL_STEPS = list(range(25, 35))
TEST_STEPS = list(range(35, 50))

MODELS = ["xgboost", "gcn", "graphsage"]
FEATURE_SETS = {"all165": 165, "local93": 93}   # first N of the 165 feature columns
PRIMARY_FS = "all165"

XGB = dict(n_estimators=500, max_depth=6, learning_rate=0.1,
           subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30)
GNN = dict(hidden=128, layers=2, dropout=0.5, lr=1e-3, weight_decay=5e-4,
           max_epochs=200, patience=20)
UNDIRECTED = True            # edges symmetrized for GCN / GraphSAGE

# shifts
N_RESAMPLES = 200
ACTIVITY_LEVELS = [0.25, 0.50, 0.75]      # share of low-degree nodes
PREVALENCE_MULTS = [0.5, 1.0, 2.0, 4.0]   # x original illicit share
TEMPORAL_WINDOWS = {"35-42": (35, 42), "43-49": (43, 49)}

# Grouping rule: "transaction connectivity" = in-degree = number of parent transactions
# within the same time step (known when the transaction is created).
# Fixed bins chosen from TRAIN-period counts before any outcome: 0 / 1 / 2 or more.
GROUP_CUTS = [0, 1]     # in-degree <= 0 -> group 0, <= 1 -> group 1, else group 2
GROUP_LABELS = {0: "in0", 1: "in1", 2: "in2plus"}
MIN_ILLICIT = 30                          # below this a group/window is "underpowered"

# inductive test
INDUCTIVE_HOLDOUT = 0.20

# statistics
N_BOOT = 2000
CONTRASTS = [("graphsage", "xgboost"), ("gcn", "xgboost"), ("graphsage", "gcn")]
PRIMARY_CONTRAST = ("graphsage", "xgboost")
