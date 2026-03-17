from pathlib import Path
import torch
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed_all(SEED)
else:
    DEVICE = torch.device("cpu")

# Paths
DATA_DIR       = Path("data")
TAPING_DIR     = DATA_DIR / "taping"
CRACKS_DIR     = DATA_DIR / "cracks"
OUTPUT_DIR     = Path("predictions")
CHECKPOINT_DIR = Path("checkpoints")
REPORT_DIR     = Path("report")

for d in [OUTPUT_DIR, CHECKPOINT_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model
MODEL_NAME = "CIDAS/clipseg-rd64-refined"
IMAGE_SIZE = 352

# Training
BATCH_SIZE    = 4
NUM_EPOCHS    = 20
LR            = 5e-4      
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 3
THRESHOLD     = 0.5

# Split
VAL_SPLIT  = 0.10
TEST_SPLIT = 0.10

PROMPTS = {
    "taping": [
        "segment taping area",
        "segment joint tape",
        "segment drywall seam",
        "segment wall joint",
    ],
    "cracks": [
        "segment crack",
        "segment wall crack",
        "segment surface crack",
        "segment fracture",
    ],
}