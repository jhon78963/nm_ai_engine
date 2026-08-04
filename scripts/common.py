from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEMAND_DAILY_FILE = PROJECT_ROOT / "data" / "demand_daily.csv"
PRICE_TRAINING_FILE = PROJECT_ROOT / "data" / "price_training.csv"
DEMAND_MODELS_DIR = PROJECT_ROOT / "models" / "demand"
PRICE_MODEL_FILE = PROJECT_ROOT / "models" / "price" / "price_ridge.joblib"
DEMAND_MANIFEST_FILE = DEMAND_MODELS_DIR / "manifest.json"


def ensure_directories() -> None:
    DEMAND_DAILY_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEMAND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PRICE_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
