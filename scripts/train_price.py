#!/usr/bin/env python3
"""
Entrena un pipeline Ridge global para optimización de precios.

Lee data/price_training.csv y guarda:
  - models/price/price_ridge.joblib  (ColumnTransformer + Ridge)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import get_settings
from scripts.common import PRICE_MODEL_FILE, PRICE_TRAINING_FILE, ensure_directories

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "current_cost",
    "sales_last_month",
    "category",
    "product_age_days",
    "days_since_last_sale",
    "total_sales_all_time",
    "current_stock",
]
TARGET_COLUMN = "current_sale_price"


def main() -> None:
    settings = get_settings()
    ensure_directories()

    if not PRICE_TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"No existe {PRICE_TRAINING_FILE}. Ejecuta primero export_training_data.py"
        )

    price_df = pd.read_csv(PRICE_TRAINING_FILE)
    if len(price_df) < settings.min_price_training_rows:
        logger.warning(
            "Solo %d filas (mínimo %d). No se entrenó el modelo de precios.",
            len(price_df),
            settings.min_price_training_rows,
        )
        return

    features = price_df[FEATURE_COLUMNS]
    target = price_df[TARGET_COLUMN]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                [
                    "current_cost",
                    "sales_last_month",
                    "product_age_days",
                    "days_since_last_sale",
                    "total_sales_all_time",
                    "current_stock",
                ],
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["category"],
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    pipeline.fit(features, target)
    joblib.dump(pipeline, PRICE_MODEL_FILE)

    score = pipeline.score(features, target)
    logger.info(
        "Ridge entrenado con %d productos (R²=%.3f) → %s",
        len(price_df),
        score,
        PRICE_MODEL_FILE,
    )


if __name__ == "__main__":
    main()
