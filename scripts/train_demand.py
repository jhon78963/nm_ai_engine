#!/usr/bin/env python3
"""
Entrena modelos de demanda por producto.

Estrategia de selección automática:
  - Serie con MUCHOS ceros (demanda intermitente)  → Croston/SBA
  - Serie con historial largo y ventas regulares    → Prophet
  - Sin datos suficientes                           → fallback heurístico (2 u/día)

Umbrales de decisión:
  - intermittency_threshold = 0.70 → si ≥ 70 % de los días son cero, usar Croston
  - min_prophet_days         = MIN_DEMAND_HISTORY_DAYS (del .env)
  - min_prophet_sales        = MIN_DEMAND_TOTAL_SALES  (del .env)
  - Croston siempre se entrena si hay al menos 1 venta

Archivos generados:
  - models/demand/product_{id}.joblib   (Prophet o CrostonModel)
  - models/demand/manifest.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from prophet import Prophet

from app.core.config import get_settings
from app.ml_models.croston import CrostonModel
from scripts.common import (
    DEMAND_DAILY_FILE,
    DEMAND_MANIFEST_FILE,
    DEMAND_MODELS_DIR,
    ensure_directories,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Porcentaje mínimo de días-cero para catalogar la serie como intermitente
INTERMITTENCY_THRESHOLD = 0.70


def _prepare_series(group: pd.DataFrame) -> pd.DataFrame:
    """Rellena días faltantes con 0 para una serie diaria continua."""
    df = group[["sale_date", "quantity"]].copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values("sale_date")
    full_range = pd.date_range(df["sale_date"].min(), df["sale_date"].max(), freq="D")
    filled = (
        df.set_index("sale_date")["quantity"]
        .reindex(full_range, fill_value=0)
        .rename_axis("ds")
        .reset_index()
    )
    filled.columns = ["ds", "y"]
    filled["y"] = filled["y"].astype(float)
    return filled


def _is_intermittent(series: pd.DataFrame) -> bool:
    zero_ratio = (series["y"] == 0).mean()
    return float(zero_ratio) >= INTERMITTENCY_THRESHOLD


def _train_croston(series: pd.DataFrame, alpha: float = 0.15) -> CrostonModel:
    model = CrostonModel(alpha=alpha, use_sba=True)
    model.fit(series["y"].tolist())
    return model


def _train_prophet(series: pd.DataFrame) -> Prophet:
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=len(series) >= 365,
        seasonality_mode="multiplicative",
    )
    model.fit(series)
    return model


def main() -> None:
    settings = get_settings()
    ensure_directories()

    if not DEMAND_DAILY_FILE.exists():
        raise FileNotFoundError(
            f"No existe {DEMAND_DAILY_FILE}. Ejecuta primero export_training_data.py"
        )

    demand_df = pd.read_csv(DEMAND_DAILY_FILE)
    if demand_df.empty:
        logger.warning("demand_daily.csv está vacío. No se entrenó ningún modelo.")
        return

    manifest: dict = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "products": {},
    }
    trained_croston = 0
    trained_prophet = 0
    skipped = 0

    for product_id, group in demand_df.groupby("product_id"):
        product_id = int(product_id)
        total_sales = int(group["quantity"].sum())
        history_days = (
            pd.to_datetime(group["sale_date"].max())
            - pd.to_datetime(group["sale_date"].min())
        ).days + 1

        if total_sales == 0:
            skipped += 1
            continue

        series = _prepare_series(group)
        intermittent = _is_intermittent(series)

        # ── Decisión de algoritmo ─────────────────────────────────────────
        if intermittent:
            # Siempre entrenar Croston para series intermitentes (no hay mínimo de días)
            try:
                model = _train_croston(series)
                model_path = DEMAND_MODELS_DIR / f"product_{product_id}.joblib"
                joblib.dump(model, model_path)
                manifest["products"][str(product_id)] = {
                    "algorithm": "croston_sba",
                    "history_days": history_days,
                    "total_sales": total_sales,
                    "zero_ratio": round(float((series["y"] == 0).mean()), 2),
                    "daily_rate": round(model.fitted_demand_rate, 4),
                }
                trained_croston += 1
                logger.info(
                    "Croston/SBA entrenado product_id=%d (%d días, %d u., tasa=%.3f u/día)",
                    product_id,
                    history_days,
                    total_sales,
                    model.fitted_demand_rate,
                )
            except Exception as exc:
                skipped += 1
                logger.warning("Croston falló product_id=%d: %s", product_id, exc)
        else:
            # Prophet requiere mínimo de historial y ventas
            if history_days < settings.min_demand_history_days:
                skipped += 1
                continue
            if total_sales < settings.min_demand_total_sales:
                skipped += 1
                continue

            try:
                model = _train_prophet(series)
                model_path = DEMAND_MODELS_DIR / f"product_{product_id}.joblib"
                joblib.dump(model, model_path)
                manifest["products"][str(product_id)] = {
                    "algorithm": "prophet",
                    "history_days": history_days,
                    "total_sales": total_sales,
                    "zero_ratio": round(float((series["y"] == 0).mean()), 2),
                }
                trained_prophet += 1
                logger.info(
                    "Prophet entrenado product_id=%d (%d días, %d u.)",
                    product_id,
                    history_days,
                    total_sales,
                )
            except Exception as exc:
                skipped += 1
                logger.warning("Prophet falló product_id=%d: %s", product_id, exc)

    with DEMAND_MANIFEST_FILE.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    logger.info(
        "Demanda: %d Croston/SBA + %d Prophet entrenados, %d omitidos",
        trained_croston,
        trained_prophet,
        skipped,
    )


if __name__ == "__main__":
    main()
