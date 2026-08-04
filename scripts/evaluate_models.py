#!/usr/bin/env python3
"""
Evaluación out-of-sample de los modelos ML entrenados.

Protocolo:
  - Demand (Prophet): walk-forward holdout de 30 días por producto.
    Entrena con todos los días EXCEPTO los últimos 30, predice esos 30,
    compara con ventas reales → MAPE, MAE, RMSE por producto y global.
  - Price  (Ridge): leave-one-out por producto → MAE, RMSE, MAPE out-of-sample.
  - Baseline de referencia: naive "2 u/día" para demanda y "precio actual" para precio.

Resultados guardados en:
  - reports/evaluation_demand.csv
  - reports/evaluation_price.csv
  - reports/evaluation_summary.txt
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml_models.croston import CrostonModel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = ROOT / "reports"
DEMAND_DAILY_FILE = ROOT / "data" / "demand_daily.csv"
PRICE_TRAINING_FILE = ROOT / "data" / "price_training.csv"
HOLDOUT_DAYS = 30
BASELINE_DAILY_RATE = 2  # unidades/día (fallback heurístico actual)

PRICE_FEATURE_COLUMNS = [
    "current_cost",
    "sales_last_month",
    "category",
    "product_age_days",
    "days_since_last_sale",
    "total_sales_all_time",
    "current_stock",
]
PRICE_TARGET = "current_sale_price"
PRICE_NUM_COLS = [
    "current_cost",
    "sales_last_month",
    "product_age_days",
    "days_since_last_sale",
    "total_sales_all_time",
    "current_stock",
]


# ---------------------------------------------------------------------------
# Helpers de métricas
# ---------------------------------------------------------------------------

def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAPE ignorando días donde actual == 0 (evita división por cero)."""
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """SMAPE — más estable que MAPE cuando hay muchos ceros."""
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


# ---------------------------------------------------------------------------
# Evaluación de demanda (Prophet)
# ---------------------------------------------------------------------------

def _prepare_series(group: pd.DataFrame) -> pd.DataFrame:
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


def _train_prophet_local(train: pd.DataFrame) -> Prophet:
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=len(train) >= 365,
        seasonality_mode="multiplicative",
    )
    model.fit(train)
    return model


INTERMITTENCY_THRESHOLD = 0.70


def _is_intermittent(series: pd.DataFrame) -> bool:
    return float((series["y"] == 0).mean()) >= INTERMITTENCY_THRESHOLD


def evaluate_demand(demand_df: pd.DataFrame, min_train_days: int = 30) -> pd.DataFrame:
    """
    Walk-forward holdout por producto (últimos 30 días como test).
    Compara Croston/SBA vs Prophet vs Baseline para cada producto.
    """
    records = []
    products = demand_df["product_id"].unique()
    logger.info("Evaluando demanda en %d productos (holdout %d días)…", len(products), HOLDOUT_DAYS)

    for i, pid in enumerate(sorted(products), 1):
        group = demand_df[demand_df["product_id"] == pid].copy()
        series = _prepare_series(group)
        total_sales = int(series["y"].sum())

        if total_sales == 0:
            records.append(_demand_record(pid, "no_sales", series))
            continue

        if len(series) < min_train_days + HOLDOUT_DAYS:
            records.append(_demand_record(pid, "skipped_short", series))
            continue

        train = series.iloc[: -HOLDOUT_DAYS]
        test  = series.iloc[-HOLDOUT_DAYS :]
        actual = test["y"].values

        intermittent = _is_intermittent(series)

        # ── Croston/SBA ───────────────────────────────────────────────
        try:
            croston = CrostonModel(alpha=0.15, use_sba=True)
            croston.fit(train["y"].tolist())
            rate = croston.predict_daily_rate()
            croston_pred = np.full(HOLDOUT_DAYS, rate, dtype=float)
            croston_ok = True
        except Exception as exc:
            logger.warning("Croston falló product_id=%d: %s", pid, exc)
            croston_pred = np.full(HOLDOUT_DAYS, BASELINE_DAILY_RATE, dtype=float)
            croston_ok = False

        # ── Prophet ──────────────────────────────────────────────────
        prophet_pred = None
        if len(train) >= min_train_days:
            try:
                m = _train_prophet_local(train)
                future = m.make_future_dataframe(periods=HOLDOUT_DAYS, freq="D")
                fc = m.predict(future)
                prophet_pred = np.maximum(0, fc.tail(HOLDOUT_DAYS)["yhat"].values)
            except Exception as exc:
                logger.warning("Prophet falló product_id=%d: %s", pid, exc)

        # ── Baseline ──────────────────────────────────────────────────
        baseline_pred = np.full(HOLDOUT_DAYS, BASELINE_DAILY_RATE, dtype=float)

        # ── Mejor modelo para este producto ──────────────────────────
        if intermittent:
            best_pred = croston_pred
            best_algo = "croston_sba"
        elif prophet_pred is not None:
            best_pred = prophet_pred
            best_algo = "prophet"
        else:
            best_pred = croston_pred
            best_algo = "croston_sba_fallback"

        rec: dict = {
            "product_id": pid,
            "status": "evaluated",
            "is_intermittent": intermittent,
            "algorithm_used": best_algo,
            "history_days": len(series),
            "total_sales": total_sales,
            "holdout_actual_total": int(actual.sum()),
            # Mejor modelo (el que usa el sistema)
            "best_mae":   round(_mae(actual,   best_pred), 3),
            "best_rmse":  round(_rmse(actual,  best_pred), 3),
            "best_mape":  round(_mape(actual,  best_pred), 1),
            "best_smape": round(_smape(actual, best_pred), 1),
            "best_pred_total": round(float(best_pred.sum()), 1),
            # Croston individual
            "croston_mape":  round(_mape(actual, croston_pred), 1) if croston_ok else None,
            "croston_smape": round(_smape(actual, croston_pred), 1) if croston_ok else None,
            "croston_mae":   round(_mae(actual,  croston_pred), 3) if croston_ok else None,
            # Prophet individual
            "prophet_mape":  round(_mape(actual, prophet_pred), 1) if prophet_pred is not None else None,
            "prophet_smape": round(_smape(actual, prophet_pred), 1) if prophet_pred is not None else None,
            "prophet_mae":   round(_mae(actual,  prophet_pred), 3) if prophet_pred is not None else None,
            # Baseline
            "baseline_mape": round(_mape(actual, baseline_pred), 1),
            "baseline_mae":  round(_mae(actual,  baseline_pred), 3),
        }
        records.append(rec)

        if i % 20 == 0:
            logger.info("  %d/%d productos evaluados…", i, len(products))

    return pd.DataFrame(records)


def _demand_record(pid: int, status: str, series: pd.DataFrame) -> dict:
    return {
        "product_id": pid,
        "status": status,
        "is_intermittent": None,
        "algorithm_used": None,
        "history_days": len(series),
        "total_sales": int(series["y"].sum()),
        "holdout_actual_total": None,
        "best_mae": None, "best_rmse": None, "best_mape": None, "best_smape": None, "best_pred_total": None,
        "croston_mape": None, "croston_smape": None, "croston_mae": None,
        "prophet_mape": None, "prophet_smape": None, "prophet_mae": None,
        "baseline_mape": None, "baseline_mae": None,
    }


# ---------------------------------------------------------------------------
# Evaluación de precios (Ridge Leave-One-Out)
# ---------------------------------------------------------------------------

def _build_price_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), PRICE_NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["category"]),
        ]
    )
    return Pipeline([("pre", preprocessor), ("reg", Ridge(alpha=1.0))])


def evaluate_price(price_df: pd.DataFrame) -> pd.DataFrame:
    """Leave-One-Out cross-validation sobre el dataset de precios."""
    logger.info("Evaluando precios con Leave-One-Out (%d productos)…", len(price_df))

    X = price_df[PRICE_FEATURE_COLUMNS]
    y = price_df[PRICE_TARGET].values

    preds = np.empty(len(price_df))
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        pipe = _build_price_pipeline()
        pipe.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = pipe.predict(X.iloc[test_idx])

    price_df = price_df.copy()
    price_df["pred_price"] = np.maximum(0, preds).round(2)
    price_df["error"] = price_df["pred_price"] - price_df[PRICE_TARGET]
    price_df["abs_error"] = price_df["error"].abs()
    price_df["pct_error"] = (price_df["abs_error"] / price_df[PRICE_TARGET].replace(0, np.nan) * 100).round(1)

    return price_df[
        [
            "product_id",
            "category",
            "current_cost",
            PRICE_TARGET,
            "pred_price",
            "error",
            "abs_error",
            "pct_error",
            "product_age_days",
            "days_since_last_sale",
            "total_sales_all_time",
            "current_stock",
        ]
    ]


# ---------------------------------------------------------------------------
# Resumen ejecutivo
# ---------------------------------------------------------------------------

def _demand_summary(df: pd.DataFrame) -> str:
    evaluated = df[df["status"] == "evaluated"]
    skipped = len(df) - len(evaluated)
    if evaluated.empty:
        return "Sin productos evaluados."

    best_mape     = evaluated["best_mape"].dropna()
    best_smape    = evaluated["best_smape"].dropna()
    baseline_mape = evaluated["baseline_mape"].dropna()
    croston_mape  = evaluated["croston_mape"].dropna()
    prophet_mape  = evaluated["prophet_mape"].dropna()

    n_intermittent = evaluated["is_intermittent"].sum()
    n_regular      = len(evaluated) - n_intermittent

    # Calidad del modelo seleccionado
    excellent  = (best_mape <= 20).sum()
    good       = ((best_mape > 20) & (best_mape <= 40)).sum()
    acceptable = ((best_mape > 40) & (best_mape <= 60)).sum()
    weak       = (best_mape > 60).sum()

    mejora = 0.0
    if baseline_mape.mean() > 0:
        mejora = (baseline_mape.mean() - best_mape.mean()) / baseline_mape.mean() * 100

    lines = [
        "═" * 60,
        "  EVALUACIÓN DEMANDA — Croston/SBA + Prophet vs Baseline",
        "═" * 60,
        f"  Productos evaluados  : {len(evaluated)} / {len(df)} ({skipped} omitidos)",
        f"  Series intermitentes : {n_intermittent}  →  Croston/SBA",
        f"  Series regulares     : {n_regular}  →  Prophet",
        "",
        "  ── Sistema híbrido (modelo seleccionado por producto) ──",
        f"  MAPE promedio        : {best_mape.mean():.1f} %  (mediana {best_mape.median():.1f} %)",
        f"  SMAPE promedio       : {best_smape.mean():.1f} %",
        f"  MAE promedio         : {evaluated['best_mae'].dropna().mean():.3f} u/día",
        "",
        "  ── Calidad por producto (modelo usado) ──",
        f"  Excelente  (≤ 20%)   : {excellent} productos",
        f"  Bueno      (21–40%)  : {good} productos",
        f"  Aceptable  (41–60%)  : {acceptable} productos",
        f"  Débil      (> 60%)   : {weak} productos",
        "",
        "  ── Comparación de algoritmos ──",
        f"  MAPE Baseline (2u/d) : {baseline_mape.mean():.1f} %",
        f"  MAPE Croston/SBA     : {croston_mape.mean():.1f} %  (n={len(croston_mape)})",
        f"  MAPE Prophet         : {prophet_mape.mean():.1f} %  (n={len(prophet_mape)})",
        f"  MAPE Sistema híbrido : {best_mape.mean():.1f} %",
        f"  Mejora vs Baseline   : {mejora:+.1f} % {'✓' if mejora > 0 else '✗'}",
        "═" * 60,
    ]
    return "\n".join(lines)


def _price_summary(df: pd.DataFrame) -> str:
    mae = df["abs_error"].mean()
    rmse = np.sqrt((df["error"] ** 2).mean())
    mape = df["pct_error"].dropna().mean()
    median_err = df["abs_error"].median()

    within_5 = (df["pct_error"] <= 5).sum()
    within_10 = (df["pct_error"] <= 10).sum()
    within_20 = (df["pct_error"] <= 20).sum()

    lines = [
        "═" * 60,
        "  EVALUACIÓN PRECIO — Ridge Leave-One-Out",
        "═" * 60,
        f"  Productos evaluados  : {len(df)}",
        "",
        "  ── Métricas out-of-sample ──",
        f"  MAE                  : S/ {mae:.2f} por producto",
        f"  Error mediano        : S/ {median_err:.2f}",
        f"  RMSE                 : S/ {rmse:.2f}",
        f"  MAPE                 : {mape:.1f} %",
        "",
        "  ── Tolerancia del error ──",
        f"  Error ≤ 5%           : {within_5} / {len(df)} productos ({within_5/len(df)*100:.0f}%)",
        f"  Error ≤ 10%          : {within_10} / {len(df)} productos ({within_10/len(df)*100:.0f}%)",
        f"  Error ≤ 20%          : {within_20} / {len(df)} productos ({within_20/len(df)*100:.0f}%)",
        "═" * 60,
    ]
    return "\n".join(lines)


def _interpretation(demand_df: pd.DataFrame, price_df: pd.DataFrame) -> str:
    evaluated = demand_df[demand_df["status"] == "evaluated"]
    prophet_mape = evaluated["prophet_mape"].dropna().mean() if not evaluated.empty else float("nan")
    price_mape = price_df["pct_error"].dropna().mean()

    best_mape = evaluated["best_mape"].dropna().mean() if not evaluated.empty else float("nan")

    if best_mape <= 20:
        d_label = "ALTA PRECISIÓN — apto para decisiones de compra"
    elif best_mape <= 40:
        d_label = "PRECISIÓN BUENA — útil como referencia de restock"
    elif best_mape <= 60:
        d_label = "PRECISIÓN ACEPTABLE — usar con criterio adicional"
    else:
        d_label = "PRECISIÓN LIMITADA — complementar con juicio experto"

    if price_mape <= 10:
        p_label = "ALTA — el modelo replica bien los precios del catálogo"
    elif price_mape <= 20:
        p_label = "BUENA — error contenido, útil para sugerencias de precio"
    else:
        p_label = "MODERADA — validar precio sugerido antes de aplicar"

    lines = [
        "",
        "  ── Interpretación para tesis / reporte ejecutivo ──",
        f"  Demanda  : {d_label}",
        f"           MAPE = {best_mape:.1f} %  → margen de error medio por producto en 30 días",
        f"  Precio   : {p_label}",
        f"           MAPE = {price_mape:.1f} %  → diferencia media entre precio sugerido y catálogo",
        "",
        "  Nota: las métricas son out-of-sample (datos no vistos durante",
        "  el entrenamiento), válidas para reportar en resultados de tesis.",
        "═" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DEMAND_DAILY_FILE.exists() or not PRICE_TRAINING_FILE.exists():
        raise FileNotFoundError(
            "Faltan archivos CSV. Ejecuta primero: make export  (o scripts/export_training_data.py)"
        )

    demand_df = pd.read_csv(DEMAND_DAILY_FILE)
    price_df = pd.read_csv(PRICE_TRAINING_FILE)

    # ── Evaluar demanda ───────────────────────────────────────────────────
    eval_demand = evaluate_demand(demand_df)
    eval_demand.to_csv(REPORTS_DIR / "evaluation_demand.csv", index=False)

    # ── Evaluar precios ───────────────────────────────────────────────────
    eval_price = evaluate_price(price_df)
    eval_price.to_csv(REPORTS_DIR / "evaluation_price.csv", index=False)

    # ── Resumen legible ───────────────────────────────────────────────────
    demand_sum = _demand_summary(eval_demand)
    price_sum = _price_summary(eval_price)
    interp = _interpretation(eval_demand, eval_price)
    full_report = "\n".join([demand_sum, "", price_sum, interp])

    summary_path = REPORTS_DIR / "evaluation_summary.txt"
    summary_path.write_text(full_report, encoding="utf-8")

    print(full_report)
    logger.info("Reportes guardados en %s/", REPORTS_DIR)


if __name__ == "__main__":
    main()
