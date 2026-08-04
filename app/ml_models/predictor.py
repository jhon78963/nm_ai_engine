from __future__ import annotations

import logging

import pandas as pd

from app.ml_models.croston import CrostonModel
from app.ml_models.registry import get_registry
from app.ml_models.stock_aging import (
    TIER_AGING,
    TIER_CRITICAL,
    TIER_HIGH,
    DeadStockAssessment,
    StockAgingSignals,
    evaluate_dead_stock,
    project_historical_sales,
)
from app.schemas.prediction_schema import (
    PriceOptimizationRequest,
    PriceOptimizationResponse,
    PurchasePredictionRequest,
    PurchasePredictionResponse,
)

logger = logging.getLogger(__name__)

# Retail ropa: margen mínimo viable sobre costo (ej. costo 15 → piso ~27)
_MIN_MARKUP_RATE: float = 0.80
# Markup objetivo cuando no hay precio de venta de referencia
_DEFAULT_MARKUP_RATE: float = 1.53

# Ajuste sobre el precio actual según rotación mensual (unidades vendidas / 30 días)
_VELOCITY_RULES: tuple[tuple[int, float, str], ...] = (
    (30, 1.03, "Alta rotación: se sugiere un leve incremento (+3%)."),
    (15, 1.0, "Rotación saludable: mantener el precio actual."),
    (5, 0.95, "Rotación moderada: ligero ajuste a la baja (-5%) para estimular demanda."),
    (1, 0.90, "Baja rotación: descuento moderado (-10%) sin comprometer el margen mínimo."),
    (0, 0.85, "Sin ventas recientes: descuento más agresivo (-15%) respetando el piso de margen."),
)

_DAILY_SALES_RATE: int = 2

# Margen mínimo en liquidación (atacasco) — más bajo que el margen operativo normal
_CLEARANCE_MIN_MARKUP: dict[str, float] = {
    TIER_CRITICAL: 0.05,
    TIER_HIGH: 0.15,
    TIER_AGING: 0.30,
}


def _aging_signals_from_price(request: PriceOptimizationRequest) -> StockAgingSignals:
    return StockAgingSignals(
        product_age_days=request.product_age_days,
        days_since_last_sale=request.days_since_last_sale,
        sales_last_month=request.sales_last_month,
        current_stock=request.current_stock,
        total_sales_all_time=request.total_sales_all_time,
    )


def _aging_signals_from_demand(request: PurchasePredictionRequest) -> StockAgingSignals:
    return StockAgingSignals(
        product_age_days=request.product_age_days,
        days_since_last_sale=request.days_since_last_sale,
        sales_last_month=request.sales_last_month,
        current_stock=request.current_stock,
        total_sales_all_time=request.total_sales_all_time,
    )


class PriceOptimizer:
    """
    Singleton — instanciado una sola vez al arrancar FastAPI.

    Usa Ridge (Scikit-learn) si hay modelo entrenado; si no, reglas heurísticas.
    Aplica liquidación adicional para productos atacados (antigüedad + baja rotación).
    """

    _instance: PriceOptimizer | None = None

    def __new__(cls) -> PriceOptimizer:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("PriceOptimizer: Singleton inicializado.")
        return cls._instance

    def predict(self, request: PriceOptimizationRequest) -> PriceOptimizationResponse:
        aging = _aging_signals_from_price(request)
        dead_stock = evaluate_dead_stock(aging)
        minimum_price = self._minimum_price(request.current_cost, dead_stock)
        registry = get_registry()
        pipeline = registry.get_price_pipeline()
        source = "heuristic"

        if pipeline is not None:
            features = pd.DataFrame(
                [
                    {
                        "current_cost": request.current_cost,
                        "sales_last_month": request.sales_last_month,
                        "category": request.category,
                        "product_age_days": request.product_age_days,
                        "days_since_last_sale": request.days_since_last_sale,
                        "total_sales_all_time": request.total_sales_all_time,
                        "current_stock": request.current_stock,
                    }
                ]
            )
            ml_price = float(pipeline.predict(features)[0])
            anchor_price = round(max(minimum_price, ml_price), 2)
            source = "ridge"
        else:
            anchor_price = (
                request.current_sale_price
                if request.current_sale_price > 0
                else round(request.current_cost * (1 + _DEFAULT_MARKUP_RATE), 2)
            )

        adjustment, velocity_summary = self._velocity_adjustment(request.sales_last_month)
        price_after_velocity = anchor_price * adjustment
        price_after_clearance = price_after_velocity * dead_stock.clearance_multiplier
        suggested_price = round(max(minimum_price, price_after_clearance), 2)

        recommendation_summary = self._build_price_summary(
            source=source,
            velocity_summary=velocity_summary,
            dead_stock=dead_stock,
            suggested_price=suggested_price,
            minimum_price=minimum_price,
        )

        expected_margin_increase = round(
            ((suggested_price - minimum_price) / minimum_price * 100)
            if minimum_price > 0
            else 0.0,
            2,
        )

        markup_over_cost = round(
            ((suggested_price - request.current_cost) / request.current_cost * 100)
            if request.current_cost > 0
            else 0.0,
            2,
        )

        logger.info(
            "PriceOptimizer [%s/%s]: product_id=%d cost=%.2f sale=%.2f sales=%d "
            "age=%dd idle=%dd stock=%d min=%.2f suggested=%.2f",
            source,
            dead_stock.tier,
            request.product_id,
            request.current_cost,
            request.current_sale_price,
            request.sales_last_month,
            request.product_age_days,
            aging.idle_days,
            request.current_stock,
            minimum_price,
            suggested_price,
        )

        return PriceOptimizationResponse(
            product_id=request.product_id,
            suggested_price=suggested_price,
            minimum_price=minimum_price,
            expected_margin_increase=expected_margin_increase,
            markup_over_cost_percent=markup_over_cost,
            recommendation_summary=recommendation_summary,
        )

    @staticmethod
    def _minimum_price(current_cost: float, dead_stock: DeadStockAssessment) -> float:
        markup = _CLEARANCE_MIN_MARKUP.get(dead_stock.tier, _MIN_MARKUP_RATE)
        return round(current_cost * (1 + markup), 2)

    @staticmethod
    def _velocity_adjustment(sales_last_month: int) -> tuple[float, str]:
        for threshold, multiplier, summary in _VELOCITY_RULES:
            if sales_last_month >= threshold:
                return multiplier, summary

        return _VELOCITY_RULES[-1][1], _VELOCITY_RULES[-1][2]

    @staticmethod
    def _build_price_summary(
        *,
        source: str,
        velocity_summary: str,
        dead_stock: DeadStockAssessment,
        suggested_price: float,
        minimum_price: float,
    ) -> str:
        if dead_stock.is_dead_stock:
            clearance_pct = round((1 - dead_stock.clearance_multiplier) * 100)
            summary = (
                f"Liquidación por atacasco ({clearance_pct}% adicional). "
                f"{dead_stock.label} "
            )
            if suggested_price <= minimum_price:
                summary += "Precio ajustado al piso mínimo viable."
            return summary.strip()

        if source == "ridge":
            return f"Modelo Ridge + ajuste por rotación. {velocity_summary}"

        return velocity_summary


class DemandForecaster:
    """
    Singleton — instanciado una sola vez al arrancar FastAPI.

    Selección de modelo por producto:
      - Atacasco                → proyección histórica conservadora + restock 0
      - CrostonModel en memoria → SBA (demanda intermitente)
      - Prophet en memoria      → series con patrón temporal claro
      - Sin modelo              → fallback heurístico (2 u/día)
    """

    _instance: DemandForecaster | None = None

    def __new__(cls) -> DemandForecaster:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("DemandForecaster: Singleton inicializado.")
        return cls._instance

    def predict(self, request: PurchasePredictionRequest) -> PurchasePredictionResponse:
        aging = _aging_signals_from_demand(request)
        dead_stock = evaluate_dead_stock(aging)
        registry = get_registry()
        model = registry.get_demand_model(request.product_id)

        if dead_stock.is_dead_stock:
            projected_sales = project_historical_sales(aging, request.horizon_days)
            source = f"dead_stock_{dead_stock.tier}"
            suggested_purchase_quantity = 0
        elif isinstance(model, CrostonModel):
            projected_sales = model.predict(request.horizon_days)
            source = "croston_sba"
            suggested_purchase_quantity = max(0, projected_sales - request.current_stock)
        elif model is not None:
            projected_sales = self._prophet_forecast(model, request.horizon_days)
            source = "prophet"
            suggested_purchase_quantity = max(0, projected_sales - request.current_stock)
        else:
            projected_sales = request.horizon_days * _DAILY_SALES_RATE
            source = "heuristic"
            suggested_purchase_quantity = max(0, projected_sales - request.current_stock)

        logger.info(
            "DemandForecaster [%s]: product_id=%d stock=%d horizon=%d "
            "projected=%d suggest=%d age=%dd total_sales=%d",
            source,
            request.product_id,
            request.current_stock,
            request.horizon_days,
            projected_sales,
            suggested_purchase_quantity,
            request.product_age_days,
            request.total_sales_all_time,
        )

        return PurchasePredictionResponse(
            product_id=request.product_id,
            projected_sales=projected_sales,
            suggested_purchase_quantity=suggested_purchase_quantity,
        )

    @staticmethod
    def _prophet_forecast(model: Prophet, horizon_days: int) -> int:
        future = model.make_future_dataframe(periods=horizon_days, freq="D")
        forecast = model.predict(future)
        tail = forecast.tail(horizon_days)
        projected = max(0, int(round(float(tail["yhat"].sum()))))
        return projected
