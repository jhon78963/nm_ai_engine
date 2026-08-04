from __future__ import annotations

import logging

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


class PriceOptimizer:
    """
    Singleton — instanciado una sola vez al arrancar FastAPI.

    Optimiza precio de venta para retail/wholesale de ropa usando:
    - costo de compra como piso de margen
    - precio de venta actual como ancla comercial
    - rotación de los últimos 30 días como señal de demanda
    """

    _instance: PriceOptimizer | None = None

    def __new__(cls) -> PriceOptimizer:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("PriceOptimizer: Singleton inicializado.")
        return cls._instance

    def predict(self, request: PriceOptimizationRequest) -> PriceOptimizationResponse:
        minimum_price = round(request.current_cost * (1 + _MIN_MARKUP_RATE), 2)

        anchor_price = (
            request.current_sale_price
            if request.current_sale_price > 0
            else round(request.current_cost * (1 + _DEFAULT_MARKUP_RATE), 2)
        )

        adjustment, summary = self._velocity_adjustment(request.sales_last_month)
        suggested_price = round(max(minimum_price, anchor_price * adjustment), 2)

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
            "PriceOptimizer: product_id=%d cost=%.2f sale=%.2f sales=%d "
            "min=%.2f suggested=%.2f markup=%.1f%%",
            request.product_id,
            request.current_cost,
            request.current_sale_price,
            request.sales_last_month,
            minimum_price,
            suggested_price,
            markup_over_cost,
        )

        return PriceOptimizationResponse(
            product_id=request.product_id,
            suggested_price=suggested_price,
            minimum_price=minimum_price,
            expected_margin_increase=expected_margin_increase,
            markup_over_cost_percent=markup_over_cost,
            recommendation_summary=summary,
        )

    @staticmethod
    def _velocity_adjustment(sales_last_month: int) -> tuple[float, str]:
        for threshold, multiplier, summary in _VELOCITY_RULES:
            if sales_last_month >= threshold:
                return multiplier, summary

        return _VELOCITY_RULES[-1][1], _VELOCITY_RULES[-1][2]


class DemandForecaster:
    """
    Singleton — instanciado una sola vez al arrancar FastAPI.

    Proyecta ventas futuras y sugiere cantidad de restock para un producto.
    Placeholder hasta conectar la BD y usar Prophet con historial real.

    Regla simulada:
    - projected_sales            = horizon_days * 2  (2 unidades/día)
    - suggested_purchase_quantity = max(0, projected_sales - current_stock)
    """

    _instance: DemandForecaster | None = None

    def __new__(cls) -> DemandForecaster:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("DemandForecaster: Singleton inicializado.")
        return cls._instance

    def predict(self, request: PurchasePredictionRequest) -> PurchasePredictionResponse:
        projected_sales = request.horizon_days * _DAILY_SALES_RATE
        suggested_purchase_quantity = max(0, projected_sales - request.current_stock)

        logger.info(
            "DemandForecaster: product_id=%d stock=%d horizon=%d "
            "projected=%d suggest=%d",
            request.product_id,
            request.current_stock,
            request.horizon_days,
            projected_sales,
            suggested_purchase_quantity,
        )

        return PurchasePredictionResponse(
            product_id=request.product_id,
            projected_sales=projected_sales,
            suggested_purchase_quantity=suggested_purchase_quantity,
        )
