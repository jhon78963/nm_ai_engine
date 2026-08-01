from __future__ import annotations

import logging

from app.schemas.prediction_schema import (
    PriceOptimizationRequest,
    PriceOptimizationResponse,
    PurchasePredictionRequest,
    PurchasePredictionResponse,
)

logger = logging.getLogger(__name__)

_HIGH_VOLUME_THRESHOLD: int = 50
_HIGH_VOLUME_MARGIN: float = 0.10
_LOW_VOLUME_MARGIN: float = 0.05
_MINIMUM_MARGIN: float = 0.05
_DAILY_SALES_RATE: int = 2


class PriceOptimizer:
    """
    Singleton — instanciado una sola vez al arrancar FastAPI.

    Optimiza el precio de venta de un producto según su volumen de ventas
    del último mes. Placeholder hasta conectar la BD y cargar un modelo .pkl.

    Regla simulada:
    - sales_last_month > 50  → +10% sobre current_cost
    - sales_last_month <= 50 → +5%  sobre current_cost
    """

    _instance: PriceOptimizer | None = None

    def __new__(cls) -> PriceOptimizer:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("PriceOptimizer: Singleton inicializado.")
        return cls._instance

    def predict(self, request: PriceOptimizationRequest) -> PriceOptimizationResponse:
        """
        Calcula el precio sugerido para un producto.

        Args:
            request: Datos del producto enviados por nm-backend (Laravel).

        Returns:
            PriceOptimizationResponse con precio sugerido y márgenes calculados.
        """
        margin_rate = (
            _HIGH_VOLUME_MARGIN
            if request.sales_last_month > _HIGH_VOLUME_THRESHOLD
            else _LOW_VOLUME_MARGIN
        )

        suggested_price = round(request.current_cost * (1 + margin_rate), 2)
        minimum_price = round(request.current_cost * (1 + _MINIMUM_MARGIN), 2)
        expected_margin_increase = round(
            ((suggested_price - minimum_price) / minimum_price * 100)
            if minimum_price > 0
            else 0.0,
            2,
        )

        logger.info(
            "PriceOptimizer: product_id=%d sales=%d margin=%.0f%% suggested=%.2f",
            request.product_id,
            request.sales_last_month,
            margin_rate * 100,
            suggested_price,
        )

        return PriceOptimizationResponse(
            product_id=request.product_id,
            suggested_price=suggested_price,
            minimum_price=minimum_price,
            expected_margin_increase=expected_margin_increase,
        )


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
        """
        Proyecta ventas y calcula la cantidad sugerida de compra.

        Args:
            request: Datos de stock y horizonte enviados por nm-backend (Laravel).

        Returns:
            PurchasePredictionResponse con proyección y cantidad sugerida.
        """
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
