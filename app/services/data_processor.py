from __future__ import annotations

import pandas as pd

from app.schemas.prediction_schema import (
    PriceOptimizationRequest,
    PurchasePredictionRequest,
)


class DataProcessor:
    """Convierte DTOs validados del backend Laravel en DataFrames listos para ML."""

    DEFAULT_CATEGORY: str = "Sin categoría"

    @staticmethod
    def price_optimization_to_dataframe(
        request: PriceOptimizationRequest,
    ) -> pd.DataFrame:
        """
        Transforma un producto de nm-backend en un DataFrame de una fila para Ridge.

        Columnas resultantes: product_id, current_cost, category, sales_last_month
        """
        record = request.model_dump()
        dataframe = pd.DataFrame([record])
        return DataProcessor._sanitize_price_dataframe(dataframe)

    @staticmethod
    def purchase_prediction_to_dataframe(
        request: PurchasePredictionRequest,
    ) -> pd.DataFrame:
        """
        Transforma el request de restock en un DataFrame de una fila.

        Columnas resultantes: product_id, current_stock, horizon_days

        Nota: la proyección de ventas se calculará en DemandForecaster.
        Cuando esté disponible la conexión a BD, este método enriquecerá
        el DataFrame con el historial de ventas de sale_details.
        """
        record = request.model_dump()
        dataframe = pd.DataFrame([record])
        return DataProcessor._sanitize_purchase_dataframe(dataframe)

    # ------------------------------------------------------------------
    # Métodos privados de saneamiento
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_price_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        sanitized = dataframe.reindex(
            columns=[
                "product_id",
                "current_cost",
                "category",
                "sales_last_month",
                "product_age_days",
                "days_since_last_sale",
                "total_sales_all_time",
                "current_stock",
            ]
        ).copy()

        sanitized["product_id"] = (
            pd.to_numeric(sanitized["product_id"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        sanitized["current_cost"] = (
            pd.to_numeric(sanitized["current_cost"], errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )
        sanitized["category"] = (
            sanitized["category"]
            .fillna(DataProcessor.DEFAULT_CATEGORY)
            .astype(str)
            .str.strip()
            .replace("", DataProcessor.DEFAULT_CATEGORY)
        )
        sanitized["sales_last_month"] = (
            pd.to_numeric(sanitized["sales_last_month"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        for column in (
            "product_age_days",
            "days_since_last_sale",
            "total_sales_all_time",
            "current_stock",
        ):
            sanitized[column] = (
                pd.to_numeric(sanitized[column], errors="coerce")
                .fillna(0)
                .astype("int64")
            )

        return sanitized.reset_index(drop=True)

    @staticmethod
    def _sanitize_purchase_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        sanitized = dataframe.reindex(
            columns=["product_id", "current_stock", "horizon_days"]
        ).copy()

        sanitized["product_id"] = (
            pd.to_numeric(sanitized["product_id"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        sanitized["current_stock"] = (
            pd.to_numeric(sanitized["current_stock"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        sanitized["horizon_days"] = (
            pd.to_numeric(sanitized["horizon_days"], errors="coerce")
            .fillna(30)
            .astype("int64")
        )

        return sanitized.reset_index(drop=True)
