from __future__ import annotations

import pandas as pd

from app.schemas.prediction_schema import (
    PriceOptimizationRequest,
    PurchasePredictionRequest,
)


class DataProcessor:
    """Convierte DTOs validados del backend Laravel en DataFrames listos para ML."""

    DEFAULT_CATEGORY: str = "Sin categoría"
    DATE_COLUMN: str = "date"
    QUANTITY_COLUMN: str = "quantity"

    @staticmethod
    def price_optimization_to_dataframe(
        request: PriceOptimizationRequest,
    ) -> pd.DataFrame:
        """
        Transforma productos del backend Laravel en un DataFrame para optimización de precios.

        Mapeo esperado desde nm-backend:
        - product_id  <- products.id
        - current_cost <- product_size.purchase_price (nullable -> 0.0)
        - category    <- genders.name (nullable -> "Sin categoría")
        - sales_last_month <- SUM(sale_details.quantity) del último mes
        """
        records = [product.model_dump() for product in request.products]
        dataframe = pd.DataFrame(records)

        return DataProcessor._sanitize_price_optimization_dataframe(dataframe)

    @staticmethod
    def purchase_prediction_to_dataframe(
        request: PurchasePredictionRequest,
    ) -> pd.DataFrame:
        """
        Transforma el historial de ventas del backend Laravel en un DataFrame temporal.

        Mapeo esperado desde nm-backend:
        - product_id <- products.id
        - date       <- sales.creation_time (YYYY-MM-DD o ISO-8601)
        - quantity   <- SUM(sale_details.quantity) agrupado por día
        """
        records = [entry.model_dump() for entry in request.sales_history]
        dataframe = pd.DataFrame(records)
        dataframe["product_id"] = request.product_id

        return DataProcessor._sanitize_purchase_prediction_dataframe(dataframe)

    @staticmethod
    def _sanitize_price_optimization_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        expected_columns = [
            "product_id",
            "current_cost",
            "category",
            "sales_last_month",
        ]
        sanitized = dataframe.reindex(columns=expected_columns).copy()

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

        return sanitized.reset_index(drop=True)

    @staticmethod
    def _sanitize_purchase_prediction_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        expected_columns = ["product_id", DataProcessor.DATE_COLUMN, DataProcessor.QUANTITY_COLUMN]
        sanitized = dataframe.reindex(columns=expected_columns).copy()

        sanitized["product_id"] = (
            pd.to_numeric(sanitized["product_id"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        sanitized[DataProcessor.DATE_COLUMN] = DataProcessor._parse_dates(
            sanitized[DataProcessor.DATE_COLUMN],
        )
        sanitized[DataProcessor.QUANTITY_COLUMN] = (
            pd.to_numeric(sanitized[DataProcessor.QUANTITY_COLUMN], errors="coerce")
            .fillna(0)
            .astype("int64")
        )

        sanitized = sanitized.dropna(subset=[DataProcessor.DATE_COLUMN])

        if sanitized.empty:
            raise ValueError(
                "No se encontraron fechas válidas en sales_history. "
                "El backend debe enviar fechas en formato YYYY-MM-DD o ISO-8601.",
            )

        sanitized = (
            sanitized.groupby(
                ["product_id", DataProcessor.DATE_COLUMN],
                as_index=False,
            )[DataProcessor.QUANTITY_COLUMN]
            .sum()
            .sort_values(DataProcessor.DATE_COLUMN)
            .reset_index(drop=True)
        )

        return sanitized

    @staticmethod
    def _parse_dates(series: pd.Series) -> pd.Series:
        """
        Normaliza fechas enviadas por Laravel.

        Soporta:
        - date objects (post-validación Pydantic)
        - YYYY-MM-DD
        - ISO-8601 (sales.creation_time?->toIso8601String())
        """
        parsed_dates = pd.to_datetime(series, errors="coerce", utc=False)

        if parsed_dates.dt.tz is not None:
            parsed_dates = parsed_dates.dt.tz_convert(None)

        return parsed_dates.dt.normalize()
