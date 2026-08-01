from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

_MIN_PROPHET_DATAPOINTS: int = 2


# ---------------------------------------------------------------------------
# Response type contracts (consumed by the api/ layer)
# ---------------------------------------------------------------------------


class PricePrediction(TypedDict):
    product_id: int
    current_cost: float
    suggested_price: float
    margin_applied: float


class DemandForecastEntry(TypedDict):
    date: str
    predicted_quantity: int


class DemandForecast(TypedDict):
    product_id: int
    forecast_days: int
    predictions: list[DemandForecastEntry]


class RestockPrediction(TypedDict):
    product_id: int
    projected_sales: int
    suggested_purchase_quantity: int


# ---------------------------------------------------------------------------
# Singleton: PriceOptimizer
# ---------------------------------------------------------------------------


class PriceOptimizer:
    """
    Singleton — cargado una sola vez en memoria al arrancar FastAPI.

    Usa Ridge regression entrenado con datos semilla que codifican la lógica
    de márgenes reales de nm-backend:
      - Volumen alto (>100 und/mes)  -> margen ~25% (precio competitivo)
      - Volumen medio (30-100 und/mes) -> margen ~40%
      - Volumen bajo  (<30 und/mes)  -> margen ~60% (precio premium)

    DataFrame de entrada esperado (salida de DataProcessor):
      product_id | current_cost | category | sales_last_month
    """

    _instance: PriceOptimizer | None = None
    _model: Ridge
    _label_encoder: LabelEncoder

    def __new__(cls) -> PriceOptimizer:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self._label_encoder = LabelEncoder()
        self._model = Ridge(alpha=1.0)
        self._train_with_seed_data()

    def _train_with_seed_data(self) -> None:
        """
        Entrena Ridge con datos sintéticos que representan la lógica de márgenes
        de nm-backend hasta que se cargue un modelo .pkl real.

        Categorías reflejan genders.name del backend Laravel.
        """
        rng = np.random.default_rng(seed=42)
        n_samples = 400

        costs = rng.uniform(5.0, 250.0, n_samples)
        sales = rng.integers(1, 350, n_samples)
        categories = rng.choice(["Dama", "Caballero", "Niños", "Sin categoría"], n_samples)

        known_categories = ["Caballero", "Dama", "Niños", "Sin categoría"]
        self._label_encoder.fit(known_categories)
        cat_encoded = self._label_encoder.transform(categories)

        margin_factors = np.where(
            sales > 100, 1.25,
            np.where(sales >= 30, 1.40, 1.60),
        )
        target_prices = costs * margin_factors + rng.uniform(-0.5, 0.5, n_samples)

        X = np.column_stack([costs, sales, cat_encoded])
        self._model.fit(X, target_prices)

        logger.info("PriceOptimizer: modelo inicializado con datos semilla.")

    def predict(self, dataframe: pd.DataFrame) -> list[PricePrediction]:
        """
        Devuelve el precio sugerido por producto con el margen óptimo aplicado.

        Args:
            dataframe: Salida de DataProcessor.price_optimization_to_dataframe().
                       Columnas: product_id, current_cost, category, sales_last_month.

        Returns:
            Lista de PricePrediction — una entrada por producto.
        """
        results: list[PricePrediction] = []

        for _, row in dataframe.iterrows():
            category_str = str(row["category"]).strip()

            if category_str not in self._label_encoder.classes_:
                logger.warning(
                    "PriceOptimizer: categoría desconocida '%s', usando 'Sin categoría'.",
                    category_str,
                )
                category_str = "Sin categoría"

            cat_encoded = int(self._label_encoder.transform([category_str])[0])
            current_cost = float(row["current_cost"])
            sales_volume = int(row["sales_last_month"])

            features = np.array([[current_cost, sales_volume, cat_encoded]])
            suggested = float(self._model.predict(features)[0])

            # Garantizar margen mínimo del 5% sobre el costo (regla de negocio nm)
            min_price = current_cost * 1.05
            suggested = max(suggested, min_price)

            margin = (
                round((suggested - current_cost) / current_cost * 100, 2)
                if current_cost > 0
                else 0.0
            )

            results.append(
                PricePrediction(
                    product_id=int(row["product_id"]),
                    current_cost=round(current_cost, 2),
                    suggested_price=round(suggested, 2),
                    margin_applied=margin,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Singleton: DemandForecaster
# ---------------------------------------------------------------------------


class DemandForecaster:
    """
    Singleton — cargado una sola vez en memoria al arrancar FastAPI.

    Usa Prophet para predecir la demanda de un producto para los próximos N días.
    Prophet espera columnas `ds` (fecha) e `y` (cantidad vendida).

    Si el historial tiene menos de 2 puntos (mínimo de Prophet), se activa
    un fallback de promedio móvil que devuelve la misma estructura de respuesta.

    DataFrame de entrada esperado (salida de DataProcessor):
      product_id | date | quantity
    """

    _instance: DemandForecaster | None = None

    def __new__(cls) -> DemandForecaster:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("DemandForecaster: Singleton creado.")
        return cls._instance

    def predict(
        self,
        dataframe: pd.DataFrame,
        forecast_days: int = 7,
    ) -> DemandForecast:
        """
        Devuelve la predicción de demanda para los próximos forecast_days días.

        Args:
            dataframe: Salida de DataProcessor.purchase_prediction_to_dataframe().
                       Columnas: product_id, date, quantity.
            forecast_days: Días a predecir hacia adelante (por defecto 7).

        Returns:
            DemandForecast con product_id y lista de {date, predicted_quantity}.
        """
        product_id = int(dataframe["product_id"].iloc[0])

        if len(dataframe) < _MIN_PROPHET_DATAPOINTS:
            logger.warning(
                "DemandForecaster: product_id=%d solo tiene %d registro(s). "
                "Activando fallback de promedio.",
                product_id,
                len(dataframe),
            )
            return self._fallback_forecast(dataframe, product_id, forecast_days)

        return self._prophet_forecast(dataframe, product_id, forecast_days)

    def _prophet_forecast(
        self,
        dataframe: pd.DataFrame,
        product_id: int,
        forecast_days: int,
    ) -> DemandForecast:
        prophet_df = (
            dataframe[["date", "quantity"]]
            .rename(columns={"date": "ds", "quantity": "y"})
            .copy()
        )
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df["y"] = prophet_df["y"].astype(float)

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=len(prophet_df) >= 365,
            interval_width=0.80,
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=forecast_days, freq="D")
        forecast = model.predict(future)

        last_known_date = prophet_df["ds"].max()
        future_rows = forecast[forecast["ds"] > last_known_date].head(forecast_days)

        predictions: list[DemandForecastEntry] = [
            DemandForecastEntry(
                date=str(row["ds"].date()),
                predicted_quantity=max(0, round(float(row["yhat"]))),
            )
            for _, row in future_rows.iterrows()
        ]

        return DemandForecast(
            product_id=product_id,
            forecast_days=forecast_days,
            predictions=predictions,
        )

    def _fallback_forecast(
        self,
        dataframe: pd.DataFrame,
        product_id: int,
        forecast_days: int,
    ) -> DemandForecast:
        """Promedio móvil cuando Prophet no puede ajustarse por falta de datos."""
        avg_quantity = max(0, round(float(dataframe["quantity"].mean())))
        last_date = pd.to_datetime(dataframe["date"].max())

        predictions: list[DemandForecastEntry] = [
            DemandForecastEntry(
                date=str((last_date + pd.Timedelta(days=i + 1)).date()),
                predicted_quantity=avg_quantity,
            )
            for i in range(forecast_days)
        ]

        return DemandForecast(
            product_id=product_id,
            forecast_days=forecast_days,
            predictions=predictions,
        )

    def predict_restock(self, dataframe: pd.DataFrame) -> RestockPrediction:
        """
        Proyecta ventas y calcula la cantidad sugerida de restock.

        Recibe un DataFrame de una fila con columnas: product_id, current_stock, horizon_days.
        Usa una tasa diaria sintética de venta mientras no haya conexión a BD.

        Una vez disponible DATABASE_URL, este método consultará sale_details para
        calcular la tasa real de ventas del producto.

        Args:
            dataframe: Salida de DataProcessor.purchase_prediction_to_dataframe().

        Returns:
            RestockPrediction con projected_sales y suggested_purchase_quantity.
        """
        row = dataframe.iloc[0]
        product_id = int(row["product_id"])
        current_stock = int(row["current_stock"])
        horizon_days = int(row["horizon_days"])

        # Tasa diaria sintética — placeholder hasta integración con sale_details
        # Será reemplazado por: SELECT AVG(daily_qty) FROM sale_details WHERE product_id = ?
        _SYNTHETIC_DAILY_RATE: float = 5.0
        projected_sales = max(0, round(_SYNTHETIC_DAILY_RATE * horizon_days))
        suggested_quantity = max(0, projected_sales - current_stock)

        logger.info(
            "DemandForecaster.predict_restock: product_id=%d stock=%d "
            "horizon=%d days projected=%d suggest=%d",
            product_id,
            current_stock,
            horizon_days,
            projected_sales,
            suggested_quantity,
        )

        return RestockPrediction(
            product_id=product_id,
            projected_sales=projected_sales,
            suggested_purchase_quantity=suggested_quantity,
        )
