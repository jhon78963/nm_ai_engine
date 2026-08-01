from fastapi import APIRouter, HTTPException

from app.ml_models.predictor import DemandForecaster, DemandForecast, PriceOptimizer
from app.schemas.prediction_schema import (
    PriceOptimizationRequest,
    PurchasePredictionRequest,
)
from app.services.data_processor import DataProcessor

router = APIRouter(tags=["Predictions"])


@router.post("/predict/price")
def predict_price(request: PriceOptimizationRequest) -> dict[str, list[dict[str, float | int]]]:
    """
    Optimiza precios para productos enviados desde nm-backend (Laravel).

    Payload esperado:
    - products[].product_id      <- products.id
    - products[].current_cost    <- product_size.purchase_price
    - products[].category        <- genders.name
    - products[].sales_last_month <- SUM(sale_details.quantity) del último mes
    """
    try:
        dataframe = DataProcessor.price_optimization_to_dataframe(request)
        predictions = PriceOptimizer().predict(dataframe)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"predictions": predictions}


@router.post("/predict/demand")
def predict_demand(request: PurchasePredictionRequest) -> DemandForecast:
    """
    Predice la demanda de un producto para los próximos 7 días.

    Payload esperado desde nm-backend (Laravel):
    - product_id    <- products.id
    - sales_history <- ventas agrupadas por sales.creation_time (YYYY-MM-DD)
    """
    try:
        dataframe = DataProcessor.purchase_prediction_to_dataframe(request)
        forecast = DemandForecaster().predict(dataframe, forecast_days=7)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return forecast
