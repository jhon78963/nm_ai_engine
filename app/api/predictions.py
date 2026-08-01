from fastapi import APIRouter, Depends, HTTPException

from app.core.security import verify_api_key
from app.ml_models.predictor import DemandForecaster, PriceOptimizer
from app.schemas.prediction_schema import (
    PriceOptimizationRequest,
    PriceOptimizationResponse,
    PurchasePredictionRequest,
    PurchasePredictionResponse,
)

router = APIRouter(prefix="/api/v1/predict", tags=["Predictions"])


@router.post(
    "/price",
    response_model=PriceOptimizationResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict_price(request: PriceOptimizationRequest) -> PriceOptimizationResponse:
    """
    Devuelve el precio sugerido y márgenes para un producto.

    Payload desde nm-backend (Laravel):
    - product_id       <- products.id
    - current_cost     <- product_size.purchase_price
    - category         <- genders.name
    - sales_last_month <- SUM(sale_details.quantity) último mes
    """
    try:
        return PriceOptimizer().predict(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/demand",
    response_model=PurchasePredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict_demand(request: PurchasePredictionRequest) -> PurchasePredictionResponse:
    """
    Proyecta ventas y sugiere cantidad de restock para un producto.

    Payload desde nm-backend (Laravel):
    - product_id    <- products.id
    - current_stock <- stock actual del producto en inventario
    - horizon_days  <- días a proyectar (por defecto 30)
    """
    try:
        return DemandForecaster().predict(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
