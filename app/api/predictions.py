from fastapi import APIRouter, Depends, HTTPException

from app.core.security import verify_api_key
from app.ml_models.predictor import DemandForecaster, PriceOptimizer
from app.schemas.prediction_schema import (
    PriceOptimizationRequest,
    PriceOptimizationResponse,
    PurchasePredictionRequest,
    PurchasePredictionResponse,
)
from app.services.data_processor import DataProcessor

router = APIRouter(tags=["Predictions"], dependencies=[Depends(verify_api_key)])


@router.post("/predict/price", response_model=PriceOptimizationResponse)
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
        dataframe = DataProcessor.price_optimization_to_dataframe(request)
        predictions = PriceOptimizer().predict(dataframe)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    prediction = predictions[0]
    current_cost = prediction["current_cost"]
    suggested_price = prediction["suggested_price"]
    minimum_price = round(current_cost * 1.05, 2)
    expected_margin_increase = round(
        ((suggested_price - minimum_price) / minimum_price * 100) if minimum_price > 0 else 0.0,
        2,
    )

    return PriceOptimizationResponse(
        product_id=prediction["product_id"],
        suggested_price=suggested_price,
        minimum_price=minimum_price,
        expected_margin_increase=expected_margin_increase,
    )


@router.post("/predict/demand", response_model=PurchasePredictionResponse)
def predict_demand(request: PurchasePredictionRequest) -> PurchasePredictionResponse:
    """
    Proyecta ventas y sugiere cantidad de restock para un producto.

    Payload desde nm-backend (Laravel):
    - product_id    <- products.id
    - current_stock <- stock actual del producto en inventario
    - horizon_days  <- días a proyectar (por defecto 30)
    """
    try:
        dataframe = DataProcessor.purchase_prediction_to_dataframe(request)
        result = DemandForecaster().predict_restock(dataframe)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PurchasePredictionResponse(
        product_id=result["product_id"],
        projected_sales=result["projected_sales"],
        suggested_purchase_quantity=result["suggested_purchase_quantity"],
    )
