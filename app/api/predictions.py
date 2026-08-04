from fastapi import APIRouter, Depends, HTTPException

from app.core.security import verify_api_key
from app.ml_models.predictor import DemandForecaster, PriceOptimizer
from app.ml_models.stock_aging import StockAgingSignals, evaluate_dead_stock
from app.schemas.prediction_schema import (
    BulkPredictionItemResponse,
    BulkPredictionRequest,
    BulkPredictionResponse,
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


@router.post(
    "/bulk",
    response_model=BulkPredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict_bulk(request: BulkPredictionRequest) -> BulkPredictionResponse:
    """
    Predicciones masivas de precio y demanda para el reporte de inventario.

    Procesa hasta 500 productos en una sola llamada (sin HTTP por producto).
    """
    price_optimizer = PriceOptimizer()
    demand_forecaster = DemandForecaster()
    results: list[BulkPredictionItemResponse] = []
    errors = 0

    for item in request.items:
        price_result: PriceOptimizationResponse | None = None
        price_error: str | None = None
        demand_result: PurchasePredictionResponse | None = None
        demand_error: str | None = None
        is_dead_stock = False

        if item.price is not None:
            try:
                price_result = price_optimizer.predict(item.price)
            except (ValueError, TypeError) as exc:
                price_error = str(exc)
                errors += 1

            aging = StockAgingSignals(
                product_age_days=item.price.product_age_days,
                days_since_last_sale=item.price.days_since_last_sale,
                sales_last_month=item.price.sales_last_month,
                current_stock=item.price.current_stock,
                total_sales_all_time=item.price.total_sales_all_time,
            )
            is_dead_stock = evaluate_dead_stock(aging).is_dead_stock

        try:
            demand_result = demand_forecaster.predict(item.demand)
        except (ValueError, TypeError) as exc:
            demand_error = str(exc)
            errors += 1

        results.append(
            BulkPredictionItemResponse(
                product_id=item.product_id,
                suggested_price=price_result.suggested_price if price_result else None,
                suggested_min_price=price_result.minimum_price if price_result else None,
                suggested_purchase_quantity=(
                    demand_result.suggested_purchase_quantity if demand_result else None
                ),
                projected_sales=demand_result.projected_sales if demand_result else None,
                is_dead_stock=is_dead_stock,
                price_error=price_error,
                demand_error=demand_error,
            )
        )

    return BulkPredictionResponse(
        items=results,
        processed=len(results),
        errors=errors,
    )
