from datetime import date as DateType

from pydantic import BaseModel, ConfigDict, Field


class ProductItem(BaseModel):
    """Producto individual para optimización de precios."""

    model_config = ConfigDict(str_strip_whitespace=True)

    product_id: int = Field(..., gt=0, description="Identificador único del producto")
    current_cost: float = Field(..., ge=0, description="Costo actual del producto")
    category: str = Field(..., min_length=1, description="Categoría del producto")
    sales_last_month: int = Field(..., ge=0, description="Ventas del último mes")


class PriceOptimizationRequest(BaseModel):
    """Solicitud de optimización de precios para múltiples productos."""

    products: list[ProductItem] = Field(
        ...,
        min_length=1,
        description="Lista de productos a evaluar",
    )


class SalesHistoryEntry(BaseModel):
    """Registro histórico de ventas por fecha."""

    date: DateType = Field(..., description="Fecha de la venta")
    quantity: int = Field(..., ge=0, description="Cantidad vendida en la fecha")


class PurchasePredictionRequest(BaseModel):
    """Solicitud de predicción de compras basada en historial de ventas."""

    product_id: int = Field(..., gt=0, description="Identificador del producto")
    sales_history: list[SalesHistoryEntry] = Field(
        ...,
        min_length=1,
        description="Historial de ventas con fecha y cantidad",
    )
