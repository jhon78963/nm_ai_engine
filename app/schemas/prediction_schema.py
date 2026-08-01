from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request Models — recibidos desde nm-backend (Laravel)
# ---------------------------------------------------------------------------


class PriceOptimizationRequest(BaseModel):
    """
    Producto enviado por nm-backend para optimización de precio.

    Mapeo desde Laravel:
    - product_id       <- products.id
    - current_cost     <- product_size.purchase_price
    - category         <- genders.name
    - sales_last_month <- SUM(sale_details.quantity) último mes
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    product_id: int = Field(..., gt=0, description="Identificador único del producto")
    current_cost: float = Field(..., ge=0.0, description="Costo actual del producto")
    category: str = Field(..., min_length=1, description="Categoría (genders.name)")
    sales_last_month: int = Field(..., ge=0, description="Ventas totales del último mes")


class PurchasePredictionRequest(BaseModel):
    """
    Solicitud de predicción de restock enviada por nm-backend.

    Mapeo desde Laravel:
    - product_id    <- products.id
    - current_stock <- SUM(inventory_movements.quantity) para el producto
    - horizon_days  <- días hacia adelante a proyectar (por defecto 30)
    """

    product_id: int = Field(..., gt=0, description="Identificador del producto")
    current_stock: int = Field(..., ge=0, description="Stock actual en inventario")
    horizon_days: int = Field(
        default=30,
        gt=0,
        le=365,
        description="Días a proyectar (1-365, por defecto 30)",
    )


# ---------------------------------------------------------------------------
# Response Models — devueltos a nm-backend (Laravel)
# ---------------------------------------------------------------------------


class PriceOptimizationResponse(BaseModel):
    """Respuesta con el precio sugerido para un producto."""

    product_id: int = Field(..., description="Identificador del producto evaluado")
    suggested_price: float = Field(..., description="Precio de venta sugerido")
    minimum_price: float = Field(
        ..., description="Precio mínimo viable (costo + 5% margen base)"
    )
    expected_margin_increase: float = Field(
        ..., description="Incremento de margen esperado sobre el mínimo viable (%)"
    )


class PurchasePredictionResponse(BaseModel):
    """Respuesta con la proyección de ventas y la cantidad sugerida de compra."""

    product_id: int = Field(..., description="Identificador del producto evaluado")
    projected_sales: int = Field(
        ..., description="Unidades proyectadas a vender en horizon_days"
    )
    suggested_purchase_quantity: int = Field(
        ..., description="Unidades sugeridas a comprar (proyección - stock actual)"
    )
