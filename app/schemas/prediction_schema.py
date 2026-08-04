from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request Models — recibidos desde nm-backend (Laravel)
# ---------------------------------------------------------------------------


class PriceOptimizationRequest(BaseModel):
    """
    Producto enviado por nm-backend para optimización de precio.

    Mapeo desde Laravel:
    - product_id            <- products.id
    - current_cost          <- product_size.purchase_price
    - category              <- genders.name
    - sales_last_month      <- SUM(sale_details.quantity) último mes
    - current_stock         <- inventario maestro del producto
    - product_age_days      <- días desde products.creation_time
    - days_since_last_sale  <- días desde última venta (o antigüedad si nunca vendió)
    - total_sales_all_time  <- ventas históricas totales
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    product_id: int = Field(..., gt=0, description="Identificador único del producto")
    current_cost: float = Field(..., ge=0.0, description="Costo actual del producto")
    current_sale_price: float = Field(
        ...,
        ge=0.0,
        description="Precio de venta actual en catálogo (product_size.sale_price)",
    )
    category: str = Field(..., min_length=1, description="Categoría (genders.name)")
    sales_last_month: int = Field(..., ge=0, description="Ventas totales del último mes")
    current_stock: int = Field(default=0, ge=0, description="Stock actual en inventario")
    product_age_days: int = Field(default=0, ge=0, description="Días desde el alta del producto")
    days_since_last_sale: int = Field(
        default=0,
        ge=0,
        description="Días desde la última venta completada",
    )
    total_sales_all_time: int = Field(default=0, ge=0, description="Unidades vendidas históricas")


class PurchasePredictionRequest(BaseModel):
    """
    Solicitud de predicción de restock enviada por nm-backend.

    Mapeo desde Laravel:
    - product_id            <- products.id
    - current_stock         <- inventario maestro del producto
    - horizon_days          <- días a proyectar (por defecto 30)
    - sales_last_month      <- ventas últimos 30 días
    - product_age_days      <- antigüedad del producto
    - days_since_last_sale  <- días sin venta
    - total_sales_all_time  <- ventas históricas
    """

    product_id: int = Field(..., gt=0, description="Identificador del producto")
    current_stock: int = Field(..., ge=0, description="Stock actual en inventario")
    horizon_days: int = Field(
        default=30,
        gt=0,
        le=365,
        description="Días a proyectar (1-365, por defecto 30)",
    )
    sales_last_month: int = Field(default=0, ge=0, description="Ventas totales del último mes")
    product_age_days: int = Field(default=0, ge=0, description="Días desde el alta del producto")
    days_since_last_sale: int = Field(
        default=0,
        ge=0,
        description="Días desde la última venta completada",
    )
    total_sales_all_time: int = Field(default=0, ge=0, description="Unidades vendidas históricas")


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
    markup_over_cost_percent: float = Field(
        ..., description="Margen bruto sugerido sobre el costo (%)"
    )
    recommendation_summary: str = Field(
        ..., description="Resumen legible de la estrategia de precio aplicada"
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


# ---------------------------------------------------------------------------
# Bulk — reporte masivo de inventario con IA
# ---------------------------------------------------------------------------


class BulkPredictionItemRequest(BaseModel):
    product_id: int = Field(..., gt=0)
    price: PriceOptimizationRequest | None = None
    demand: PurchasePredictionRequest


class BulkPredictionRequest(BaseModel):
    items: list[BulkPredictionItemRequest] = Field(..., min_length=1, max_length=500)


class BulkPredictionItemResponse(BaseModel):
    product_id: int
    suggested_price: float | None = None
    suggested_min_price: float | None = None
    suggested_purchase_quantity: int | None = None
    projected_sales: int | None = None
    is_dead_stock: bool = False
    price_error: str | None = None
    demand_error: str | None = None


class BulkPredictionResponse(BaseModel):
    items: list[BulkPredictionItemResponse]
    processed: int
    errors: int
