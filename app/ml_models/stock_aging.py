from __future__ import annotations

from dataclasses import dataclass

TIER_NONE = "none"
TIER_AGING = "aging"
TIER_HIGH = "high"
TIER_CRITICAL = "critical"


@dataclass(frozen=True)
class StockAgingSignals:
    product_age_days: int
    days_since_last_sale: int
    sales_last_month: int
    current_stock: int
    total_sales_all_time: int

    @property
    def idle_days(self) -> int:
        return self.days_since_last_sale if self.days_since_last_sale > 0 else self.product_age_days

    @property
    def historical_daily_sales(self) -> float:
        age = max(self.product_age_days, 1)
        return self.total_sales_all_time / age


@dataclass(frozen=True)
class DeadStockAssessment:
    is_dead_stock: bool
    tier: str
    clearance_multiplier: float
    label: str


def evaluate_dead_stock(signals: StockAgingSignals) -> DeadStockAssessment:
    """Detecta atacasco y devuelve multiplicador de liquidación (alineado con Laravel)."""
    idle = signals.idle_days

    if (
        signals.product_age_days >= 90
        and signals.total_sales_all_time == 0
        and signals.current_stock >= 5
    ):
        return _result(
            TIER_CRITICAL,
            0.55,
            "Sin ventas registradas: liquidación urgente recomendada.",
        )

    if (
        signals.product_age_days >= 180
        and signals.sales_last_month <= 4
        and signals.current_stock >= 10
        and idle >= 30
    ):
        return _result(
            TIER_CRITICAL,
            0.60,
            "Atacasco crítico: antigüedad alta, stock elevado y ventas mínimas.",
        )

    if (
        signals.product_age_days >= 120
        and signals.sales_last_month < 5
        and signals.current_stock >= 5
        and idle >= 45
    ):
        return _result(
            TIER_HIGH,
            0.72,
            "Producto estancado: se recomienda rebaja fuerte para liberar capital.",
        )

    if (
        signals.product_age_days >= 90
        and signals.total_sales_all_time <= 10
        and signals.current_stock >= 10
        and idle >= 30
    ):
        return _result(
            TIER_AGING,
            0.80,
            "Rotación muy lenta: descuento adicional para acelerar salida.",
        )

    return _result(TIER_NONE, 1.0, "")


def project_historical_sales(signals: StockAgingSignals, horizon_days: int) -> int:
    """Proyección conservadora basada en velocidad histórica real."""
    projected = signals.historical_daily_sales * horizon_days
    return max(0, int(round(projected)))


def _result(tier: str, multiplier: float, label: str) -> DeadStockAssessment:
    return DeadStockAssessment(
        is_dead_stock=tier != TIER_NONE,
        tier=tier,
        clearance_multiplier=multiplier,
        label=label,
    )
