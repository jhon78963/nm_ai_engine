"""
Croston / SBA (Syntetos-Boylan Approximation) para demanda intermitente.

Estándar académico para series de demanda con muchos ceros y picos esporádicos,
típico en retail de ropa donde un producto vende 0–3 unidades en la mayoría de
los días y tiene picos ocasionales.

Referencias:
  - Croston, J.D. (1972). Forecasting and stock control for intermittent demands.
  - Syntetos, A.A. & Boylan, J.E. (2001). On the bias of intermittent demand estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CrostonModel:
    """
    Modelo Croston/SBA serializable con joblib.

    alpha : factor de suavizado (0 < alpha < 1).
            Valores bajos → memoria larga, responde lento.
            Valores altos → responde rápido a cambios recientes.

    use_sba : True → SBA (sesgo corregido, recomendado para tesis).
              False → Croston original.

    fitted_demand_rate : unidades por período (día) estimadas tras el ajuste.
    """

    alpha: float = 0.15
    use_sba: bool = True
    fitted_demand_rate: float = 0.0
    _demand_size: float = field(default=0.0, repr=False)  # z: tamaño medio de pedido
    _inter_arrival: float = field(default=0.0, repr=False)  # q: intervalo medio entre pedidos

    # ------------------------------------------------------------------ fit

    def fit(self, series: list[float] | list[int]) -> CrostonModel:
        """
        Ajusta el modelo a una serie diaria (incluye ceros).

        series : lista de unidades vendidas por día, en orden cronológico.
        """
        if not series:
            return self

        # Separar eventos de demanda (días con venta > 0)
        demand_events = [(i, v) for i, v in enumerate(series) if v > 0]

        if not demand_events:
            self.fitted_demand_rate = 0.0
            return self

        # Inicializar con el primer evento
        z = float(demand_events[0][1])   # tamaño del primer pedido
        q = 1.0                           # primer intervalo = 1 (no hay previo)

        prev_idx = demand_events[0][0]

        for idx, val in demand_events[1:]:
            inter = float(idx - prev_idx)
            z = self.alpha * val + (1 - self.alpha) * z
            q = self.alpha * inter + (1 - self.alpha) * q
            prev_idx = idx

        self._demand_size = z
        self._inter_arrival = max(q, 1e-9)

        # SBA: corrección de sesgo multiplicando por (1 - alpha/2)
        if self.use_sba:
            self.fitted_demand_rate = (1 - self.alpha / 2) * (z / self._inter_arrival)
        else:
            self.fitted_demand_rate = z / self._inter_arrival

        return self

    # ------------------------------------------------------------------ predict

    def predict(self, horizon_days: int) -> int:
        """
        Proyecta ventas para los próximos horizon_days días.

        Devuelve un entero (unidades totales proyectadas).
        """
        projected = self.fitted_demand_rate * horizon_days
        return max(0, int(round(projected)))

    def predict_daily_rate(self) -> float:
        """Tasa diaria estimada (unidades/día)."""
        return max(0.0, self.fitted_demand_rate)
