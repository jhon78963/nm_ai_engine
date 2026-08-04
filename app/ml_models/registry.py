from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
from prophet import Prophet
from sklearn.pipeline import Pipeline

from app.core.config import DEMAND_MODELS_DIR, PRICE_MODELS_DIR, get_settings
from app.ml_models.croston import CrostonModel

logger = logging.getLogger(__name__)

PRICE_MODEL_FILE = PRICE_MODELS_DIR / "price_ridge.joblib"
DEMAND_MANIFEST_FILE = DEMAND_MODELS_DIR / "manifest.json"

DemandModel = Prophet | CrostonModel


class ModelRegistry:
    """
    Singleton — carga modelos .joblib una sola vez al arrancar FastAPI.

    - demand: dict[product_id → Prophet | CrostonModel]
    - price:  Pipeline Ridge global (puede ser None si no hay entrenamiento)
    """

    _instance: ModelRegistry | None = None

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._demand_models: dict[int, DemandModel] = {}
        self._price_pipeline: Pipeline | None = None
        self._demand_manifest: dict[str, Any] = {}
        self._initialized = True

    def load_all(self) -> None:
        """Carga todos los modelos desde disco. Llamar en lifespan startup."""
        self._load_demand_models()
        self._load_price_model()

        croston_count = sum(
            1 for m in self._demand_models.values() if isinstance(m, CrostonModel)
        )
        prophet_count = sum(
            1 for m in self._demand_models.values() if isinstance(m, Prophet)
        )
        logger.info(
            "ModelRegistry: %d Croston/SBA + %d Prophet cargados | Ridge=%s",
            croston_count,
            prophet_count,
            "cargado" if self._price_pipeline else "no disponible",
        )

    def get_demand_model(self, product_id: int) -> DemandModel | None:
        return self._demand_models.get(product_id)

    def get_price_pipeline(self) -> Pipeline | None:
        return self._price_pipeline

    @property
    def demand_model_count(self) -> int:
        return len(self._demand_models)

    @property
    def has_price_model(self) -> bool:
        return self._price_pipeline is not None

    def _load_demand_models(self) -> None:
        if not DEMAND_MODELS_DIR.exists():
            logger.warning("ModelRegistry: directorio demand no existe (%s)", DEMAND_MODELS_DIR)
            return

        if DEMAND_MANIFEST_FILE.exists():
            with DEMAND_MANIFEST_FILE.open(encoding="utf-8") as handle:
                self._demand_manifest = json.load(handle)

        for model_path in sorted(DEMAND_MODELS_DIR.glob("product_*.joblib")):
            product_id_str = model_path.stem.replace("product_", "")
            try:
                product_id = int(product_id_str)
                self._demand_models[product_id] = joblib.load(model_path)
            except (ValueError, OSError) as exc:
                logger.warning("ModelRegistry: no se pudo cargar %s: %s", model_path.name, exc)

    def _load_price_model(self) -> None:
        if not PRICE_MODEL_FILE.exists():
            logger.info("ModelRegistry: no hay modelo Ridge de precio en %s", PRICE_MODEL_FILE)
            return

        try:
            self._price_pipeline = joblib.load(PRICE_MODEL_FILE)
        except OSError as exc:
            logger.warning("ModelRegistry: error cargando precio Ridge: %s", exc)


def get_registry() -> ModelRegistry:
    return ModelRegistry()
