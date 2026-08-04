from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DEMAND_MODELS_DIR = MODELS_DIR / "demand"
PRICE_MODELS_DIR = MODELS_DIR / "price"


class Settings(BaseSettings):
    """Configuración central del microservicio de IA."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = Field(default="Novedades Maritex AI", validation_alias="APP_NAME")
    environment: str = Field(default="local", validation_alias="ENVIRONMENT")
    database_url: str = Field(
        default="postgresql+psycopg2://user:pass@localhost:5432/nm_database",
        validation_alias="DATABASE_URL",
        description="URL de conexión para Pandas/SQLAlchemy (misma BD que nm-backend Laravel).",
    )
    api_key: str = Field(
        validation_alias="API_KEY",
        description="Clave compartida con nm-backend para proteger los endpoints. OBLIGATORIO.",
    )
    min_demand_history_days: int = Field(
        default=30,
        validation_alias="MIN_DEMAND_HISTORY_DAYS",
        description="Días mínimos de historial para entrenar Prophet por producto.",
    )
    min_demand_total_sales: int = Field(
        default=5,
        validation_alias="MIN_DEMAND_TOTAL_SALES",
        description="Unidades vendidas mínimas en el historial para entrenar Prophet.",
    )
    min_price_training_rows: int = Field(
        default=10,
        validation_alias="MIN_PRICE_TRAINING_ROWS",
        description="Filas mínimas para entrenar el modelo Ridge de precios.",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
