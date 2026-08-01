from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


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


@lru_cache
def get_settings() -> Settings:
    return Settings()
