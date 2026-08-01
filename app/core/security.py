import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> None:
    """
    Dependencia de FastAPI que valida el header X-API-Key.

    - 403 si el header no está presente.
    - 403 si la clave no coincide con settings.api_key.

    Usada como guard en todos los endpoints /api/v1/predict/*.
    nm-backend (Laravel) envía esta clave como AiEngineService::withHeaders(['X-API-Key' => ...]).
    """
    expected = get_settings().api_key

    if api_key is None or not secrets.compare_digest(api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key inválida o no proporcionada.",
        )
