from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.api.predictions import router as predictions_router
from app.core.config import get_settings
from app.ml_models.predictor import DemandForecaster, PriceOptimizer

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    PriceOptimizer()
    DemandForecaster()
    yield


app = FastAPI(
    title=settings.app_name,
    description="Microservicio de predicciones y análisis para el sistema NM.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # En producción Python vive en red interna: Laravel lo llama server-to-server.
    # CORS no aplica a llamadas server-to-server, solo importa para desarrollo local
    # con Swagger UI. En producción restringir o eliminar este middleware.
    allow_origins=["http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

app.include_router(predictions_router, prefix="/api/v1")


def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
    }
    schema["security"] = [{"ApiKeyAuth": []}]
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi  # type: ignore[method-assign]


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Motor de IA encendido",
        "environment": settings.environment,
    }
