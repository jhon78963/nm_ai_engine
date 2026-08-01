from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "message": "Motor de IA encendido",
        "environment": settings.environment,
    }
