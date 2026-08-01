from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup: aquí se cargarán los modelos ML como singletons
    yield
    # Shutdown: liberar recursos si es necesario


app = FastAPI(
    title="Novedades Maritex - Motor de IA",
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


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok", "message": "Motor de IA encendido"}
