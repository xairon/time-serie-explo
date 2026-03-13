import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.cache import get_redis, pool as redis_pool
from api.config import settings
from api.database import engine, brgm_engine, get_db
from api.json_response import FastJSONResponse
from api.routers import datasets, training, models, forecasting, explainability, counterfactual, db_introspection, pumping_detection, latent_space

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=settings.log_level)

    # Startup: check Redis
    r = get_redis()
    if r is not None:
        try:
            await r.ping()
            logger.info("Redis connection OK")
        except Exception as e:
            logger.warning("Redis ping failed: %s", e)

    yield

    # Shutdown: close connection pools
    if redis_pool is not None:
        await redis_pool.aclose()
    await engine.dispose()
    await brgm_engine.dispose()


app = FastAPI(
    title="Junon Time-Series API",
    version="0.1.0",
    lifespan=lifespan,
    default_response_class=FastJSONResponse,
    docs_url=None if not settings.debug else "/docs",
    redoc_url=None if not settings.debug else "/redoc",
    openapi_url=None if not settings.debug else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)

# Register routers
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(forecasting.router)
app.include_router(explainability.router)
app.include_router(counterfactual.router)
app.include_router(db_introspection.router)
app.include_router(pumping_detection.router)
app.include_router(latent_space.router)


def _check_gpu() -> dict:
    """Check GPU availability (non-async, lightweight)."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
            }
        return {"available": False}
    except ImportError:
        return {"available": False, "reason": "torch not installed"}


@app.get("/api/v1/health")
async def health(db: AsyncSession = Depends(get_db)):
    # Database check
    try:
        await db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")

    # Redis check
    r = get_redis()
    redis_status = "disabled"
    if r is not None:
        try:
            await r.ping()
            redis_status = "ok"
        except Exception:
            redis_status = "unavailable"

    # GPU check
    gpu = _check_gpu()

    return {
        "status": "ok",
        "db": db_status,
        "redis": redis_status,
        "gpu": gpu,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", traceback.format_exc())
    return FastJSONResponse({"detail": "Internal server error"}, status_code=500)
