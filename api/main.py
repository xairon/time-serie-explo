import asyncio
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.cache import get_redis, pool as redis_pool
from dashboard.utils.cache import delete_cached
from api.config import settings
from api.database import engine, brgm_engine, get_db
from api.json_response import FastJSONResponse
from api.routers import datasets, training, models, forecasting, explainability, counterfactual, db_introspection, pumping_detection, pastas
from api.routers import observatory_piezo, observatory_hydro, observatory_common, observatory_era5, observatory_wfs, observatory_bdlisa, observatory_situation, observatory_meteo
from api.routers import auth as auth_router
from api.routers import admin as admin_router
from api.routers import admin_audit as admin_audit_router
from api.auth.deps import get_current_user

logger = logging.getLogger(__name__)

# Re-warm the climatology one day before its 7-day TTL so the cold window never hits.
CLIMATOLOGY_REWARM_SECONDS = 6 * 24 * 3600


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

    # Warm the stations GeoJSON cache so the first map load is fast (best-effort).
    try:
        from api.routers.observatory_common import get_stations_geojson

        await asyncio.to_thread(get_stations_geojson, "all")
        logger.info("Stations GeoJSON cache warmed")
    except Exception as e:
        logger.warning("GeoJSON cache warm failed: %s", e)

    # Warm the météo-des-nappes situation caches (best-effort).
    for _t in ("piezo", "hydro"):
        try:
            await asyncio.to_thread(observatory_situation.get_national_situation, type=_t)
            await asyncio.to_thread(
                observatory_situation.get_territory_situation, level="region", type=_t
            )
        except Exception:
            logger.warning("situation warm-up failed for %s", _t, exc_info=True)

    # Warm the by-sector layer (situation + slider timeline) in the BACKGROUND so it
    # never blocks startup/healthcheck; the precomputed station→sector map keeps each
    # call fast, so even a pre-warm request finishes within the client timeout.
    async def _warm_sectors():
        for _t in ("piezo", "hydro"):
            try:
                await asyncio.to_thread(observatory_situation.get_sector_situation, type=_t)
                await asyncio.to_thread(observatory_situation.get_sector_timeline, type=_t)
            except Exception:
                logger.warning("sector warm-up failed for %s", _t, exc_info=True)
        try:
            await asyncio.to_thread(observatory_meteo.get_brgm_sectors)
            logger.info("BRGM sectors cache warmed")
        except Exception:
            logger.warning("BRGM sectors warm-up failed", exc_info=True)
        try:
            await asyncio.to_thread(observatory_meteo.get_brgm_timeline)
            logger.info("BRGM timeline cache warmed")
        except Exception:
            logger.warning("BRGM timeline warm-up failed", exc_info=True)

    asyncio.create_task(_warm_sectors())

    # Warm the ERA5 climatology caches (temperature + precipitation) in the BACKGROUND
    # (~71s full-table scan each) so the first /temp-anomaly or /anomaly request after
    # a restart or 7-day TTL expiry doesn't hit the scan and exceed the frontend timeout.
    # After the initial warm, re-warm periodically (every 6 days) so the 7-day TTL
    # cold window never occurs in production: delete the key first so the re-warm
    # actually recomputes rather than no-oping on a still-live entry.
    async def _warm_era5_climatology():
        try:
            await asyncio.to_thread(observatory_era5._era5_temp_climatology)
            logger.info("ERA5 temperature climatology cache warmed")
        except Exception:
            logger.warning("ERA5 temperature climatology warm-up failed", exc_info=True)
        try:
            await asyncio.to_thread(observatory_era5._era5_precip_climatology)
            logger.info("ERA5 precipitation climatology cache warmed")
        except Exception:
            logger.warning("ERA5 precipitation climatology warm-up failed", exc_info=True)

        while True:
            await asyncio.sleep(CLIMATOLOGY_REWARM_SECONDS)
            try:
                delete_cached("obs_era5_temp_climatology", {})
                await asyncio.to_thread(observatory_era5._era5_temp_climatology)
                logger.info("ERA5 temperature climatology cache re-warmed")
            except Exception:
                logger.warning("ERA5 temperature climatology re-warm failed", exc_info=True)
            try:
                delete_cached("obs_era5_precip_climatology", {})
                await asyncio.to_thread(observatory_era5._era5_precip_climatology)
                logger.info("ERA5 precipitation climatology cache re-warmed")
            except Exception:
                logger.warning("ERA5 precipitation climatology re-warm failed", exc_info=True)

    asyncio.create_task(_warm_era5_climatology())

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
_auth = [Depends(get_current_user)]

app.include_router(auth_router.router)
app.include_router(admin_router.router)
app.include_router(admin_audit_router.router)
app.include_router(datasets.router, dependencies=_auth)
app.include_router(training.router, dependencies=_auth)
app.include_router(models.router, dependencies=_auth)
app.include_router(forecasting.router, dependencies=_auth)
app.include_router(explainability.router, dependencies=_auth)
app.include_router(counterfactual.router, dependencies=_auth)
app.include_router(db_introspection.router, dependencies=_auth)
app.include_router(pumping_detection.router, dependencies=_auth)
app.include_router(pastas.router, dependencies=_auth)
app.include_router(observatory_piezo.router)
app.include_router(observatory_hydro.router)
app.include_router(observatory_common.router)
app.include_router(observatory_era5.router)
app.include_router(observatory_wfs.router)
app.include_router(observatory_bdlisa.router)
app.include_router(observatory_situation.router)
app.include_router(observatory_meteo.router)


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


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return FastJSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", traceback.format_exc())
    return FastJSONResponse({"detail": "Internal server error"}, status_code=500)
