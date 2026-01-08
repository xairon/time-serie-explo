"""
API v1 Router - Aggregates all endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import datasets, sources, models, training, forecasting

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    sources.router,
    prefix="/sources",
    tags=["Data Sources"],
)
api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["Datasets"],
)
api_router.include_router(
    models.router,
    prefix="/models",
    tags=["Models"],
)
api_router.include_router(
    training.router,
    prefix="/training",
    tags=["Training"],
)
api_router.include_router(
    forecasting.router,
    prefix="/forecasting",
    tags=["Forecasting"],
)
