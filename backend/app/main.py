"""
Junon Time Series - FastAPI Application.

Main entry point for the backend API.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.v1.router import api_router

# Initialize logging configuration
try:
    import sys
    from pathlib import Path
    # Add backend dir to path to import logging_setup matching file location
    sys.path.insert(0, str(Path(__file__).parents[1]))
    import logging_setup
except ImportError:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    settings = get_settings()
    
    # Startup: ensure directories exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Shutdown: cleanup if needed


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.app_name,
        description="Time Series Forecasting Platform for Groundwater Data",
        version="0.1.0",
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API router
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "version": "0.1.0"}
    
    return app



# Force reload - Log setup check
logging.getLogger("uvicorn.error").info("Backend Reloaded")

app = create_app()
