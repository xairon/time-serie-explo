"""
Junon Time Series - Backend Configuration.

Uses Pydantic Settings for type-safe configuration with environment variable support.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    app_name: str = "Junon Time Series API"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    
    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://localhost:3001"]
    
    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_db: str = "postgres"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres_default_pass_2024"
    
    @property
    def database_url(self) -> str:
        """Construct database URL from components."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    # Celery
    celery_broker_url: Optional[str] = None
    celery_result_backend: Optional[str] = None
    
    @property
    def celery_broker(self) -> str:
        """Get Celery broker URL, defaulting to Redis."""
        return self.celery_broker_url or self.redis_url
    
    @property
    def celery_backend(self) -> str:
        """Get Celery result backend URL, defaulting to Redis."""
        return self.celery_result_backend or self.redis_url
    
    # Storage paths
    data_dir: Path = Path("data")
    checkpoints_dir: Path = Path("checkpoints")
    uploads_dir: Path = Path("uploads")
    
    # Training defaults
    default_train_ratio: float = 0.7
    default_val_ratio: float = 0.15
    max_training_epochs: int = 200


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
