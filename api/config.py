from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode


class Settings(BaseSettings):
    # Internal Junon DB
    db_host: str = "junon-postgres"
    db_port: int = 5432
    db_name: str = "junon_db"
    db_user: str = "junon"
    db_password: str = ""
    # BRGM data warehouse
    brgm_db_host: str = "brgm-postgres"
    brgm_db_port: int = 5432
    brgm_db_name: str = "postgres"
    brgm_db_user: str = "postgres"
    brgm_db_password: str = ""
    # Redis — explicit container name (the bare "redis" alias collides with the
    # BRGM stack's redis on the shared brgm_net network). Overridable via REDIS_URL.
    redis_url: str = "redis://junon-redis:6379/0"
    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:49513"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    debug: bool = False
    # Auth
    jwt_secret: str = "dev-insecure-change-me"
    jwt_alg: str = "HS256"
    session_ttl_hours: int = 12
    cookie_name: str = "junon_session"
    cookie_secure: bool = True
    cookie_samesite: Literal["strict", "lax", "none"] = "strict"

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        # Accept comma-separated env var value (ALLOWED_ORIGINS=http://a,http://b)
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v
    # Paths
    data_dir: str = "/app/data"
    checkpoints_dir: str = "/app/checkpoints"
    results_dir: str = "/app/results"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def brgm_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.brgm_db_user}:{self.brgm_db_password}"
            f"@{self.brgm_db_host}:{self.brgm_db_port}/{self.brgm_db_name}"
        )

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
