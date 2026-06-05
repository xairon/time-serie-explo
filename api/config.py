from typing import Annotated, Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode

INSECURE_JWT_DEFAULT = "dev-insecure-change-me"


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
    jwt_secret: str = INSECURE_JWT_DEFAULT
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

    @model_validator(mode="after")
    def _require_strong_secrets(self):
        # Fail closed in production: never sign sessions with the public dev secret.
        # In debug mode the insecure default is tolerated for local development.
        if not self.debug:
            if self.jwt_secret in ("", INSECURE_JWT_DEFAULT) or len(self.jwt_secret) < 32:
                raise ValueError(
                    "JWT_SECRET must be set to a strong value (>= 32 chars) when "
                    "DEBUG is disabled. Generate one with: openssl rand -hex 32"
                )
            if not self.db_password:
                raise ValueError(
                    "DB_PASSWORD must be set when DEBUG is disabled."
                )
        return self
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
