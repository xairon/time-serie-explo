from sqlalchemy import create_engine

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.config import settings

# Internal Junon database
engine = create_async_engine(
    settings.database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={"command_timeout": 30},
)

async_session = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# BRGM data warehouse (gold schema)
brgm_engine = create_async_engine(
    settings.brgm_database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={"command_timeout": 30},
)

brgm_async_session = async_sessionmaker(
    brgm_engine, class_=AsyncSession, expire_on_commit=False
)


async def get_db():
    """Dependency injection for the internal Junon database."""
    async with async_session() as session:
        yield session


async def get_brgm_db():
    """Dependency injection for the BRGM data warehouse."""
    async with brgm_async_session() as session:
        yield session


# Pooled SYNC engine for the BRGM warehouse, shared by the (sync) observatory
# routers. Created lazily so importing this module never opens a connection
# (keeps tests on sqlite untouched). Reuse this instead of create_engine() per
# request, which paid a new-connection setup to the external warehouse each call.
_brgm_sync_engine = None


def get_brgm_sync_engine():
    global _brgm_sync_engine
    if _brgm_sync_engine is None:
        sync_url = (
            f"postgresql://{settings.brgm_db_user}:{settings.brgm_db_password}"
            f"@{settings.brgm_db_host}:{settings.brgm_db_port}/{settings.brgm_db_name}"
        )
        _brgm_sync_engine = create_engine(
            sync_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _brgm_sync_engine
