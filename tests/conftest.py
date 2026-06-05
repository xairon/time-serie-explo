import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.config import settings
from api.database import get_db
from api.main import app
from api.models_db import Base, User, UserRole
from api.auth.passwords import hash_password

# Tests run over http://test — disable Secure flag so httpx persists cookies
settings.cookie_secure = False

# Disable Redis in tests: rate-limit/lockout helpers no-op cleanly and we avoid
# slow connection timeouts to a non-existent redis host.
import api.cache as _api_cache  # noqa: E402

_api_cache._client = None

# Point data/checkpoint/result dirs at writable temp locations (the defaults like
# /app/data are not writable on dev machines and break registry mkdir()).
import tempfile  # noqa: E402

settings.data_dir = tempfile.mkdtemp(prefix="junon-test-data-")
settings.checkpoints_dir = tempfile.mkdtemp(prefix="junon-test-ckpt-")
settings.results_dir = tempfile.mkdtemp(prefix="junon-test-results-")


@pytest_asyncio.fixture
async def db_sessionmaker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_sessionmaker):
    async def _override_get_db():
        async with db_sessionmaker() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=True
    ) as c:
        yield c
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def db_session(db_sessionmaker):
    async with db_sessionmaker() as session:
        yield session


@pytest_asyncio.fixture
async def auth_client(client, make_user):
    """An AsyncClient already logged in as a regular user (session cookie set)."""
    await make_user(email="testuser@test.fr", password="pw-123456", role=UserRole.user)
    r = await client.post(
        "/api/v1/auth/login", json={"email": "testuser@test.fr", "password": "pw-123456"}
    )
    assert r.status_code == 200, r.text
    return client


@pytest_asyncio.fixture
async def admin_client(client, make_user):
    """An AsyncClient already logged in as an admin (session cookie set)."""
    await make_user(email="testadmin@test.fr", password="pw-123456", role=UserRole.admin)
    r = await client.post(
        "/api/v1/auth/login", json={"email": "testadmin@test.fr", "password": "pw-123456"}
    )
    assert r.status_code == 200, r.text
    return client


@pytest_asyncio.fixture
async def make_user(db_sessionmaker):
    async def _make(email="u@test.fr", password="pw-123456", role=UserRole.user, is_active=True):
        async with db_sessionmaker() as session:
            user = User(
                email=email, display_name=email.split("@")[0],
                password_hash=hash_password(password), role=role, is_active=is_active,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user
    return _make
