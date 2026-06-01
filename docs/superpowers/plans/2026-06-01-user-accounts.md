# User Accounts System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add admin-provisioned email/password accounts so each user manages only their own datasets, models, and scenarios, while the observatory stays public.

**Architecture:** First real table (`users`) in the empty Junon Postgres DB via SQLAlchemy 2.0 async + Alembic. Auth is a short-lived JWT in an `httpOnly`/`Secure`/`SameSite=Strict` cookie, re-verified against the DB (`is_active`, `token_version`) on every request — revocable without a session store. Resource ownership is stamped into native metadata (MLflow tag `owner_id` for models/scenarios, `config.yaml` for datasets) and enforced in the atelier routers; observatory routers are untouched.

**Tech Stack:** FastAPI, SQLAlchemy 2.0 async, Alembic, `pwdlib[argon2]`, PyJWT, React + react-router, pytest + httpx + aiosqlite.

**Spec:** `docs/superpowers/specs/2026-06-01-user-accounts-design.md`

---

## Conventions

- Backend tests run inside the backend container or a venv with the API deps. Test command prefix used below: `pytest`.
- Tests use an in-process ASGI client (`httpx.ASGITransport`) and an SQLite (`aiosqlite`) database via a `get_db` override, so they need neither Postgres nor MLflow.
- The `User.id` is a generic `Uuid` (works on Postgres and SQLite).
- All new backend modules live under `api/auth/` except routers (`api/routers/`).

---

## Phase 1 — Backend foundation (DB, hashing, JWT)

### Task 1: Dependencies and settings

**Files:**
- Modify: `pyproject.toml`
- Modify: `api/config.py`

- [ ] **Step 1: Add runtime + test dependencies**

In `pyproject.toml`, add to the `api` optional-dependencies group (find the existing `[project.optional-dependencies]` `api = [...]` list and append):

```toml
    "pwdlib[argon2]>=0.2.1",
    "pyjwt>=2.9.0",
    "alembic>=1.13.0",
```

And to the test/dev dependencies group (the group used for pytest; if none exists, add it):

```toml
    "aiosqlite>=0.20.0",
    "httpx>=0.27.0",
    "pytest-asyncio>=0.23.0",
```

- [ ] **Step 2: Add auth settings**

In `api/config.py`, add these fields to `Settings` (after `debug: bool = False`):

```python
    # Auth
    jwt_secret: str = "dev-insecure-change-me"
    jwt_alg: str = "HS256"
    session_ttl_hours: int = 12
    cookie_name: str = "junon_session"
    cookie_secure: bool = True
    cookie_samesite: Literal["strict", "lax", "none"] = "strict"
```

- [ ] **Step 3: Verify config imports**

Run: `python -c "from api.config import settings; print(settings.jwt_alg, settings.session_ttl_hours)"`
Expected: `HS256 12`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml api/config.py
git commit -m "feat(auth): add auth deps and settings"
```

---

### Task 2: SQLAlchemy Base and User model

**Files:**
- Create: `api/models_db/__init__.py`
- Create: `api/models_db/base.py`
- Create: `api/models_db/user.py`

- [ ] **Step 1: Create the declarative base**

`api/models_db/base.py`:

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

- [ ] **Step 2: Create the User model**

`api/models_db/user.py`:

```python
import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from api.models_db.base import Base


class UserRole(str, enum.Enum):
    admin = "admin"
    user = "user"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole, name="user_role"), default=UserRole.user, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    token_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

- [ ] **Step 3: Create the package init**

`api/models_db/__init__.py`:

```python
from api.models_db.base import Base
from api.models_db.user import User, UserRole

__all__ = ["Base", "User", "UserRole"]
```

- [ ] **Step 4: Verify model imports and table name**

Run: `python -c "from api.models_db import User; print(User.__tablename__, [c.name for c in User.__table__.columns])"`
Expected: `users ['id', 'email', 'display_name', 'password_hash', 'role', 'is_active', 'token_version', 'created_at', 'last_login_at']`

- [ ] **Step 5: Commit**

```bash
git add api/models_db/
git commit -m "feat(auth): add User SQLAlchemy model"
```

---

### Task 3: Alembic setup and users migration

**Files:**
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`
- Create: `alembic/versions/0001_users.py`

- [ ] **Step 1: Create `alembic.ini`** (minimal — URL is built in env.py)

```ini
[alembic]
script_location = alembic
prepend_sys_path = .

[loggers]
keys = root

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARNING
handlers = console

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

- [ ] **Step 2: Create `alembic/script.py.mako`**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 3: Create `alembic/env.py`** (async, uses app settings + metadata)

```python
import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from api.config import settings
from api.models_db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online():
    asyncio.run(run_async_migrations())


run_migrations_online()
```

- [ ] **Step 4: Create the first migration `alembic/versions/0001_users.py`**

```python
"""users table

Revision ID: 0001_users
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_users"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.Enum("admin", "user", name="user_role"), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("token_version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    sa.Enum(name="user_role").drop(op.get_bind(), checkfirst=True)
```

- [ ] **Step 5: Apply the migration against the running Postgres**

Run (from repo root, with the backend stack up): `alembic upgrade head`
Expected: no error; `psql` shows a `users` table. Verify:
Run: `docker exec junon-postgres psql -U junon -d junon_db -c "\dt"`
Expected: lists `users`.

- [ ] **Step 6: Commit**

```bash
git add alembic.ini alembic/
git commit -m "feat(auth): alembic setup + users table migration"
```

---

### Task 4: Password hashing utility

**Files:**
- Create: `api/auth/__init__.py` (empty)
- Create: `api/auth/passwords.py`
- Test: `tests/auth/test_passwords.py`

- [ ] **Step 1: Write the failing test**

`tests/auth/test_passwords.py`:

```python
from api.auth.passwords import hash_password, verify_password


def test_hash_and_verify_roundtrip():
    h = hash_password("s3cret-pass")
    assert h != "s3cret-pass"
    assert verify_password("s3cret-pass", h) is True


def test_verify_rejects_wrong_password():
    h = hash_password("s3cret-pass")
    assert verify_password("wrong", h) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/auth/test_passwords.py -v`
Expected: FAIL with `ModuleNotFoundError: api.auth.passwords`

- [ ] **Step 3: Implement**

`api/auth/passwords.py`:

```python
from pwdlib import PasswordHash

_hasher = PasswordHash.recommended()


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _hasher.verify(plain, hashed)
```

Also create an empty `api/auth/__init__.py` and `tests/auth/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/auth/test_passwords.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add api/auth/__init__.py api/auth/passwords.py tests/auth/
git commit -m "feat(auth): argon2 password hashing"
```

---

### Task 5: JWT encode/decode utility

**Files:**
- Create: `api/auth/tokens.py`
- Test: `tests/auth/test_tokens.py`

- [ ] **Step 1: Write the failing test**

`tests/auth/test_tokens.py`:

```python
import uuid

import pytest

from api.auth.tokens import create_session_token, decode_session_token


def test_roundtrip_encodes_subject_and_token_version():
    uid = uuid.uuid4()
    token = create_session_token(uid, token_version=3)
    claims = decode_session_token(token)
    assert claims["sub"] == str(uid)
    assert claims["tv"] == 3


def test_decode_rejects_garbage():
    with pytest.raises(Exception):
        decode_session_token("not-a-jwt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/auth/test_tokens.py -v`
Expected: FAIL with `ModuleNotFoundError: api.auth.tokens`

- [ ] **Step 3: Implement**

`api/auth/tokens.py`:

```python
import uuid
from datetime import datetime, timedelta, timezone

import jwt

from api.config import settings


def create_session_token(user_id: uuid.UUID, token_version: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "tv": token_version,
        "iat": now,
        "exp": now + timedelta(hours=settings.session_ttl_hours),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_alg)


def decode_session_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/auth/test_tokens.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add api/auth/tokens.py tests/auth/test_tokens.py
git commit -m "feat(auth): JWT session token utils"
```

---

## Phase 2 — Auth dependencies, endpoints, gating

### Task 6: Test harness (async client + sqlite DB)

**Files:**
- Create: `tests/conftest.py` (or extend if it exists — check first)

- [ ] **Step 1: Inspect existing conftest**

Run: `cat tests/conftest.py 2>/dev/null || echo "none"`
If it exists, merge the fixtures below rather than overwriting.

- [ ] **Step 2: Write the fixtures**

`tests/conftest.py`:

```python
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.database import get_db
from api.main import app
from api.models_db import Base, User, UserRole
from api.auth.passwords import hash_password


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
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


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
```

- [ ] **Step 3: Smoke-test the harness**

Add `tests/auth/test_harness.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_health_reachable(client):
    res = await client.get("/api/v1/health")
    assert res.status_code in (200, 503)  # 503 if redis/db checks differ under sqlite
```

Run: `pytest tests/auth/test_harness.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/auth/test_harness.py
git commit -m "test(auth): async client + sqlite db fixtures"
```

---

### Task 7: Auth dependencies (`get_current_user`, `require_admin`)

**Files:**
- Create: `api/auth/deps.py`
- Test: `tests/auth/test_deps.py`

- [ ] **Step 1: Write the failing test**

`tests/auth/test_deps.py`:

```python
import pytest

from api.auth.tokens import create_session_token


@pytest.mark.asyncio
async def test_me_requires_cookie(client):
    res = await client.get("/api/v1/auth/me")
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_me_returns_user_with_valid_cookie(client, make_user):
    user = await make_user(email="alice@test.fr")
    token = create_session_token(user.id, user.token_version)
    res = await client.get("/api/v1/auth/me", cookies={"junon_session": token})
    assert res.status_code == 200
    assert res.json()["email"] == "alice@test.fr"


@pytest.mark.asyncio
async def test_revoked_by_token_version(client, make_user):
    user = await make_user(email="bob@test.fr")
    stale = create_session_token(user.id, user.token_version - 1)
    res = await client.get("/api/v1/auth/me", cookies={"junon_session": stale})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_inactive_user_rejected(client, make_user):
    user = await make_user(email="carol@test.fr", is_active=False)
    token = create_session_token(user.id, user.token_version)
    res = await client.get("/api/v1/auth/me", cookies={"junon_session": token})
    assert res.status_code == 401
```

(These depend on Task 8's `/auth/me`; this test file is committed together with Task 8.)

- [ ] **Step 2: Implement the dependencies**

`api/auth/deps.py`:

```python
import uuid

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.database import get_db
from api.models_db import User, UserRole
from api.auth.tokens import decode_session_token


async def get_current_user(
    request: Request, db: AsyncSession = Depends(get_db)
) -> User:
    token = request.cookies.get(settings.cookie_name)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        claims = decode_session_token(token)
        user_id = uuid.UUID(claims["sub"])
        tv = int(claims["tv"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session")

    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None or not user.is_active or user.token_version != tv:
        raise HTTPException(status_code=401, detail="Session expired")
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Admin only")
    return user
```

- [ ] **Step 3: Commit (tests run in Task 8)**

```bash
git add api/auth/deps.py tests/auth/test_deps.py
git commit -m "feat(auth): current-user and admin dependencies"
```

---

### Task 8: Auth router (login / logout / me / change-password)

**Files:**
- Create: `api/auth/schemas.py`
- Create: `api/routers/auth.py`
- Modify: `api/main.py` (register router)
- Test: `tests/auth/test_auth_flow.py`

- [ ] **Step 1: Write the failing test**

`tests/auth/test_auth_flow.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_login_sets_cookie_and_me_works(client, make_user):
    await make_user(email="dave@test.fr", password="pw-123456")
    res = await client.post("/api/v1/auth/login", json={"email": "dave@test.fr", "password": "pw-123456"})
    assert res.status_code == 200
    assert "junon_session" in res.cookies
    me = await client.get("/api/v1/auth/me")
    assert me.status_code == 200 and me.json()["email"] == "dave@test.fr"


@pytest.mark.asyncio
async def test_login_wrong_password(client, make_user):
    await make_user(email="erin@test.fr", password="pw-123456")
    res = await client.post("/api/v1/auth/login", json={"email": "erin@test.fr", "password": "nope"})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_login_inactive(client, make_user):
    await make_user(email="frank@test.fr", password="pw-123456", is_active=False)
    res = await client.post("/api/v1/auth/login", json={"email": "frank@test.fr", "password": "pw-123456"})
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_logout_clears_cookie(client, make_user):
    await make_user(email="gina@test.fr", password="pw-123456")
    await client.post("/api/v1/auth/login", json={"email": "gina@test.fr", "password": "pw-123456"})
    res = await client.post("/api/v1/auth/logout")
    assert res.status_code == 204
    me = await client.get("/api/v1/auth/me")
    assert me.status_code == 401
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/auth/test_auth_flow.py -v`
Expected: FAIL (404 on `/api/v1/auth/login`)

- [ ] **Step 3: Implement schemas**

`api/auth/schemas.py`:

```python
import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: EmailStr
    display_name: str
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None

    model_config = {"from_attributes": True}
```

- [ ] **Step 4: Implement the router**

`api/routers/auth.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import func

from api.config import settings
from api.database import get_db
from api.models_db import User
from api.auth.deps import get_current_user
from api.auth.passwords import hash_password, verify_password
from api.auth.schemas import ChangePasswordRequest, LoginRequest, UserOut
from api.auth.tokens import create_session_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _set_session_cookie(response: Response, user: User) -> None:
    token = create_session_token(user.id, user.token_version)
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=settings.session_ttl_hours * 3600,
        path="/",
    )


@router.post("/login", response_model=UserOut)
async def login(req: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.email == req.email))).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    user.last_login_at = func.now()
    await db.commit()
    await db.refresh(user)
    _set_session_cookie(response, user)
    return user


@router.post("/logout", status_code=204)
async def logout(response: Response):
    response.delete_cookie(settings.cookie_name, path="/")


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return user


@router.post("/change-password", status_code=204)
async def change_password(
    req: ChangePasswordRequest,
    response: Response,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not verify_password(req.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Ancien mot de passe incorrect")
    user.password_hash = hash_password(req.new_password)
    user.token_version += 1
    await db.commit()
    await db.refresh(user)
    _set_session_cookie(response, user)
```

- [ ] **Step 5: Register the router in `api/main.py`**

Add to the imports block (line 14-15 area):

```python
from api.routers import auth as auth_router
```

And after `app = FastAPI(...)` router registrations, add as the FIRST registration:

```python
app.include_router(auth_router.router)
```

- [ ] **Step 6: Run all auth tests**

Run: `pytest tests/auth/ -v`
Expected: PASS (Task 7 + Task 8 tests all green)

- [ ] **Step 7: Commit**

```bash
git add api/auth/schemas.py api/routers/auth.py api/main.py tests/auth/test_auth_flow.py
git commit -m "feat(auth): login/logout/me/change-password endpoints"
```

---

### Task 9: Gate the atelier routers

**Files:**
- Modify: `api/main.py`
- Test: `tests/auth/test_gating.py`

- [ ] **Step 1: Write the failing test**

`tests/auth/test_gating.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_atelier_requires_auth(client):
    # datasets list is in the atelier
    res = await client.get("/api/v1/datasets")
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_observatory_is_public(client):
    # Any observatory GET must NOT return 401 (may be 200/404/503 but never 401)
    res = await client.get("/api/v1/observatory/hydro/alerts")
    assert res.status_code != 401
```

- [ ] **Step 2: Run to verify the first assertion fails**

Run: `pytest tests/auth/test_gating.py -v`
Expected: `test_atelier_requires_auth` FAILS (currently returns 200)

- [ ] **Step 3: Apply the gating dependency in `api/main.py`**

Replace the atelier router registrations (the `datasets`…`pastas` block) with the dependency-guarded form. Keep observatory registrations unchanged.

```python
from fastapi import Depends
from api.auth.deps import get_current_user

_auth = [Depends(get_current_user)]

app.include_router(auth_router.router)
app.include_router(datasets.router, dependencies=_auth)
app.include_router(training.router, dependencies=_auth)
app.include_router(models.router, dependencies=_auth)
app.include_router(forecasting.router, dependencies=_auth)
app.include_router(explainability.router, dependencies=_auth)
app.include_router(counterfactual.router, dependencies=_auth)
app.include_router(db_introspection.router, dependencies=_auth)
app.include_router(pumping_detection.router, dependencies=_auth)
app.include_router(pastas.router, dependencies=_auth)
# Observatory routers stay public (no dependencies)
app.include_router(observatory_piezo.router)
app.include_router(observatory_hydro.router)
app.include_router(observatory_common.router)
app.include_router(observatory_era5.router)
app.include_router(observatory_wfs.router)
app.include_router(observatory_bdlisa.router)
```

(`Depends` is already imported in main.py; ensure `get_current_user` import is added.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/auth/test_gating.py -v`
Expected: PASS (both)

- [ ] **Step 5: Commit**

```bash
git add api/main.py tests/auth/test_gating.py
git commit -m "feat(auth): gate atelier routers, keep observatory public"
```

---

### Task 10: Bootstrap admin script

**Files:**
- Create: `scripts/create_admin.py`

- [ ] **Step 1: Implement**

`scripts/create_admin.py`:

```python
"""Create (or promote) an admin user. Usage:
    python scripts/create_admin.py --email a@b.fr --name "Nicolas" [--password ...]
If --password is omitted, prompts securely.
"""
import argparse
import asyncio
import getpass

from sqlalchemy import select

from api.database import async_session
from api.models_db import User, UserRole
from api.auth.passwords import hash_password


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--email", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--password")
    args = p.parse_args()
    password = args.password or getpass.getpass("Password: ")

    async with async_session() as db:
        existing = (await db.execute(select(User).where(User.email == args.email))).scalar_one_or_none()
        if existing:
            existing.role = UserRole.admin
            existing.password_hash = hash_password(password)
            existing.is_active = True
            print(f"Promoted existing user {args.email} to admin.")
        else:
            db.add(User(
                email=args.email, display_name=args.name,
                password_hash=hash_password(password), role=UserRole.admin, is_active=True,
            ))
            print(f"Created admin {args.email}.")
        await db.commit()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run it against the live DB**

Run: `docker exec -it junon-backend python scripts/create_admin.py --email ringuetnicolas2@gmail.com --name "Nicolas Ringuet" --password "<choose>"`
Expected: `Created admin ...`. Verify: `docker exec junon-postgres psql -U junon -d junon_db -c "SELECT email, role FROM users;"`

- [ ] **Step 3: Commit**

```bash
git add scripts/create_admin.py
git commit -m "feat(auth): create_admin bootstrap script"
```

---

## Phase 3 — Admin user management

### Task 11: Admin users router

**Files:**
- Create: `api/routers/admin.py`
- Modify: `api/main.py` (register, admin-gated)
- Test: `tests/auth/test_admin.py`

- [ ] **Step 1: Write the failing test**

`tests/auth/test_admin.py`:

```python
import pytest

from api.auth.tokens import create_session_token
from api.models_db import UserRole


def _cookies(user):
    return {"junon_session": create_session_token(user.id, user.token_version)}


@pytest.mark.asyncio
async def test_user_cannot_list_users(client, make_user):
    u = await make_user(email="plain@test.fr", role=UserRole.user)
    res = await client.get("/api/v1/admin/users", cookies=_cookies(u))
    assert res.status_code == 403


@pytest.mark.asyncio
async def test_admin_can_create_and_list(client, make_user):
    admin = await make_user(email="admin@test.fr", role=UserRole.admin)
    create = await client.post(
        "/api/v1/admin/users", cookies=_cookies(admin),
        json={"email": "new@test.fr", "display_name": "New", "role": "user", "initial_password": "pw-123456"},
    )
    assert create.status_code == 201
    listed = await client.get("/api/v1/admin/users", cookies=_cookies(admin))
    emails = [u["email"] for u in listed.json()]
    assert "new@test.fr" in emails


@pytest.mark.asyncio
async def test_admin_can_disable_user(client, make_user):
    admin = await make_user(email="admin2@test.fr", role=UserRole.admin)
    target = await make_user(email="target@test.fr", role=UserRole.user)
    res = await client.patch(
        f"/api/v1/admin/users/{target.id}", cookies=_cookies(admin),
        json={"is_active": False},
    )
    assert res.status_code == 200
    assert res.json()["is_active"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/auth/test_admin.py -v`
Expected: FAIL (404)

- [ ] **Step 3: Implement the router**

`api/routers/admin.py`:

```python
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models_db import User, UserRole
from api.auth.deps import require_admin
from api.auth.passwords import hash_password
from api.auth.schemas import UserOut

router = APIRouter(prefix="/api/v1/admin/users", tags=["admin"])


class CreateUserRequest(BaseModel):
    email: EmailStr
    display_name: str
    role: UserRole = UserRole.user
    initial_password: str


class UpdateUserRequest(BaseModel):
    is_active: bool | None = None
    role: UserRole | None = None
    new_password: str | None = None


@router.get("", response_model=list[UserOut])
async def list_users(_: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    return list((await db.execute(select(User).order_by(User.created_at))).scalars())


@router.post("", response_model=UserOut, status_code=201)
async def create_user(req: CreateUserRequest, _: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    exists = (await db.execute(select(User).where(User.email == req.email))).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=409, detail="Email déjà utilisé")
    user = User(
        email=req.email, display_name=req.display_name,
        password_hash=hash_password(req.initial_password), role=req.role, is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.patch("/{user_id}", response_model=UserOut)
async def update_user(user_id: uuid.UUID, req: UpdateUserRequest, _: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    if req.is_active is not None:
        user.is_active = req.is_active
        user.token_version += 1
    if req.role is not None:
        user.role = req.role
    if req.new_password is not None:
        user.password_hash = hash_password(req.new_password)
        user.token_version += 1
    await db.commit()
    await db.refresh(user)
    return user
```

- [ ] **Step 4: Register in `api/main.py`**

```python
from api.routers import admin as admin_router
# require_admin already enforced inside the router via Depends(require_admin)
app.include_router(admin_router.router)
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/auth/test_admin.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add api/routers/admin.py api/main.py tests/auth/test_admin.py
git commit -m "feat(auth): admin user-management endpoints"
```

---

## Phase 4 — Ownership stamping & enforcement

### Task 12: Ownership helpers

**Files:**
- Create: `api/auth/ownership.py`
- Test: `tests/auth/test_ownership.py`

- [ ] **Step 1: Write the failing test** (pure logic, no MLflow)

`tests/auth/test_ownership.py`:

```python
import uuid

import pytest

from api.auth.ownership import is_owner_or_admin
from api.models_db import User, UserRole


def _user(role=UserRole.user):
    u = User(id=uuid.uuid4(), email="x@y.fr", display_name="x", password_hash="h", role=role, is_active=True, token_version=0)
    return u


def test_owner_matches():
    u = _user()
    assert is_owner_or_admin(u, str(u.id)) is True


def test_other_owner_denied():
    u = _user()
    assert is_owner_or_admin(u, str(uuid.uuid4())) is False


def test_admin_always_allowed():
    u = _user(role=UserRole.admin)
    assert is_owner_or_admin(u, str(uuid.uuid4())) is True


def test_missing_owner_denied_for_user_allowed_for_admin():
    assert is_owner_or_admin(_user(), None) is False
    assert is_owner_or_admin(_user(UserRole.admin), None) is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/auth/test_ownership.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement**

`api/auth/ownership.py`:

```python
from fastapi import HTTPException

from api.models_db import User, UserRole


def is_owner_or_admin(user: User, owner_id: str | None) -> bool:
    if user.role == UserRole.admin:
        return True
    if owner_id is None:
        return False
    return str(user.id) == str(owner_id)


def assert_owner_or_admin(user: User, owner_id: str | None) -> None:
    """Raise 404 (not 403) so we do not disclose existence of others' resources."""
    if not is_owner_or_admin(user, owner_id):
        raise HTTPException(status_code=404, detail="Introuvable")


def owner_filter_clause(user: User) -> str | None:
    """MLflow search filter for the current user, or None for admin (no filter)."""
    if user.role == UserRole.admin:
        return None
    return f"tags.owner_id = '{user.id}'"
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/auth/test_ownership.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add api/auth/ownership.py tests/auth/test_ownership.py
git commit -m "feat(auth): ownership helper logic"
```

---

### Task 13: Stamp & enforce ownership on MODELS

**Files:**
- Modify: `dashboard/utils/mlflow_client.py:108-133` (start_run — owner tag)
- Modify: `api/routers/training.py` (capture current_user, pass owner_id into the run)
- Modify: `api/routers/models.py` (filter list, enforce get/delete/test-info/download)
- Test: `tests/auth/test_models_ownership.py`

- [ ] **Step 1: Add `owner_id` to `start_run`**

In `dashboard/utils/mlflow_client.py`, change `start_run` to accept `owner_id` and set it as a tag (replace the legacy `tags['user'] = ...` line):

```python
    def start_run(self, run_name=None, tags: Optional[Dict] = None, owner_id: Optional[str] = None, nested: bool = False):
        tags = tags or {}
        tags['framework'] = self.framework
        if owner_id is not None:
            tags['owner_id'] = owner_id
        return mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
```

(Adjust kwargs to match the existing signature exactly; the key change is the `owner_id` tag.)

- [ ] **Step 2: Thread the owner through training**

In `api/routers/training.py`:
- Add `current=Depends(get_current_user)` to `start_training` (import `from api.auth.deps import get_current_user` and `from api.models_db import User`).
- Capture `owner_id = str(current.id)` and pass it into `_run_training_thread` (extend its signature `(task_id, req, owner_id)`), which forwards `owner_id` into the `mlflow_manager.start_run(..., owner_id=owner_id)` call.

```python
@router.post("/start", response_model=TrainingStatus, status_code=202)
async def start_training(req: TrainingRequest, current: "User" = Depends(get_current_user)):
    ...
    threading.Thread(target=_run_training_thread, args=(task_id, req, str(current.id)), daemon=True).start()
    ...
```

- [ ] **Step 3: Write the failing test for list/get enforcement**

`tests/auth/test_models_ownership.py` (mocks the registry so MLflow is not needed):

```python
import uuid
import pytest
from unittest.mock import patch

from api.auth.tokens import create_session_token
from api.models_db import UserRole


def _cookies(u):
    return {"junon_session": create_session_token(u.id, u.token_version)}


@pytest.mark.asyncio
async def test_list_models_filters_by_owner(client, make_user):
    u = await make_user(email="owner@test.fr")
    captured = {}

    class FakeRegistry:
        def list_models(self, owner_filter=None):
            captured["filter"] = owner_filter
            return []

    with patch("api.routers.models._get_model_registry", return_value=FakeRegistry()):
        res = await client.get("/api/v1/models", cookies=_cookies(u))
    assert res.status_code == 200
    assert captured["filter"] == f"tags.owner_id = '{u.id}'"


@pytest.mark.asyncio
async def test_get_model_of_other_user_404(client, make_user):
    u = await make_user(email="intruder@test.fr", role=UserRole.user)

    class FakeRegistry:
        def get_model_owner(self, model_id):
            return str(uuid.uuid4())  # belongs to someone else

    with patch("api.routers.models._get_model_registry", return_value=FakeRegistry()):
        res = await client.get(f"/api/v1/models/{uuid.uuid4()}", cookies=_cookies(u))
    assert res.status_code == 404
```

- [ ] **Step 4: Implement enforcement in `api/routers/models.py`**

- Add imports: `from api.auth.deps import get_current_user`, `from api.auth.ownership import assert_owner_or_admin, owner_filter_clause`, `from api.models_db import User`.
- Add a `get_model_owner(model_id) -> str | None` method to the registry (reads the `owner_id` tag of the run) and an `owner_filter` argument to `list_models`.
- Update endpoints:

```python
@router.get("", response_model=list[ModelSummary])
async def list_models(current: User = Depends(get_current_user)):
    reg = _get_model_registry()
    return reg.list_models(owner_filter=owner_filter_clause(current))


@router.get("/{model_id}", response_model=ModelDetail)
async def get_model(model_id: str, current: User = Depends(get_current_user)):
    reg = _get_model_registry()
    assert_owner_or_admin(current, reg.get_model_owner(model_id))
    ...  # existing body
```

Apply the same `assert_owner_or_admin(current, reg.get_model_owner(model_id))` guard at the top of `delete_model`, `get_model_test_info`, `download_model`, and the `/{model_id}/...` explainability-style endpoints in this router.

- In `dashboard/utils/model_registry.py`, implement `get_model_owner` (read run tag `owner_id`) and make `list_models` accept and forward `owner_filter` to `search_runs(filter_string=owner_filter or "")`.

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/auth/test_models_ownership.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add dashboard/utils/mlflow_client.py dashboard/utils/model_registry.py api/routers/training.py api/routers/models.py tests/auth/test_models_ownership.py
git commit -m "feat(auth): stamp + enforce model ownership"
```

---

### Task 14: Stamp & enforce ownership on DATASETS

**Files:**
- Modify: `dashboard/utils/dataset_registry.py` (write/read `owner_id`, filter list)
- Modify: `api/routers/datasets.py` (pass owner on create, filter list, enforce access/delete)
- Test: `tests/auth/test_datasets_ownership.py`

- [ ] **Step 1: Registry support**

In `dashboard/utils/dataset_registry.py`:
- When creating a dataset, accept an `owner_id: str | None` and write it into `config.yaml` (`config['owner_id'] = owner_id`).
- Add `list(owner_id=None)` filtering: skip datasets whose `config.get('owner_id')` != `owner_id` unless `owner_id is None` (admin).
- Add `get_owner(dataset_id) -> str | None` reading `config.yaml`.

- [ ] **Step 2: Write the failing test**

`tests/auth/test_datasets_ownership.py`:

```python
import uuid
import pytest
from unittest.mock import patch

from api.auth.tokens import create_session_token


def _cookies(u):
    return {"junon_session": create_session_token(u.id, u.token_version)}


@pytest.mark.asyncio
async def test_list_datasets_filtered_by_owner(client, make_user):
    u = await make_user(email="dsowner@test.fr")
    captured = {}

    class FakeReg:
        def list(self, owner_id=None):
            captured["owner_id"] = owner_id
            return []

    with patch("api.routers.datasets._get_registry", return_value=FakeReg()):
        res = await client.get("/api/v1/datasets", cookies=_cookies(u))
    assert res.status_code == 200
    assert captured["owner_id"] == str(u.id)


@pytest.mark.asyncio
async def test_get_other_dataset_404(client, make_user):
    u = await make_user(email="dsintruder@test.fr")

    class FakeReg:
        def get_owner(self, dataset_id):
            return str(uuid.uuid4())

    with patch("api.routers.datasets._get_registry", return_value=FakeReg()):
        res = await client.get("/api/v1/datasets/whatever", cookies=_cookies(u))
    assert res.status_code == 404
```

(Adjust `_get_registry` to the actual accessor name in `datasets.py`; if the router builds the registry inline, refactor it into a `_get_registry()` helper first — this also improves testability.)

- [ ] **Step 3: Run to verify it fails**

Run: `pytest tests/auth/test_datasets_ownership.py -v`
Expected: FAIL

- [ ] **Step 4: Implement enforcement in `api/routers/datasets.py`**

- Add `current: User = Depends(get_current_user)` to list/get/delete/profile/preview and the create/import endpoints.
- List: `reg.list(owner_id=None if current.role==admin else str(current.id))` — use `owner_filter_clause`-style logic via a small inline `None if admin else str(id)`.
- Create/import: pass `owner_id=str(current.id)` to the registry create call.
- Get/preview/profile/delete: `assert_owner_or_admin(current, reg.get_owner(dataset_id))` at the top.

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/auth/test_datasets_ownership.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add dashboard/utils/dataset_registry.py api/routers/datasets.py tests/auth/test_datasets_ownership.py
git commit -m "feat(auth): stamp + enforce dataset ownership"
```

---

### Task 15: Enforce ownership on derived endpoints & scenarios

**Files:**
- Modify: `api/routers/forecasting.py`, `api/routers/explainability.py`, `api/routers/counterfactual.py`, `api/routers/pastas.py`
- Test: `tests/auth/test_derived_ownership.py`

- [ ] **Step 1: Identify the model/dataset reference in each endpoint**

For each route that takes a `model_id`/`run_id` or `dataset_id`, add `current: User = Depends(get_current_user)` and guard with the matching helper before doing work:
- model-referencing: `assert_owner_or_admin(current, model_registry.get_model_owner(model_id))`
- dataset-referencing: `assert_owner_or_admin(current, dataset_registry.get_owner(dataset_id))`
- pastas scenarios: a scenario lives under a model `run_id` → guard with `get_model_owner(run_id)`.

- [ ] **Step 2: Write the failing test (forecasting on another user's model → 404)**

`tests/auth/test_derived_ownership.py`:

```python
import uuid
import pytest
from unittest.mock import patch

from api.auth.tokens import create_session_token


def _cookies(u):
    return {"junon_session": create_session_token(u.id, u.token_version)}


@pytest.mark.asyncio
async def test_forecast_on_foreign_model_404(client, make_user):
    u = await make_user(email="fc@test.fr")

    class FakeReg:
        def get_model_owner(self, model_id):
            return str(uuid.uuid4())

    with patch("api.routers.forecasting._get_model_registry", return_value=FakeReg()):
        res = await client.post(
            "/api/v1/forecasting/predict",
            json={"model_id": str(uuid.uuid4()), "horizon": 10},
            cookies=_cookies(u),
        )
    assert res.status_code == 404
```

(Adjust endpoint path/body and registry accessor to the real ones in `forecasting.py`.)

- [ ] **Step 3: Implement guards, run tests**

Run: `pytest tests/auth/test_derived_ownership.py -v`
Expected: PASS

- [ ] **Step 4: Full backend auth suite**

Run: `pytest tests/auth/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add api/routers/forecasting.py api/routers/explainability.py api/routers/counterfactual.py api/routers/pastas.py tests/auth/test_derived_ownership.py
git commit -m "feat(auth): enforce ownership on derived endpoints and scenarios"
```

---

## Phase 5 — Migrate existing data

### Task 16: Assign legacy ownership

**Files:**
- Create: `scripts/assign_legacy_ownership.py`

- [ ] **Step 1: Implement**

`scripts/assign_legacy_ownership.py`:

```python
"""Assign all ownerless models (MLflow runs) and datasets to a given admin email.
Usage: python scripts/assign_legacy_ownership.py --email admin@x.fr
"""
import argparse
import asyncio
from pathlib import Path

import yaml
from sqlalchemy import select

from api.config import settings
from api.database import async_session
from api.models_db import User
from dashboard.utils.mlflow_client import get_mlflow_manager


async def _resolve_owner(email: str) -> str:
    async with async_session() as db:
        user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
        if user is None:
            raise SystemExit(f"No user with email {email}")
        return str(user.id)


def _tag_models(owner_id: str) -> int:
    mgr = get_mlflow_manager()
    import mlflow
    runs = mgr.search_runs(filter_string="")
    n = 0
    for run in runs:
        if "owner_id" not in run.data.tags:
            mlflow.set_tag("owner_id", owner_id)  # within run context; adapt to client.set_tag(run_id, ...)
            n += 1
    return n


def _tag_datasets(owner_id: str) -> int:
    prepared = Path(settings.data_dir) / "prepared"
    n = 0
    for cfg in prepared.rglob("config.yaml"):
        data = yaml.safe_load(cfg.read_text()) or {}
        if not data.get("owner_id"):
            data["owner_id"] = owner_id
            cfg.write_text(yaml.safe_dump(data))
            n += 1
    return n


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--email", required=True)
    args = p.parse_args()
    owner_id = await _resolve_owner(args.email)
    print(f"Models tagged: {_tag_models(owner_id)}")
    print(f"Datasets tagged: {_tag_datasets(owner_id)}")


if __name__ == "__main__":
    asyncio.run(main())
```

(For MLflow tagging outside a run context, use the tracking client: `MlflowClient().set_tag(run_id, 'owner_id', owner_id)` — adapt `_tag_models` to iterate `run.info.run_id`.)

- [ ] **Step 2: Run against the live stack**

Run: `docker exec junon-backend python scripts/assign_legacy_ownership.py --email ringuetnicolas2@gmail.com`
Expected: prints counts; existing models/datasets now visible to that admin.

- [ ] **Step 3: Commit**

```bash
git add scripts/assign_legacy_ownership.py
git commit -m "feat(auth): legacy ownership assignment script"
```

---

## Phase 6 — Frontend

### Task 17: Send cookies + handle 401

**Files:**
- Modify: `frontend/src/lib/api.ts:39-48` (add `credentials: 'include'`)

- [ ] **Step 1: Add credentials to fetch**

In `fetchJson`, add `credentials: 'include'` to the `fetch` options, and the same in the `deleteJson` raw fetch. On a 401 response, dispatch a global event so the app can redirect:

```typescript
    const res = await fetch(url, {
      ...init,
      credentials: 'include',
      signal: controller.signal,
      headers: { 'Accept': 'application/json', ...init?.headers },
    })
    if (res.status === 401) {
      window.dispatchEvent(new CustomEvent('auth:unauthorized'))
    }
    if (!res.ok) { /* existing error handling */ }
```

- [ ] **Step 2: Build the frontend to verify it compiles**

Run: `cd frontend && npm run build`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat(auth): send session cookie + emit 401 event"
```

---

### Task 18: Auth API + context

**Files:**
- Create: `frontend/src/lib/auth.ts`
- Create: `frontend/src/contexts/AuthContext.tsx`
- Modify: `frontend/src/App.tsx` (wrap with AuthProvider)

- [ ] **Step 1: Auth API helpers**

`frontend/src/lib/auth.ts`:

```typescript
import { API_BASE } from './constants'

export interface AuthUser {
  id: string; email: string; display_name: string; role: 'admin' | 'user'; is_active: boolean
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    credentials: 'include',
    headers: { 'Accept': 'application/json', 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) throw new Error(`${res.status}`)
  return (res.status === 204 ? undefined : await res.json()) as T
}

export const authApi = {
  me: () => req<AuthUser>('/auth/me'),
  login: (email: string, password: string) =>
    req<AuthUser>('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  logout: () => req<void>('/auth/logout', { method: 'POST' }),
}
```

- [ ] **Step 2: Auth context**

`frontend/src/contexts/AuthContext.tsx`:

```tsx
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { authApi, type AuthUser } from '@/lib/auth'

interface AuthState {
  user: AuthUser | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
}

const Ctx = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    authApi.me().then(setUser).catch(() => setUser(null)).finally(() => setLoading(false))
    const onUnauth = () => setUser(null)
    window.addEventListener('auth:unauthorized', onUnauth)
    return () => window.removeEventListener('auth:unauthorized', onUnauth)
  }, [])

  const login = async (email: string, password: string) => setUser(await authApi.login(email, password))
  const logout = async () => { await authApi.logout(); setUser(null) }

  return <Ctx.Provider value={{ user, loading, login, logout }}>{children}</Ctx.Provider>
}

export function useAuth() {
  const v = useContext(Ctx)
  if (!v) throw new Error('useAuth must be used within AuthProvider')
  return v
}
```

- [ ] **Step 3: Wrap the app**

In `frontend/src/App.tsx`, wrap the existing tree (RouterProvider) with `<AuthProvider>`.

- [ ] **Step 4: Build**

Run: `cd frontend && npm run build`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/auth.ts frontend/src/contexts/AuthContext.tsx frontend/src/App.tsx
git commit -m "feat(auth): frontend auth context"
```

---

### Task 19: Login page

**Files:**
- Create: `frontend/src/pages/LoginPage.tsx`
- Modify: `frontend/src/routes.tsx` (add `/login` route, outside Layout if desired)

- [ ] **Step 1: Login page component**

`frontend/src/pages/LoginPage.tsx`:

```tsx
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'
import { useTranslation } from 'react-i18next'

export default function LoginPage() {
  const { login } = useAuth()
  const nav = useNavigate()
  const { t } = useTranslation()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    try { await login(email, password); nav('/ai/data') }
    catch { setError(t('auth.invalidCredentials')) }
  }

  return (
    <div className="flex items-center justify-center h-full">
      <form onSubmit={submit} className="flex flex-col gap-3 w-80 p-6 rounded-lg bg-surface">
        <h1 className="text-xl font-semibold text-text-primary">{t('auth.signIn')}</h1>
        <input className="px-3 py-2 rounded bg-background text-text-primary" type="email"
          placeholder={t('auth.email')} value={email} onChange={e => setEmail(e.target.value)} required />
        <input className="px-3 py-2 rounded bg-background text-text-primary" type="password"
          placeholder={t('auth.password')} value={password} onChange={e => setPassword(e.target.value)} required />
        {error && <p className="text-sm text-red-400">{error}</p>}
        <button type="submit" className="px-3 py-2 rounded bg-blue-600 text-white">{t('auth.signIn')}</button>
      </form>
    </div>
  )
}
```

Add i18n keys `auth.signIn`, `auth.email`, `auth.password`, `auth.invalidCredentials` to the FR locale files (follow the existing i18n structure under `frontend/src/i18n/`).

- [ ] **Step 2: Add the route**

In `frontend/src/routes.tsx`, add a lazy import and a `{ path: '/login', element: <SW><LoginPage /></SW> }` entry inside the Layout children (or as a sibling route without Layout if the design prefers a bare login screen).

- [ ] **Step 3: Build**

Run: `cd frontend && npm run build`
Expected: success.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/LoginPage.tsx frontend/src/routes.tsx frontend/src/i18n/
git commit -m "feat(auth): login page"
```

---

### Task 20: Route guards + account menu

**Files:**
- Create: `frontend/src/components/auth/RequireAuth.tsx`
- Modify: `frontend/src/routes.tsx` (wrap `/ai` and `/pastas` subtrees)
- Modify: the Layout/header component (add account menu with logout) — locate via `frontend/src/components/layout/Layout.tsx`

- [ ] **Step 1: Guard component**

`frontend/src/components/auth/RequireAuth.tsx`:

```tsx
import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'

export function RequireAuth({ children, adminOnly = false }: { children: React.ReactNode; adminOnly?: boolean }) {
  const { user, loading } = useAuth()
  const loc = useLocation()
  if (loading) return <div className="flex items-center justify-center h-full text-text-secondary">…</div>
  if (!user) return <Navigate to="/login" replace state={{ from: loc.pathname }} />
  if (adminOnly && user.role !== 'admin') return <Navigate to="/" replace />
  return <>{children}</>
}
```

- [ ] **Step 2: Wrap atelier subtrees**

In `frontend/src/routes.tsx`, wrap the `/ai` and `/pastas` route `element`s with `<RequireAuth>…</RequireAuth>` (observatory routes stay unwrapped).

- [ ] **Step 3: Account menu**

In the header/layout, when `user` is set show display name + a logout button calling `logout()` then `navigate('/')`; when not set, show a "Se connecter" link to `/login`. Show an "Admin" link to `/admin/users` when `user.role === 'admin'`.

- [ ] **Step 4: Build + manual check**

Run: `cd frontend && npm run build`
Expected: success. Manual: visiting `/ai/data` while logged out redirects to `/login`; observatory `/` loads without login.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/auth/RequireAuth.tsx frontend/src/routes.tsx frontend/src/components/layout/
git commit -m "feat(auth): route guards + account menu"
```

---

### Task 21: Admin users page

**Files:**
- Create: `frontend/src/pages/admin/UsersPage.tsx`
- Modify: `frontend/src/routes.tsx` (add `/admin/users`, adminOnly)
- Modify: `frontend/src/lib/auth.ts` (admin API calls)

- [ ] **Step 1: Admin API**

Add to `frontend/src/lib/auth.ts`:

```typescript
export const adminApi = {
  list: () => req<AuthUser[]>('/admin/users'),
  create: (body: { email: string; display_name: string; role: string; initial_password: string }) =>
    req<AuthUser>('/admin/users', { method: 'POST', body: JSON.stringify(body) }),
  update: (id: string, body: { is_active?: boolean; role?: string; new_password?: string }) =>
    req<AuthUser>(`/admin/users/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
}
```

- [ ] **Step 2: Users page**

`frontend/src/pages/admin/UsersPage.tsx`: a table listing users (email, name, role, active) with a "create user" form (email, name, role, initial password) and per-row enable/disable + reset-password actions, all calling `adminApi`. Reuse existing table/button styles from the codebase.

- [ ] **Step 3: Route (adminOnly)**

In `routes.tsx`: `{ path: '/admin/users', element: <RequireAuth adminOnly><SW><UsersPage /></SW></RequireAuth> }`.

- [ ] **Step 4: Build**

Run: `cd frontend && npm run build`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/admin/UsersPage.tsx frontend/src/routes.tsx frontend/src/lib/auth.ts
git commit -m "feat(auth): admin users management page"
```

---

## Final verification

- [ ] **Backend suite green:** `pytest tests/auth/ -v` → all PASS
- [ ] **Frontend builds:** `cd frontend && npm run build` → success
- [ ] **Manual end-to-end** (stacks rebuilt):
  - Observatory `/` loads without login.
  - `/ai/data` redirects to `/login` when logged out.
  - After login, `/api/v1/datasets` and `/api/v1/models` show only the user's resources.
  - A second user cannot fetch the first user's model id (`404`).
  - Admin sees all + can create/disable users at `/admin/users`.
- [ ] **Rebuild deployed stacks:** backend (`deploy/dib-backend`) and frontend (`deploy/frontend`) per `deploy/README.md`, run `alembic upgrade head`, then `create_admin` + `assign_legacy_ownership`.

---

## Notes for the implementer

- **404 vs 403:** foreign-resource access returns **404** (don't disclose existence); only the `/admin/*` role gate returns **403**.
- **Secrets:** set a real `JWT_SECRET` env var in `deploy/dib-backend/.env` (and `.env.example`) before deploying; the default is insecure.
- **Cookie over HTTP in local dev:** `cookie_secure=True` blocks cookies on plain HTTP. For local non-TLS testing set `COOKIE_SECURE=false`; keep `true` in production (TLS at the K8s ingress).
- **Don't touch observatory routers** — they must remain public.
