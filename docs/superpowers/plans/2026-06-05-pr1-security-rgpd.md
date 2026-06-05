# PR1 — Security + RGPD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the FastAPI auth surface (rate limiting, lockout, audit log, header hardening, admin password reset) and add RGPD-minimal features (account erasure, retention purge, privacy notice) so the app is safe to expose publicly.

**Architecture:** Additive changes to the existing FastAPI app. Rate limiting and lockout use the existing async Redis client (`api/cache.py::get_redis`) — no new dependency (deviation from spec which named `slowapi`; rationale: avoid a second Redis pool and slowapi's sync/cookie-auth friction). New `auth_events` table + `users.must_change_password` column via Alembic. Erasure deletes the user row and owned artifacts (datasets dirs, in-memory tasks, MLflow runs tagged `owner_id`).

**Tech Stack:** FastAPI, SQLAlchemy async, Alembic, Redis (redis.asyncio), pytest (async via existing conftest), React/TS frontend.

---

## File Structure

- Create `api/auth/ratelimit.py` — async Redis IP rate limit + per-email lockout helpers.
- Create `api/auth/audit.py` — `record_event()` helper.
- Create `api/models_db/auth_event.py` — `AuthEvent` model.
- Create `api/auth/erasure.py` — `erase_user_artifacts(user_id)`.
- Modify `api/models_db/__init__.py` — export `AuthEvent`.
- Modify `api/models_db/user.py` — add `must_change_password`.
- Modify `api/auth/schemas.py` — `UserOut.must_change_password`; reset/delete response schemas.
- Modify `api/routers/auth.py` — login (limit/lockout/audit), logout audit, change-password audit + clear flag, `DELETE /me`.
- Modify `api/routers/admin.py` — reset-password, delete user, list auth-events.
- Modify `api/config.py` — retention + lockout settings.
- Create Alembic migration — `must_change_password` + `auth_events`.
- Create `scripts/purge_expired.py` — retention purge CLI.
- Modify `nginx/nginx.conf` + `deploy/frontend/nginx.conf.template` — HSTS + Permissions-Policy.
- Frontend: `frontend/src/pages/PrivacyPage.tsx`, route in `routes.tsx`, link on `LoginPage.tsx`, must-change-password handling.
- Create `PRIVACY.md`.
- Tests under `tests/`.

> **Note on async tests:** `tests/conftest.py` already wires the API test client + DB. Follow the patterns in `tests/test_api_health.py` / `tests/test_api_datasets.py` (read them before writing tests). Use the existing authenticated-client fixture if present; otherwise create users via the DB session fixture and log in through `/api/v1/auth/login`.

---

## Task 1: Settings for lockout + retention

**Files:**
- Modify: `api/config.py`

- [ ] **Step 1: Add settings fields** after `session_ttl_hours` block in `Settings`:

```python
    # Brute-force protection (Redis-backed)
    login_max_failures: int = 5
    login_lockout_minutes: int = 15
    login_fail_window_minutes: int = 15
    login_rate_limit_per_minute: int = 10
    # RGPD retention
    auth_event_retention_days: int = 365
```

- [ ] **Step 2: Verify import** — `python -c "import ast; ast.parse(open('api/config.py').read())"` → no error.

- [ ] **Step 3: Commit**

```bash
git add api/config.py
git commit -m "feat(config): lockout + retention settings"
```

---

## Task 2: AuthEvent model + migration column

**Files:**
- Create: `api/models_db/auth_event.py`
- Modify: `api/models_db/__init__.py`
- Modify: `api/models_db/user.py`

- [ ] **Step 1: Create the model** `api/models_db/auth_event.py`:

```python
import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from api.models_db.base import Base


class AuthEvent(Base):
    __tablename__ = "auth_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # No FK: events must survive user deletion (RGPD traceability).
    user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True, index=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(400), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
```

- [ ] **Step 2: Add `must_change_password` to `User`** in `api/models_db/user.py`, after `token_version`:

```python
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
```

(`Boolean` is already imported.)

- [ ] **Step 3: Export AuthEvent** — in `api/models_db/__init__.py` add `AuthEvent` to imports and `__all__` (match existing style for `User`/`UserRole`).

- [ ] **Step 4: Verify** — `python -c "import ast; [ast.parse(open(f).read()) for f in ['api/models_db/auth_event.py','api/models_db/user.py','api/models_db/__init__.py']]"`.

- [ ] **Step 5: Create Alembic migration.** Inspect `alembic/versions/` for the latest revision id and `down_revision` chaining, then create `alembic/versions/<new>_auth_events_and_must_change_password.py`:

```python
"""auth_events table and users.must_change_password"""
from alembic import op
import sqlalchemy as sa

revision = "<generate: 12 hex>"
down_revision = "<latest existing revision id>"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "users",
        sa.Column("must_change_password", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_table(
        "auth_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("event_type", sa.String(length=40), nullable=False),
        sa.Column("ip", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=400), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_auth_events_user_id", "auth_events", ["user_id"])
    op.create_index("ix_auth_events_event_type", "auth_events", ["event_type"])
    op.create_index("ix_auth_events_created_at", "auth_events", ["created_at"])


def downgrade():
    op.drop_table("auth_events")
    op.drop_column("users", "must_change_password")
```

- [ ] **Step 6: Commit**

```bash
git add api/models_db/ alembic/versions/
git commit -m "feat(db): auth_events table + users.must_change_password"
```

---

## Task 3: Audit helper

**Files:**
- Create: `api/auth/audit.py`
- Test: `tests/test_auth_audit.py`

- [ ] **Step 1: Write the failing test** `tests/test_auth_audit.py`:

```python
import pytest
from sqlalchemy import select

from api.auth.audit import record_event
from api.models_db.auth_event import AuthEvent


@pytest.mark.asyncio
async def test_record_event_persists(db_session):  # db_session: existing async fixture
    await record_event(db_session, event_type="login_success", email="a@b.c", user_id=None,
                       ip="1.2.3.4", user_agent="pytest")
    await db_session.commit()
    rows = (await db_session.execute(select(AuthEvent))).scalars().all()
    assert len(rows) == 1
    assert rows[0].event_type == "login_success"
    assert rows[0].ip == "1.2.3.4"
```

> If the conftest async DB fixture has a different name, adapt `db_session` to it (check `tests/conftest.py`).

- [ ] **Step 2: Run → fails** `pytest tests/test_auth_audit.py -v` → ImportError / no `record_event`.

- [ ] **Step 3: Implement** `api/auth/audit.py`:

```python
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from api.models_db.auth_event import AuthEvent


async def record_event(
    db: AsyncSession,
    *,
    event_type: str,
    email: str | None = None,
    user_id: uuid.UUID | None = None,
    ip: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Append an auth audit event. Caller is responsible for commit."""
    db.add(AuthEvent(
        user_id=user_id, email=email, event_type=event_type,
        ip=ip, user_agent=(user_agent or "")[:400] or None,
    ))
```

- [ ] **Step 4: Run → passes** `pytest tests/test_auth_audit.py -v`.

- [ ] **Step 5: Commit** `git add api/auth/audit.py tests/test_auth_audit.py && git commit -m "feat(auth): audit event helper"`

---

## Task 4: Rate limit + lockout helpers (Redis)

**Files:**
- Create: `api/auth/ratelimit.py`
- Test: `tests/test_ratelimit.py`

- [ ] **Step 1: Write failing test** `tests/test_ratelimit.py` (uses a fake redis to stay unit-level):

```python
import pytest
from api.auth import ratelimit


class FakeRedis:
    def __init__(self): self.store = {}
    async def incr(self, k): self.store[k] = self.store.get(k, 0) + 1; return self.store[k]
    async def expire(self, k, ttl): return True
    async def get(self, k):
        v = self.store.get(k); return None if v is None else str(v).encode()
    async def setex(self, k, ttl, v): self.store[k] = v; return True
    async def delete(self, *ks):
        for k in ks: self.store.pop(k, None)
        return True


@pytest.mark.asyncio
async def test_lockout_after_threshold(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(ratelimit, "get_redis", lambda: fake)
    email = "victim@example.com"
    assert await ratelimit.is_locked(email) is False
    for _ in range(5):
        await ratelimit.register_failure(email)
    assert await ratelimit.is_locked(email) is True


@pytest.mark.asyncio
async def test_success_clears_failures(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(ratelimit, "get_redis", lambda: fake)
    email = "ok@example.com"
    await ratelimit.register_failure(email)
    await ratelimit.clear_failures(email)
    for _ in range(4):
        await ratelimit.register_failure(email)
    assert await ratelimit.is_locked(email) is False


@pytest.mark.asyncio
async def test_no_redis_is_noop(monkeypatch):
    monkeypatch.setattr(ratelimit, "get_redis", lambda: None)
    assert await ratelimit.is_locked("x@y.z") is False
    await ratelimit.register_failure("x@y.z")  # must not raise
```

- [ ] **Step 2: Run → fails** `pytest tests/test_ratelimit.py -v`.

- [ ] **Step 3: Implement** `api/auth/ratelimit.py`:

```python
import logging

from fastapi import HTTPException, Request

from api.cache import get_redis
from api.config import settings

logger = logging.getLogger(__name__)


def _norm(email: str) -> str:
    return (email or "").strip().lower()


def _fail_key(email: str) -> str:
    return f"junon:loginfail:{_norm(email)}"


def _lock_key(email: str) -> str:
    return f"junon:loginlock:{_norm(email)}"


def _rate_key(ip: str) -> str:
    return f"junon:loginrate:{ip}"


async def is_locked(email: str) -> bool:
    r = get_redis()
    if r is None:
        return False
    try:
        return (await r.get(_lock_key(email))) is not None
    except Exception as e:
        logger.debug("lockout check failed: %s", e)
        return False


async def register_failure(email: str) -> None:
    r = get_redis()
    if r is None:
        return
    try:
        n = await r.incr(_fail_key(email))
        if n == 1:
            await r.expire(_fail_key(email), settings.login_fail_window_minutes * 60)
        if n >= settings.login_max_failures:
            await r.setex(_lock_key(email), settings.login_lockout_minutes * 60, b"1")
    except Exception as e:
        logger.debug("register_failure failed: %s", e)


async def clear_failures(email: str) -> None:
    r = get_redis()
    if r is None:
        return
    try:
        await r.delete(_fail_key(email), _lock_key(email))
    except Exception as e:
        logger.debug("clear_failures failed: %s", e)


async def enforce_ip_rate_limit(request: Request) -> None:
    """Sliding 60s counter per client IP. Raises 429 over the limit."""
    r = get_redis()
    if r is None:
        return
    ip = (request.client.host if request.client else "unknown")
    key = _rate_key(ip)
    try:
        n = await r.incr(key)
        if n == 1:
            await r.expire(key, 60)
        if n > settings.login_rate_limit_per_minute:
            raise HTTPException(status_code=429, detail="Trop de tentatives, réessayez plus tard")
    except HTTPException:
        raise
    except Exception as e:
        logger.debug("rate limit check failed: %s", e)
```

- [ ] **Step 4: Run → passes** `pytest tests/test_ratelimit.py -v`.

- [ ] **Step 5: Commit** `git add api/auth/ratelimit.py tests/test_ratelimit.py && git commit -m "feat(auth): redis rate limit + lockout helpers"`

---

## Task 5: Wire login/logout/change-password (lockout + audit + flag)

**Files:**
- Modify: `api/routers/auth.py`
- Modify: `api/auth/schemas.py` (UserOut)
- Test: `tests/test_api_auth_security.py`

- [ ] **Step 1: Add `must_change_password` to `UserOut`** in `api/auth/schemas.py` (mirror existing fields; ensure `from_attributes`/ORM mode already set as for other fields).

- [ ] **Step 2: Write failing test** `tests/test_api_auth_security.py` (adapt client fixture name to conftest):

```python
import pytest


@pytest.mark.asyncio
async def test_bad_login_then_lockout(client, make_user):
    # make_user: creates a user with known password; adapt to conftest helpers
    await make_user(email="lock@test.io", password="correct-horse-1")
    for _ in range(5):
        r = await client.post("/api/v1/auth/login", json={"email": "lock@test.io", "password": "wrong"})
        assert r.status_code in (401, 429)
    # now locked: even correct creds are refused with 429
    r = await client.post("/api/v1/auth/login", json={"email": "lock@test.io", "password": "correct-horse-1"})
    assert r.status_code == 429


@pytest.mark.asyncio
async def test_login_success_sets_cookie_and_audit(client, make_user, db_session):
    from sqlalchemy import select
    from api.models_db.auth_event import AuthEvent
    await make_user(email="ok@test.io", password="correct-horse-1")
    r = await client.post("/api/v1/auth/login", json={"email": "ok@test.io", "password": "correct-horse-1"})
    assert r.status_code == 200
    events = (await db_session.execute(select(AuthEvent).where(AuthEvent.event_type == "login_success"))).scalars().all()
    assert len(events) >= 1
```

> If Redis is not available in CI, lockout helpers no-op and `test_bad_login_then_lockout` would not reach 429. Guard it: `pytestmark = pytest.mark.skipif(get_redis() is None, ...)` or run a fakeredis fixture. Prefer skip if no redis in CI — check whether `tests/conftest.py` provides redis.

- [ ] **Step 3: Run → fails.**

- [ ] **Step 4: Rewrite `login` and friends** in `api/routers/auth.py`. New imports at top:

```python
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from api.auth.audit import record_event
from api.auth.ratelimit import (
    enforce_ip_rate_limit, is_locked, register_failure, clear_failures,
)
```

Replace `login`:

```python
@router.post("/login", response_model=UserOut)
async def login(req: LoginRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    await enforce_ip_rate_limit(request)
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent")
    if await is_locked(req.email):
        raise HTTPException(status_code=429, detail="Compte temporairement verrouillé, réessayez plus tard")
    user = (await db.execute(select(User).where(User.email == req.email))).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(req.password, user.password_hash):
        await register_failure(req.email)
        await record_event(db, event_type="login_failure", email=req.email,
                           user_id=(user.id if user else None), ip=ip, user_agent=ua)
        await db.commit()
        raise HTTPException(status_code=401, detail="Identifiants invalides")
    await clear_failures(req.email)
    user.last_login_at = func.now()
    await record_event(db, event_type="login_success", email=user.email, user_id=user.id, ip=ip, user_agent=ua)
    await db.commit()
    await db.refresh(user)
    _set_session_cookie(response, user)
    return user
```

Update `logout` to take `request`/`user` optionally for audit (keep 204). Minimal: audit logout only when a valid session exists — to avoid extra DB round trips, leave logout as-is OR add best-effort audit. Keep simple:

```python
@router.post("/logout", status_code=204)
async def logout(response: Response):
    response.delete_cookie(settings.cookie_name, path="/")
```

Update `change_password` to clear the flag and audit:

```python
@router.post("/change-password", status_code=204)
async def change_password(req: ChangePasswordRequest, request: Request, response: Response,
                          user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    if not verify_password(req.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Ancien mot de passe incorrect")
    user.password_hash = hash_password(req.new_password)
    user.token_version += 1
    user.must_change_password = False
    await record_event(db, event_type="password_change", email=user.email, user_id=user.id,
                       ip=(request.client.host if request.client else None),
                       user_agent=request.headers.get("user-agent"))
    await db.commit()
    await db.refresh(user)
    _set_session_cookie(response, user)
```

- [ ] **Step 5: Run → passes** (skip lockout test if no redis).

- [ ] **Step 6: Commit** `git add api/routers/auth.py api/auth/schemas.py tests/test_api_auth_security.py && git commit -m "feat(auth): lockout + audit on login/change-password"`

---

## Task 6: Erasure helper + DELETE /auth/me

**Files:**
- Create: `api/auth/erasure.py`
- Modify: `api/routers/auth.py`
- Test: `tests/test_api_account_deletion.py`

- [ ] **Step 1: Inspect registries** — read `dashboard/utils/dataset_registry.py` (`get_owner`, dataset dir layout) and `dashboard/utils/model_registry.py` (`get_model_owner`, how runs/tags are stored) to confirm the deletion calls below. Adjust method names to the actual API.

- [ ] **Step 2: Implement** `api/auth/erasure.py`:

```python
import logging
import shutil
import uuid
from pathlib import Path

from api.config import settings
from api.task_manager import task_manager

logger = logging.getLogger(__name__)


def erase_user_artifacts(user_id: uuid.UUID) -> None:
    """Best-effort deletion of resources owned by a user (datasets, tasks, MLflow runs)."""
    uid = str(user_id)
    # 1. In-memory tasks
    try:
        for t in task_manager.list_tasks():
            if getattr(t, "owner_id", None) == uid:
                task_manager.cancel(t.task_id)
    except Exception as e:
        logger.warning("task erase failed for %s: %s", uid, e)

    # 2. Dataset directories owned by the user
    try:
        from dashboard.utils.dataset_registry import DatasetRegistry
        reg = DatasetRegistry(Path(settings.data_dir) / "prepared")
        for ds in reg.scan_datasets():
            try:
                if reg.get_owner(ds.path.name) == uid:
                    shutil.rmtree(ds.path, ignore_errors=True)
            except Exception:
                continue
    except Exception as e:
        logger.warning("dataset erase failed for %s: %s", uid, e)

    # 3. MLflow runs tagged owner_id == user
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()
        for exp in client.search_experiments():
            runs = client.search_runs([exp.experiment_id], filter_string=f"tags.owner_id = '{uid}'")
            for run in runs:
                try:
                    client.delete_run(run.info.run_id)
                except Exception:
                    continue
    except Exception as e:
        logger.warning("mlflow erase failed for %s: %s", uid, e)
```

- [ ] **Step 3: Write failing test** `tests/test_api_account_deletion.py`:

```python
import pytest
from sqlalchemy import select
from api.models_db import User


@pytest.mark.asyncio
async def test_delete_me_removes_account(auth_client, db_session, current_user):
    # auth_client: logged-in client; current_user: the User row. Adapt to conftest.
    r = await auth_client.delete("/api/v1/auth/me")
    assert r.status_code == 204
    gone = (await db_session.execute(select(User).where(User.id == current_user.id))).scalar_one_or_none()
    assert gone is None
```

- [ ] **Step 4: Run → fails.**

- [ ] **Step 5: Add endpoint** in `api/routers/auth.py`:

```python
@router.delete("/me", status_code=204)
async def delete_me(request: Request, response: Response,
                    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    from api.auth.erasure import erase_user_artifacts
    uid, email = user.id, user.email
    erase_user_artifacts(uid)
    await db.delete(user)
    await record_event(db, event_type="account_deleted", email=None, user_id=uid,
                       ip=(request.client.host if request.client else None),
                       user_agent=request.headers.get("user-agent"))
    await db.commit()
    response.delete_cookie(settings.cookie_name, path="/")
```

(Email intentionally omitted from the retained event — only the id is kept for traceability.)

- [ ] **Step 6: Run → passes.**

- [ ] **Step 7: Commit** `git add api/auth/erasure.py api/routers/auth.py tests/test_api_account_deletion.py && git commit -m "feat(rgpd): account self-erasure"`

---

## Task 7: Admin reset-password, delete user, auth-events

**Files:**
- Modify: `api/routers/admin.py`
- Test: `tests/test_api_admin_security.py`

- [ ] **Step 1: Write failing tests** `tests/test_api_admin_security.py`:

```python
import pytest
from sqlalchemy import select
from api.models_db import User, UserRole


@pytest.mark.asyncio
async def test_admin_reset_password(admin_client, make_user, db_session):
    u = await make_user(email="target@test.io", password="old-password-1")
    r = await admin_client.post(f"/api/v1/admin/users/{u.id}/reset-password")
    assert r.status_code == 200
    body = r.json()
    assert "temporary_password" in body and len(body["temporary_password"]) >= 12
    refreshed = (await db_session.execute(select(User).where(User.id == u.id))).scalar_one()
    assert refreshed.must_change_password is True


@pytest.mark.asyncio
async def test_admin_cannot_delete_last_admin(admin_client, current_admin):
    r = await admin_client.delete(f"/api/v1/admin/users/{current_admin.id}")
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_admin_delete_user(admin_client, make_user, db_session):
    u = await make_user(email="bye@test.io", password="x-password-1")
    r = await admin_client.delete(f"/api/v1/admin/users/{u.id}")
    assert r.status_code == 204
    assert (await db_session.execute(select(User).where(User.id == u.id))).scalar_one_or_none() is None
```

- [ ] **Step 2: Run → fails.**

- [ ] **Step 3: Implement** in `api/routers/admin.py`. Add imports:

```python
import secrets
from fastapi import Request, Response
from sqlalchemy import func
from api.auth.audit import record_event
from api.auth.erasure import erase_user_artifacts
from api.models_db.auth_event import AuthEvent
```

Add endpoints:

```python
@router.post("/{user_id}/reset-password")
async def reset_password(user_id: uuid.UUID, request: Request,
                         admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    temp = secrets.token_urlsafe(12)
    user.password_hash = hash_password(temp)
    user.token_version += 1
    user.must_change_password = True
    await record_event(db, event_type="password_reset", email=user.email, user_id=user.id,
                       ip=(request.client.host if request.client else None),
                       user_agent=request.headers.get("user-agent"))
    await db.commit()
    return {"temporary_password": temp}


@router.delete("/{user_id}", status_code=204)
async def delete_user(user_id: uuid.UUID, request: Request, response: Response,
                      admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    if user.role == UserRole.admin:
        admin_count = (await db.execute(
            select(func.count()).select_from(User).where(User.role == UserRole.admin)
        )).scalar_one()
        if admin_count <= 1:
            raise HTTPException(status_code=409, detail="Impossible de supprimer le dernier administrateur")
    erase_user_artifacts(user.id)
    uid = user.id
    await db.delete(user)
    await record_event(db, event_type="admin_user_deleted", email=None, user_id=uid,
                       ip=(request.client.host if request.client else None),
                       user_agent=request.headers.get("user-agent"))
    await db.commit()
```

Add a separate router file or extend admin for auth-events. Simplest: new endpoint on a sibling router under the same admin prefix. Add to `api/routers/admin.py` a second router OR extend. Create `GET /api/v1/admin/auth-events`:

```python
@router.get("/auth-events/list")  # under /api/v1/admin/users — see note
async def list_auth_events(limit: int = 100, offset: int = 0,
                           admin: User = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    limit = max(1, min(limit, 500))
    rows = (await db.execute(
        select(AuthEvent).order_by(AuthEvent.created_at.desc()).limit(limit).offset(offset)
    )).scalars().all()
    return [
        {"id": str(e.id), "user_id": str(e.user_id) if e.user_id else None,
         "email": e.email, "event_type": e.event_type, "ip": e.ip,
         "created_at": e.created_at.isoformat()} for e in rows
    ]
```

> **Routing note:** the admin router prefix is `/api/v1/admin/users`. Auth-events do not belong under `/users`. Cleaner: create a NEW router `api/routers/admin_audit.py` with prefix `/api/v1/admin` exposing `GET /auth-events`, and register it in `api/main.py`. Do that instead of the `/users/auth-events/list` hack above.

- [ ] **Step 4: Create `api/routers/admin_audit.py`** with the `list_auth_events` endpoint (prefix `/api/v1/admin`, path `/auth-events`) and register it in `api/main.py` (`app.include_router(admin_audit.router)` — admin router carries its own `require_admin`).

- [ ] **Step 5: Run → passes.**

- [ ] **Step 6: Commit** `git add api/routers/admin.py api/routers/admin_audit.py api/main.py tests/test_api_admin_security.py && git commit -m "feat(admin): password reset, user deletion, auth-events"`

---

## Task 8: nginx header hardening

**Files:**
- Modify: `nginx/nginx.conf`
- Modify: `deploy/frontend/nginx.conf.template`

- [ ] **Step 1: Add headers** right after the existing `Referrer-Policy` line in BOTH files:

```nginx
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
```

- [ ] **Step 2: Validate syntax** (if nginx available): `nginx -t -c $(pwd)/nginx/nginx.conf` — otherwise visual check. Expected: no error or "syntax is ok".

- [ ] **Step 3: Commit** `git add nginx/nginx.conf deploy/frontend/nginx.conf.template && git commit -m "feat(nginx): HSTS + Permissions-Policy"`

---

## Task 9: Retention purge CLI

**Files:**
- Create: `scripts/purge_expired.py`
- Test: `tests/test_purge_expired.py`

- [ ] **Step 1: Write failing test** `tests/test_purge_expired.py`:

```python
import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from api.models_db.auth_event import AuthEvent
from scripts.purge_expired import purge_auth_events


@pytest.mark.asyncio
async def test_purge_removes_old_events(db_session):
    old = AuthEvent(event_type="login_success", email="a@b.c")
    db_session.add(old)
    await db_session.flush()
    old.created_at = datetime.now(timezone.utc) - timedelta(days=400)
    await db_session.commit()
    deleted = await purge_auth_events(db_session, retention_days=365)
    await db_session.commit()
    assert deleted == 1
    assert (await db_session.execute(select(AuthEvent))).scalars().all() == []
```

- [ ] **Step 2: Run → fails.**

- [ ] **Step 3: Implement** `scripts/purge_expired.py`:

```python
"""RGPD retention purge. Run via cron, e.g. daily:
    python -m scripts.purge_expired
"""
import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete

from api.config import settings
from api.database import async_session_maker  # confirm exact name in api/database.py
from api.models_db.auth_event import AuthEvent


async def purge_auth_events(db, retention_days: int) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    result = await db.execute(delete(AuthEvent).where(AuthEvent.created_at < cutoff))
    return result.rowcount or 0


async def main() -> None:
    async with async_session_maker() as db:
        n = await purge_auth_events(db, settings.auth_event_retention_days)
        await db.commit()
        print(f"Purged {n} auth_events older than {settings.auth_event_retention_days} days")


if __name__ == "__main__":
    asyncio.run(main())
```

> Read `api/database.py` to confirm the sessionmaker export name; adapt the import.

- [ ] **Step 4: Run → passes.**

- [ ] **Step 5: Commit** `git add scripts/purge_expired.py tests/test_purge_expired.py && git commit -m "feat(rgpd): auth_events retention purge"`

---

## Task 10: Frontend — privacy page + must-change-password

**Files:**
- Create: `frontend/src/pages/PrivacyPage.tsx`
- Modify: `frontend/src/routes.tsx`
- Modify: `frontend/src/pages/LoginPage.tsx`
- Modify: the auth/session hook or `App` guard that reads `/auth/me`
- Create: `PRIVACY.md`

- [ ] **Step 1: Create `PrivacyPage.tsx`** — a public page rendering the privacy notice (data processed: email, display name, uploaded hydro data; purpose; retention 365 days for auth logs; rights: access/rectification/erasure incl. "Supprimer mon compte"; contact DPO). French. Mark "Contenu à valider par le DPO BRGM".

- [ ] **Step 2: Add a public route** `/privacy` in `routes.tsx` (outside the auth guard, like `/login`).

- [ ] **Step 3: Link it** from `LoginPage.tsx` footer ("Politique de confidentialité").

- [ ] **Step 4: must-change-password gate** — wherever the app loads the current user (`/auth/me`), if `must_change_password === true`, redirect to the change-password screen and block other navigation until changed. Reuse the existing change-password UI/flow.

- [ ] **Step 5: Type-check + build** — `cd frontend && npx tsc --noEmit && npm run build`. Expected: success.

- [ ] **Step 6: Create `PRIVACY.md`** mirroring the page text.

- [ ] **Step 7: Commit** `git add frontend/src PRIVACY.md && git commit -m "feat(rgpd): privacy notice page + forced password change"`

---

## Task 11: Full test + lint pass, final commit

- [ ] **Step 1: Run full suite** `PYTHONPATH=$(pwd) uv run pytest tests/ -q`. Expected: all green (lockout test skipped if no redis).
- [ ] **Step 2: Frontend** `cd frontend && npx tsc --noEmit && npm run build`. Expected: green.
- [ ] **Step 3: Sanity import** `uv run python -c "import api.main"`. Expected: no error (validator passes because tests set DEBUG/secret, or run with DEBUG=true locally).
- [ ] **Step 4: Final commit if anything pending.**

---

## Self-Review

- **Spec coverage:** rate limit + lockout (T4–T5), audit log (T2–T3, T5, T7), headers (T8), password reset admin (T7), must_change flow (T5, T10), erasure self + admin (T6, T7), retention purge (T9), privacy notice (T10), config (T1). All spec PR1 items mapped.
- **Placeholders:** registry method names in T6/T9 flagged for confirmation against real code (`get_owner`, `async_session_maker`) — these are verification steps, not unresolved placeholders; the code given is complete pending name confirmation.
- **Type consistency:** `record_event(db, *, event_type, email, user_id, ip, user_agent)` used identically in T3/T5/T6/T7. `must_change_password` name consistent across model/schema/router/frontend. `erase_user_artifacts(user_id)` consistent T6/T7.
