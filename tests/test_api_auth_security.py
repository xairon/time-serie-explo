import pytest
from sqlalchemy import select

from api.models_db.auth_event import AuthEvent


@pytest.mark.asyncio
async def test_login_success_records_audit(client, make_user, db_session):
    await make_user(email="ok@test.io", password="correct-horse-1")
    r = await client.post(
        "/api/v1/auth/login", json={"email": "ok@test.io", "password": "correct-horse-1"}
    )
    assert r.status_code == 200, r.text
    events = (
        await db_session.execute(
            select(AuthEvent).where(AuthEvent.event_type == "login_success")
        )
    ).scalars().all()
    assert len(events) == 1
    assert events[0].email == "ok@test.io"


@pytest.mark.asyncio
async def test_login_failure_records_audit(client, make_user, db_session):
    await make_user(email="bad@test.io", password="correct-horse-1")
    r = await client.post(
        "/api/v1/auth/login", json={"email": "bad@test.io", "password": "wrong"}
    )
    assert r.status_code == 401
    events = (
        await db_session.execute(
            select(AuthEvent).where(AuthEvent.event_type == "login_failure")
        )
    ).scalars().all()
    assert len(events) == 1
    assert events[0].email == "bad@test.io"


@pytest.mark.asyncio
async def test_change_password_clears_must_change(client, make_user):
    await make_user(email="chg@test.io", password="old-password-1")
    await client.post(
        "/api/v1/auth/login", json={"email": "chg@test.io", "password": "old-password-1"}
    )
    r = await client.post(
        "/api/v1/auth/change-password",
        json={"old_password": "old-password-1", "new_password": "new-password-2"},
    )
    assert r.status_code == 204
    me = await client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["must_change_password"] is False
