import pytest
from sqlalchemy import select

from api.models_db import User, UserRole


async def _login_admin(client, make_user, email="admin@test.io", password="admin-password-1"):
    await make_user(email=email, password=password, role=UserRole.admin)
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_admin_reset_password_flow(client, make_user, db_session):
    target = await make_user(email="target@test.io", password="old-password-1")
    await _login_admin(client, make_user)

    r = await client.post(f"/api/v1/admin/users/{target.id}/reset-password")
    assert r.status_code == 200, r.text
    temp = r.json()["temporary_password"]
    assert len(temp) >= 12

    refreshed = (
        await db_session.execute(select(User).where(User.id == target.id))
    ).scalar_one()
    assert refreshed.must_change_password is True


@pytest.mark.asyncio
async def test_admin_cannot_delete_last_admin(client, make_user, db_session):
    await _login_admin(client, make_user, email="solo@test.io")
    admin = (
        await db_session.execute(select(User).where(User.email == "solo@test.io"))
    ).scalar_one()
    r = await client.delete(f"/api/v1/admin/users/{admin.id}")
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_admin_delete_user(client, make_user, db_session):
    target = await make_user(email="victim@test.io", password="x-password-1")
    await _login_admin(client, make_user)
    r = await client.delete(f"/api/v1/admin/users/{target.id}")
    assert r.status_code == 204
    gone = (
        await db_session.execute(select(User).where(User.id == target.id))
    ).scalar_one_or_none()
    assert gone is None


@pytest.mark.asyncio
async def test_auth_events_endpoint(client, make_user):
    await _login_admin(client, make_user)
    r = await client.get("/api/v1/admin/auth-events")
    assert r.status_code == 200
    data = r.json()
    # admin login itself produced a login_success event
    assert any(e["event_type"] == "login_success" for e in data)
