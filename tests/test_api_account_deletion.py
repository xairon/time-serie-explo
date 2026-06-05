import pytest
from sqlalchemy import select

from api.models_db import User
from api.models_db.auth_event import AuthEvent


@pytest.mark.asyncio
async def test_delete_me_removes_account(client, make_user, db_session):
    user = await make_user(email="bye@test.io", password="correct-horse-1")
    await client.post(
        "/api/v1/auth/login", json={"email": "bye@test.io", "password": "correct-horse-1"}
    )
    r = await client.delete("/api/v1/auth/me")
    assert r.status_code == 204

    gone = (
        await db_session.execute(select(User).where(User.id == user.id))
    ).scalar_one_or_none()
    assert gone is None

    # audit event retained, without email
    ev = (
        await db_session.execute(
            select(AuthEvent).where(AuthEvent.event_type == "account_deleted")
        )
    ).scalars().one()
    assert str(ev.user_id) == str(user.id)
    assert ev.email is None


@pytest.mark.asyncio
async def test_delete_me_requires_auth(client):
    r = await client.delete("/api/v1/auth/me")
    assert r.status_code == 401
