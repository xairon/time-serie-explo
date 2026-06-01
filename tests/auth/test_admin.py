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
