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
