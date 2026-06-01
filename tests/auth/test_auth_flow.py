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
