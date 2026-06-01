import pytest


@pytest.mark.asyncio
async def test_atelier_requires_auth(client):
    res = await client.get("/api/v1/datasets")
    assert res.status_code == 401


@pytest.mark.asyncio
async def test_observatory_is_public(client):
    res = await client.get("/api/v1/observatory/hydro/alerts")
    assert res.status_code != 401
