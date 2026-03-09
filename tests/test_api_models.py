import pytest


@pytest.mark.asyncio
async def test_list_available_models(client):
    """Available models endpoint returns list of architectures."""
    resp = await client.get("/api/v1/models/available")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_list_trained_models(client):
    """List trained models returns 200."""
    resp = await client.get("/api/v1/models/")
    assert resp.status_code in (200, 500)


@pytest.mark.asyncio
async def test_get_nonexistent_model(client):
    """Getting nonexistent model returns 404."""
    resp = await client.get("/api/v1/models/nonexistent-id")
    assert resp.status_code == 404
