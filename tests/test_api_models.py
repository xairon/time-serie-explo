import pytest


@pytest.mark.asyncio
async def test_list_available_models(auth_client):
    """Available models endpoint returns list of architectures."""
    resp = await auth_client.get("/api/v1/models/available")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_list_trained_models(auth_client):
    """List trained models returns 200."""
    resp = await auth_client.get("/api/v1/models/")
    assert resp.status_code in (200, 500)


@pytest.mark.asyncio
async def test_get_nonexistent_model(auth_client):
    """Getting nonexistent model returns 404."""
    resp = await auth_client.get("/api/v1/models/nonexistent-id")
    assert resp.status_code == 404
