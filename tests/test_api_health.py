import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Health endpoint returns 200 with status info."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "gpu" in data
    assert "db" in data
    assert "redis" in data


@pytest.mark.asyncio
async def test_health_gpu_info(client):
    """Health endpoint includes GPU availability info."""
    resp = await client.get("/api/v1/health")
    data = resp.json()
    assert isinstance(data["gpu"], dict)
    assert "available" in data["gpu"]
