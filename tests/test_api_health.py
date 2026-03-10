import pytest


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Health endpoint returns 200 or 503 with relevant info."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code in (200, 503)
    data = resp.json()
    if resp.status_code == 200:
        assert "status" in data
        assert "gpu" in data
    else:
        # 503 when DB/Redis unreachable - returns error detail
        assert "detail" in data


@pytest.mark.asyncio
async def test_health_gpu_info(client):
    """Health endpoint includes GPU availability info when healthy."""
    resp = await client.get("/api/v1/health")
    if resp.status_code == 200:
        data = resp.json()
        assert isinstance(data["gpu"], dict)
        assert "available" in data["gpu"]
