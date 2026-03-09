import pytest


@pytest.mark.asyncio
async def test_training_history(client):
    """Training history returns list."""
    resp = await client.get("/api/v1/training/history")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_cancel_nonexistent_task(client):
    """Cancelling nonexistent task returns 404."""
    resp = await client.post("/api/v1/training/nonexistent/cancel")
    assert resp.status_code == 404
