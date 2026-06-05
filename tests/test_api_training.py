import pytest


@pytest.mark.asyncio
async def test_training_history(auth_client):
    """Training history returns list."""
    resp = await auth_client.get("/api/v1/training/history")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_cancel_nonexistent_task(auth_client):
    """Cancelling nonexistent task returns 404."""
    resp = await auth_client.post("/api/v1/training/nonexistent/cancel")
    assert resp.status_code == 404
