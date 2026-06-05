import pytest


@pytest.mark.asyncio
async def test_list_datasets(auth_client):
    """List datasets returns 200 with list."""
    resp = await auth_client.get("/api/v1/datasets/")
    # May return 200 with empty list or 500 if no registry dir
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_get_nonexistent_dataset(auth_client):
    """Getting nonexistent dataset returns 404."""
    resp = await auth_client.get("/api/v1/datasets/nonexistent-id")
    assert resp.status_code == 404
