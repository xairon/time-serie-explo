"""Tests for the forecasting API endpoints.

Tests endpoint status codes and request validation without requiring
actual trained models or database connections.
"""

import pytest


@pytest.mark.asyncio
async def test_single_forecast_missing_model_id(client):
    """POST /api/v1/forecasting/single with missing model_id returns 422."""
    resp = await client.post(
        "/api/v1/forecasting/single",
        json={"start_date": "2024-01-01"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_single_forecast_empty_body(client):
    """POST /api/v1/forecasting/single with empty body returns 422."""
    resp = await client.post("/api/v1/forecasting/single", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_single_forecast_nonexistent_model(client):
    """POST /api/v1/forecasting/single with nonexistent model returns 404."""
    resp = await client.post(
        "/api/v1/forecasting/single",
        json={"model_id": "nonexistent-model"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_run_alias_accepts_same_body(client):
    """POST /api/v1/forecasting/run is an alias for /single and accepts the same body."""
    resp = await client.post(
        "/api/v1/forecasting/run",
        json={"model_id": "nonexistent-model"},
    )
    # Should go through the same codepath as /single → 404 for missing model
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_run_alias_missing_model_id(client):
    """POST /api/v1/forecasting/run with missing model_id returns 422."""
    resp = await client.post("/api/v1/forecasting/run", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_rolling_forecast_missing_model_id(client):
    """POST /api/v1/forecasting/rolling with missing model_id returns 422."""
    resp = await client.post(
        "/api/v1/forecasting/rolling",
        json={"start_date": "2024-01-01", "forecast_horizon": 30},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_rolling_forecast_missing_required_fields(client):
    """POST /api/v1/forecasting/rolling missing start_date/horizon returns 422."""
    resp = await client.post(
        "/api/v1/forecasting/rolling",
        json={"model_id": "some-model"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_rolling_forecast_nonexistent_model(client):
    """POST /api/v1/forecasting/rolling with nonexistent model returns 404."""
    resp = await client.post(
        "/api/v1/forecasting/rolling",
        json={
            "model_id": "nonexistent-model",
            "start_date": "2024-01-01",
            "forecast_horizon": 30,
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_comparison_forecast_missing_model_id(client):
    """POST /api/v1/forecasting/comparison with missing model_id returns 422."""
    resp = await client.post(
        "/api/v1/forecasting/comparison",
        json={"start_date": "2024-01-01", "forecast_horizon": 30},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_comparison_forecast_nonexistent_model(client):
    """POST /api/v1/forecasting/comparison with nonexistent model returns 404."""
    resp = await client.post(
        "/api/v1/forecasting/comparison",
        json={
            "model_id": "nonexistent-model",
            "start_date": "2024-01-01",
            "forecast_horizon": 30,
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_global_forecast_missing_model_id(client):
    """POST /api/v1/forecasting/global with missing model_id returns 422."""
    resp = await client.post("/api/v1/forecasting/global", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_global_forecast_nonexistent_model(client):
    """POST /api/v1/forecasting/global with nonexistent model returns 404."""
    resp = await client.post(
        "/api/v1/forecasting/global",
        json={"model_id": "nonexistent-model"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_single_forecast_with_all_optional_fields(client):
    """POST /api/v1/forecasting/single with all optional fields returns 404 (model not found)."""
    resp = await client.post(
        "/api/v1/forecasting/single",
        json={
            "model_id": "nonexistent-model",
            "start_date": "2024-06-15",
            "use_covariates": False,
            "freq": "W",
            "horizon": 14,
            "dataset_id": "ds-123",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_rolling_forecast_with_stride(client):
    """POST /api/v1/forecasting/rolling with custom stride returns 404 (model not found)."""
    resp = await client.post(
        "/api/v1/forecasting/rolling",
        json={
            "model_id": "nonexistent-model",
            "start_date": "2024-01-01",
            "forecast_horizon": 30,
            "stride": 7,
            "use_covariates": False,
            "freq": "D",
        },
    )
    assert resp.status_code == 404
