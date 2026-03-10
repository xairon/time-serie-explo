"""Tests for the counterfactual API endpoints.

Tests endpoint status codes and request validation without requiring
actual trained models or database connections.
"""

import pytest


@pytest.mark.asyncio
async def test_run_cf_valid_request(client):
    """POST /api/v1/counterfactual/run with valid body returns 202."""
    resp = await client.post(
        "/api/v1/counterfactual/run",
        json={"model_id": "test-model-abc", "method": "physcf"},
    )
    # 202 = task accepted (background thread will fail but the endpoint itself succeeds)
    assert resp.status_code == 202
    data = resp.json()
    assert "task_id" in data
    assert "status" in data
    assert data["status"].lower() in ("pending", "running")


@pytest.mark.asyncio
async def test_run_cf_missing_model_id(client):
    """POST /api/v1/counterfactual/run with missing model_id returns 422."""
    resp = await client.post(
        "/api/v1/counterfactual/run",
        json={"method": "physcf"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_run_cf_defaults(client):
    """POST /api/v1/counterfactual/run uses default method and params."""
    resp = await client.post(
        "/api/v1/counterfactual/run",
        json={"model_id": "test-model-defaults"},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "task_id" in data


@pytest.mark.asyncio
async def test_generate_physcf(client):
    """POST /api/v1/counterfactual/generate returns 202."""
    resp = await client.post(
        "/api/v1/counterfactual/generate",
        json={"model_id": "test-model-physcf"},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "task_id" in data
    assert "status" in data


@pytest.mark.asyncio
async def test_generate_optuna(client):
    """POST /api/v1/counterfactual/generate-optuna returns 202."""
    resp = await client.post(
        "/api/v1/counterfactual/generate-optuna",
        json={"model_id": "test-model-optuna", "n_trials": 50},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "task_id" in data


@pytest.mark.asyncio
async def test_generate_comet(client):
    """POST /api/v1/counterfactual/generate-comet returns 202."""
    resp = await client.post(
        "/api/v1/counterfactual/generate-comet",
        json={"model_id": "test-model-comet", "k_sigma": 3.0},
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "task_id" in data


@pytest.mark.asyncio
async def test_stream_invalid_task_id(client):
    """GET /api/v1/counterfactual/{invalid_id}/stream returns 404."""
    resp = await client.get("/api/v1/counterfactual/nonexistent-999/stream")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ips_reference_nonexistent_model(client):
    """GET /api/v1/counterfactual/ips-reference with nonexistent model returns 404."""
    resp = await client.get(
        "/api/v1/counterfactual/ips-reference",
        params={"model_id": "nonexistent-model"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ips_reference_missing_model_id(client):
    """GET /api/v1/counterfactual/ips-reference without model_id returns 422."""
    resp = await client.get("/api/v1/counterfactual/ips-reference")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_pastas_validate_nonexistent_model(client):
    """POST /api/v1/counterfactual/pastas-validate with nonexistent model returns 404."""
    resp = await client.post(
        "/api/v1/counterfactual/pastas-validate",
        json={"model_id": "nonexistent", "cf_task_id": "fake-task"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_pastas_validate_missing_fields(client):
    """POST /api/v1/counterfactual/pastas-validate with missing fields returns 422."""
    resp = await client.post(
        "/api/v1/counterfactual/pastas-validate",
        json={"model_id": "some-model"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_run_cf_empty_body(client):
    """POST /api/v1/counterfactual/run with empty body returns 422."""
    resp = await client.post("/api/v1/counterfactual/run", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_run_cf_with_modifications(client):
    """POST /api/v1/counterfactual/run with perturbation modifications accepted."""
    resp = await client.post(
        "/api/v1/counterfactual/run",
        json={
            "model_id": "test-model-mods",
            "method": "physcf",
            "modifications": {"precip": 1.2, "temp": -0.5},
        },
    )
    assert resp.status_code == 202
