"""Tests for the counterfactual API endpoints.

Tests endpoint status codes and request validation without requiring
actual trained models or database connections.
"""

import pytest


@pytest.mark.asyncio
async def test_run_cf_valid_request(auth_client):
    """POST /api/v1/counterfactual/run validates model ownership upfront.

    A well-formed request for a nonexistent/unowned model returns 404 (the
    202 happy path requires a real owned model — see integration fixtures TODO).
    """
    resp = await auth_client.post(
        "/api/v1/counterfactual/run",
        json={"model_id": "test-model-abc", "method": "physcf"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_run_cf_missing_model_id(auth_client):
    """POST /api/v1/counterfactual/run with missing model_id returns 422."""
    resp = await auth_client.post(
        "/api/v1/counterfactual/run",
        json={"method": "physcf"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_run_cf_defaults(auth_client):
    """POST /api/v1/counterfactual/run with default method; nonexistent model -> 404."""
    resp = await auth_client.post(
        "/api/v1/counterfactual/run",
        json={"model_id": "test-model-defaults"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_generate_physcf(auth_client):
    """POST /api/v1/counterfactual/generate; nonexistent model -> 404."""
    resp = await auth_client.post(
        "/api/v1/counterfactual/generate",
        json={"model_id": "test-model-physcf"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_generate_optuna(auth_client):
    """POST /api/v1/counterfactual/generate-optuna; nonexistent model -> 404."""
    resp = await auth_client.post(
        "/api/v1/counterfactual/generate-optuna",
        json={"model_id": "test-model-optuna", "n_trials": 50},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_generate_comet(auth_client):
    """POST /api/v1/counterfactual/generate-comet; nonexistent model -> 404."""
    resp = await auth_client.post(
        "/api/v1/counterfactual/generate-comet",
        json={"model_id": "test-model-comet", "k_sigma": 3.0},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stream_invalid_task_id(auth_client):
    """GET /api/v1/counterfactual/{invalid_id}/stream returns 404."""
    resp = await auth_client.get("/api/v1/counterfactual/nonexistent-999/stream")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ips_reference_nonexistent_model(auth_client):
    """GET /api/v1/counterfactual/ips-reference with nonexistent model returns 404."""
    resp = await auth_client.get(
        "/api/v1/counterfactual/ips-reference",
        params={"model_id": "nonexistent-model"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_ips_reference_missing_model_id(auth_client):
    """GET /api/v1/counterfactual/ips-reference without model_id returns 422."""
    resp = await auth_client.get("/api/v1/counterfactual/ips-reference")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_pastas_validate_nonexistent_model(auth_client):
    """POST /api/v1/counterfactual/pastas-validate with nonexistent model returns 404."""
    resp = await auth_client.post(
        "/api/v1/counterfactual/pastas-validate",
        json={"model_id": "nonexistent", "cf_task_id": "fake-task"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_pastas_validate_missing_fields(auth_client):
    """POST /api/v1/counterfactual/pastas-validate with missing fields returns 422."""
    resp = await auth_client.post(
        "/api/v1/counterfactual/pastas-validate",
        json={"model_id": "some-model"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_run_cf_empty_body(auth_client):
    """POST /api/v1/counterfactual/run with empty body returns 422."""
    resp = await auth_client.post("/api/v1/counterfactual/run", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_run_cf_with_modifications(auth_client):
    """POST /api/v1/counterfactual/run with modifications; nonexistent model -> 404."""
    resp = await auth_client.post(
        "/api/v1/counterfactual/run",
        json={
            "model_id": "test-model-mods",
            "method": "physcf",
            "modifications": {"precip": 1.2, "temp": -0.5},
        },
    )
    assert resp.status_code == 404
