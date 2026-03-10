"""Tests for the explainability API endpoints.

Tests endpoint status codes and request validation without requiring
actual trained models or database connections.
"""

import pytest


@pytest.mark.asyncio
async def test_feature_importance_post_missing_model_id(client):
    """POST /api/v1/explainability/feature-importance with missing model_id returns 422."""
    resp = await client.post(
        "/api/v1/explainability/feature-importance",
        json={"method": "correlation"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_feature_importance_post_empty_body(client):
    """POST /api/v1/explainability/feature-importance with empty body returns 422."""
    resp = await client.post("/api/v1/explainability/feature-importance", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_feature_importance_get_nonexistent(client):
    """GET /api/v1/explainability/{model_id}/feature-importance returns 404 for nonexistent model."""
    resp = await client.get("/api/v1/explainability/nonexistent-model/feature-importance")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_feature_importance_post_nonexistent(client):
    """POST /api/v1/explainability/feature-importance with nonexistent model returns 404."""
    resp = await client.post(
        "/api/v1/explainability/feature-importance",
        json={"model_id": "nonexistent-model", "method": "correlation"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_shap_missing_model_id(client):
    """POST /api/v1/explainability/shap with missing model_id returns 422."""
    resp = await client.post(
        "/api/v1/explainability/shap",
        json={"method": "shap"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_shap_nonexistent_model(client):
    """POST /api/v1/explainability/shap with nonexistent model returns 404."""
    resp = await client.post(
        "/api/v1/explainability/shap",
        json={"model_id": "nonexistent-model"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_attention_missing_model_id(client):
    """POST /api/v1/explainability/attention with missing model_id returns 422."""
    resp = await client.post(
        "/api/v1/explainability/attention",
        json={"method": "attention"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_attention_nonexistent_model(client):
    """POST /api/v1/explainability/attention with nonexistent model returns 404."""
    resp = await client.post(
        "/api/v1/explainability/attention",
        json={"model_id": "nonexistent-model"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_gradients_missing_model_id(client):
    """POST /api/v1/explainability/gradients with missing model_id returns 422."""
    resp = await client.post(
        "/api/v1/explainability/gradients",
        json={"method": "saliency"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_gradients_nonexistent_model(client):
    """POST /api/v1/explainability/gradients with nonexistent model returns 404."""
    resp = await client.post(
        "/api/v1/explainability/gradients",
        json={"model_id": "nonexistent-model"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_shap_with_custom_params(client):
    """POST /api/v1/explainability/shap with custom n_samples returns 404 (model not found)."""
    resp = await client.post(
        "/api/v1/explainability/shap",
        json={"model_id": "nonexistent-model", "n_samples": 50},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_gradients_with_method_variants(client):
    """POST /api/v1/explainability/gradients accepts method variants."""
    for method in ("saliency", "integrated_gradients", "deeplift"):
        resp = await client.post(
            "/api/v1/explainability/gradients",
            json={"model_id": "nonexistent-model", "method": method},
        )
        assert resp.status_code == 404, f"Expected 404 for method={method}"


@pytest.mark.asyncio
async def test_feature_importance_permutation_nonexistent(client):
    """POST /api/v1/explainability/feature-importance with permutation method returns 404."""
    resp = await client.post(
        "/api/v1/explainability/feature-importance",
        json={"model_id": "nonexistent-model", "method": "permutation", "n_permutations": 3},
    )
    assert resp.status_code == 404
