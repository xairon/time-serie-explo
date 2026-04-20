"""Integration tests for Pastas API endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    tmp = tmp_path_factory.mktemp("mlflow")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp / 'mlflow.db'}"
    from api.main import app
    return TestClient(app)


def test_options_endpoint(client):
    resp = client.get("/api/v1/pastas/options")
    assert resp.status_code == 200
    data = resp.json()
    assert "recharge" in data
    assert "Gamma" in data["response"]


def test_models_empty(client):
    resp = client.get("/api/v1/pastas/models")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
