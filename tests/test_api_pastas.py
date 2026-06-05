"""Integration tests for Pastas API endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    import uuid
    tmp = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{tmp / 'mlflow.db'}"
    os.environ["MLFLOW_TRACKING_URI"] = uri
    # The endpoints read settings.mlflow_tracking_uri (not the env var), so point
    # it at the local sqlite store; otherwise they try to reach a real MLflow server.
    from api.config import settings
    settings.mlflow_tracking_uri = uri
    from api.main import app
    from api.auth.deps import get_current_user
    from api.models_db import User, UserRole

    def _fake_user():
        return User(
            id=uuid.uuid4(), email="pastas@test.fr", display_name="pastas",
            password_hash="x", role=UserRole.user, is_active=True,
        )

    app.dependency_overrides[get_current_user] = _fake_user
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)


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
