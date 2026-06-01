import uuid
import pytest
from unittest.mock import patch

from api.auth.tokens import create_session_token
from api.models_db import UserRole


def _cookies(u):
    return {"junon_session": create_session_token(u.id, u.token_version)}


@pytest.mark.asyncio
async def test_list_models_filters_by_owner(client, make_user):
    u = await make_user(email="owner@test.fr")
    captured = {}

    class FakeRegistry:
        def list_all_models(self, model_type=None, model_name=None, owner_filter=None):
            captured["filter"] = owner_filter
            return []

    with patch("api.routers.models._get_model_registry", return_value=FakeRegistry()):
        res = await client.get("/api/v1/models", cookies=_cookies(u))
    assert res.status_code == 200
    assert captured["filter"] == f"tags.owner_id = '{u.id}'"


@pytest.mark.asyncio
async def test_get_model_of_other_user_404(client, make_user):
    u = await make_user(email="intruder@test.fr", role=UserRole.user)

    class FakeRegistry:
        def get_model_owner(self, model_id):
            return str(uuid.uuid4())

    with patch("api.routers.models._get_model_registry", return_value=FakeRegistry()):
        res = await client.get(f"/api/v1/models/{uuid.uuid4()}", cookies=_cookies(u))
    assert res.status_code == 404
