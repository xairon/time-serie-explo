import uuid
import pytest
from unittest.mock import patch

from api.auth.tokens import create_session_token


def _cookies(u):
    return {"junon_session": create_session_token(u.id, u.token_version)}


@pytest.mark.asyncio
async def test_list_datasets_filtered_by_owner(client, make_user):
    u = await make_user(email="dsowner@test.fr")
    captured = {}

    class FakeReg:
        def scan_datasets(self, owner_id=None):
            captured["owner_id"] = owner_id
            return []

    with patch("api.routers.datasets._get_registry", return_value=FakeReg()):
        res = await client.get("/api/v1/datasets", cookies=_cookies(u))
    assert res.status_code == 200
    assert captured["owner_id"] == str(u.id)


@pytest.mark.asyncio
async def test_get_other_dataset_404(client, make_user):
    u = await make_user(email="dsintruder@test.fr")

    class FakeReg:
        def get_owner(self, dataset_id):
            return str(uuid.uuid4())

    with patch("api.routers.datasets._get_registry", return_value=FakeReg()):
        res = await client.get("/api/v1/datasets/whatever", cookies=_cookies(u))
    assert res.status_code == 404
