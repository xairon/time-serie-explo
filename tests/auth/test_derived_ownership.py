import uuid
import pytest
from unittest.mock import patch

from api.auth.tokens import create_session_token


def _cookies(u):
    return {"junon_session": create_session_token(u.id, u.token_version)}


class _ForeignModelReg:
    def get_model_owner(self, model_id):
        return str(uuid.uuid4())


class _ForeignDatasetReg:
    def get_owner(self, dataset_id):
        return str(uuid.uuid4())


@pytest.mark.asyncio
async def test_forecast_on_foreign_model_404(client, make_user):
    u = await make_user(email="fc-intruder@test.fr")
    body = {"model_id": str(uuid.uuid4()), "use_covariates": True, "freq": "D"}
    with patch("api.auth.ownership._model_registry", return_value=_ForeignModelReg()):
        res = await client.post(
            "/api/v1/forecasting/single", json=body, cookies=_cookies(u)
        )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_explain_residuals_foreign_model_404(client, make_user):
    u = await make_user(email="ex-intruder@test.fr")
    mid = str(uuid.uuid4())
    with patch("api.auth.ownership._model_registry", return_value=_ForeignModelReg()):
        res = await client.get(
            f"/api/v1/explainability/{mid}/residuals", cookies=_cookies(u)
        )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_counterfactual_foreign_model_404(client, make_user):
    u = await make_user(email="cf-intruder@test.fr")
    body = {"model_id": str(uuid.uuid4()), "method": "physcf"}
    with patch("api.auth.ownership._model_registry", return_value=_ForeignModelReg()):
        res = await client.post(
            "/api/v1/counterfactual/generate", json=body, cookies=_cookies(u)
        )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_training_foreign_dataset_404(client, make_user):
    u = await make_user(email="tr-intruder@test.fr")
    body = {"dataset_id": str(uuid.uuid4()), "model_name": "TFTModel"}
    with patch("api.auth.ownership._dataset_registry", return_value=_ForeignDatasetReg()):
        res = await client.post(
            "/api/v1/training/start", json=body, cookies=_cookies(u)
        )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_pastas_model_analytics_foreign_404(client, make_user):
    """Pastas analytics endpoints must return 404 for a foreign model."""
    u = await make_user(email="pastasintruder@test.fr")
    fake_run_id = uuid.uuid4().hex  # 32-char hex, passes _validate_run_id

    with patch("api.auth.ownership._model_registry", return_value=_ForeignModelReg()):
        res = await client.get(
            f"/api/v1/pastas/models/{fake_run_id}/signatures",
            cookies=_cookies(u),
        )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_pastas_model_get_foreign_404(client, make_user):
    """GET /models/{run_id} must return 404 for a foreign model."""
    u = await make_user(email="pastasget-intruder@test.fr")
    fake_run_id = uuid.uuid4().hex

    with patch("api.auth.ownership._model_registry", return_value=_ForeignModelReg()):
        res = await client.get(
            f"/api/v1/pastas/models/{fake_run_id}",
            cookies=_cookies(u),
        )
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_pastas_model_delete_foreign_404(client, make_user):
    """DELETE /models/{run_id} must return 404 for a foreign model."""
    u = await make_user(email="pastasdelete-intruder@test.fr")
    fake_run_id = uuid.uuid4().hex

    with patch("api.auth.ownership._model_registry", return_value=_ForeignModelReg()):
        res = await client.delete(
            f"/api/v1/pastas/models/{fake_run_id}",
            cookies=_cookies(u),
        )
    assert res.status_code == 404
