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
