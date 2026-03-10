"""Tests for Pydantic schema validation.

Direct instantiation tests for API schemas without requiring
any infrastructure (no DB, no models, no Redis).
"""

import pytest
from pydantic import ValidationError

from api.schemas.counterfactual import CFGenerateRequest, CFResult, IPSReferenceRequest, PastasValidateRequest
from api.schemas.forecasting import (
    ForecastRequest,
    RollingForecastRequest,
    ComparisonForecastRequest,
    GlobalForecastRequest,
    ForecastResult,
)
from api.schemas.models import AvailableModel, ModelSummary, ModelDetail
from api.schemas.datasets import DatasetProfile, DatasetSummary, DatasetDetail, DatasetCreateRequest


# --------------------------------------------------------------------------- #
# CFGenerateRequest
# --------------------------------------------------------------------------- #


class TestCFGenerateRequest:
    def test_defaults(self):
        req = CFGenerateRequest(model_id="abc")
        assert req.method == "physcf"
        assert req.target_ips_class == "normal"
        assert req.lambda_prox == 0.1
        assert req.n_iter == 500
        assert req.lr == 0.02
        assert req.cc_rate == 0.07
        assert req.device == "cpu"
        assert req.n_trials == 200
        assert req.seed == 42
        assert req.k_sigma == 4.0
        assert req.lambda_smooth == 0.1
        assert req.modifications == {}

    def test_all_fields(self):
        req = CFGenerateRequest(
            model_id="model-123",
            method="optuna",
            target_ips_class="low",
            lambda_prox=0.5,
            n_iter=100,
            lr=0.01,
            cc_rate=0.1,
            device="cuda",
            n_trials=50,
            seed=7,
            k_sigma=2.0,
            lambda_smooth=0.05,
            modifications={"precip": 1.5},
        )
        assert req.model_id == "model-123"
        assert req.method == "optuna"
        assert req.modifications == {"precip": 1.5}

    def test_missing_model_id_raises(self):
        with pytest.raises(ValidationError):
            CFGenerateRequest()

    def test_method_accepts_any_string(self):
        """Method field is a plain string, not an enum -- any value is accepted."""
        req = CFGenerateRequest(model_id="x", method="custom_method")
        assert req.method == "custom_method"


class TestCFResult:
    def test_minimal(self):
        r = CFResult(task_id="t1", status="pending")
        assert r.result is None
        assert r.error is None

    def test_with_result(self):
        r = CFResult(task_id="t1", status="completed", result={"key": "val"})
        assert r.result == {"key": "val"}


class TestIPSReferenceRequest:
    def test_defaults(self):
        req = IPSReferenceRequest(model_id="m1")
        assert req.window == 1
        assert req.aquifer_type is None

    def test_all_fields(self):
        req = IPSReferenceRequest(model_id="m1", window=6, aquifer_type="alluvial")
        assert req.window == 6


class TestPastasValidateRequest:
    def test_defaults(self):
        req = PastasValidateRequest(model_id="m1", cf_task_id="cf1")
        assert req.gamma == 1.5

    def test_missing_cf_task_id_raises(self):
        with pytest.raises(ValidationError):
            PastasValidateRequest(model_id="m1")


# --------------------------------------------------------------------------- #
# ForecastRequest and variants
# --------------------------------------------------------------------------- #


class TestForecastRequest:
    def test_defaults(self):
        req = ForecastRequest(model_id="m1")
        assert req.start_date is None
        assert req.use_covariates is True
        assert req.freq == "D"
        assert req.horizon is None
        assert req.dataset_id is None

    def test_optional_fields(self):
        req = ForecastRequest(
            model_id="m1",
            start_date="2024-01-01",
            use_covariates=False,
            freq="W",
            horizon=14,
            dataset_id="ds-1",
        )
        assert req.start_date == "2024-01-01"
        assert req.use_covariates is False
        assert req.horizon == 14

    def test_missing_model_id_raises(self):
        with pytest.raises(ValidationError):
            ForecastRequest()


class TestRollingForecastRequest:
    def test_required_fields(self):
        with pytest.raises(ValidationError):
            RollingForecastRequest(model_id="m1")

    def test_defaults(self):
        req = RollingForecastRequest(
            model_id="m1", start_date="2024-01-01", forecast_horizon=30
        )
        assert req.stride == 1
        assert req.use_covariates is True
        assert req.freq == "D"


class TestComparisonForecastRequest:
    def test_required_fields(self):
        with pytest.raises(ValidationError):
            ComparisonForecastRequest(model_id="m1")

    def test_valid(self):
        req = ComparisonForecastRequest(
            model_id="m1", start_date="2024-01-01", forecast_horizon=30
        )
        assert req.forecast_horizon == 30


class TestGlobalForecastRequest:
    def test_defaults(self):
        req = GlobalForecastRequest(model_id="m1")
        assert req.use_covariates is True
        assert req.freq == "D"


class TestForecastResult:
    def test_empty_result(self):
        r = ForecastResult()
        assert r.predictions == []
        assert r.target == []
        assert r.metrics == {}
        assert r.horizon is None
        assert r.predictions_onestep is None
        assert r.forecast_windows is None

    def test_with_data(self):
        r = ForecastResult(
            predictions=[{"date": "2024-01-01", "value": 1.0}],
            metrics={"rmse": 0.5},
            horizon=30,
        )
        assert len(r.predictions) == 1
        assert r.metrics["rmse"] == 0.5


# --------------------------------------------------------------------------- #
# AvailableModel
# --------------------------------------------------------------------------- #


class TestAvailableModel:
    def test_required_fields(self):
        with pytest.raises(ValidationError):
            AvailableModel()

    def test_minimal(self):
        m = AvailableModel(name="TFT", is_torch=True)
        assert m.name == "TFT"
        assert m.is_torch is True
        assert m.description == ""
        assert m.category == ""
        assert m.default_hyperparams == {}

    def test_all_fields(self):
        m = AvailableModel(
            name="NLinear",
            is_torch=True,
            description="A linear model",
            category="linear",
            default_hyperparams={"input_chunk_length": 30},
        )
        assert m.default_hyperparams["input_chunk_length"] == 30


class TestModelSummary:
    def test_required_fields(self):
        with pytest.raises(ValidationError):
            ModelSummary()

    def test_minimal(self):
        m = ModelSummary(
            model_id="abc",
            model_name="TFT_station1",
            model_type="single",
            created_at="2024-01-01T00:00:00",
        )
        assert m.stations == []
        assert m.metrics == {}
        assert m.primary_station is None


class TestModelDetail:
    def test_inherits_summary(self):
        d = ModelDetail(
            model_id="abc",
            model_name="TFT_station1",
            model_type="single",
            created_at="2024-01-01T00:00:00",
            run_id="run-xyz",
        )
        assert d.run_id == "run-xyz"
        assert d.hyperparams == {}
        assert d.display_name == ""


# --------------------------------------------------------------------------- #
# DatasetProfile
# --------------------------------------------------------------------------- #


class TestDatasetProfile:
    def test_optional_fields_nullable(self):
        p = DatasetProfile(
            columns={"col1": {"mean": 1.0}},
            shape=[100, 3],
            dtypes={"col1": "float64"},
            missing={"col1": 0},
        )
        assert p.correlation is None
        assert p.timeseries_data is None

    def test_with_correlation(self):
        p = DatasetProfile(
            columns={},
            shape=[10, 2],
            dtypes={},
            missing={},
            correlation={"a": {"b": 0.9}},
        )
        assert p.correlation["a"]["b"] == 0.9

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            DatasetProfile()


class TestDatasetSummary:
    def test_defaults(self):
        s = DatasetSummary(name="my_dataset")
        assert s.id == ""
        assert s.source_file == ""
        assert s.covariates == []
        assert s.n_rows == 0
        assert s.stations == []
        assert s.station_column is None

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            DatasetSummary()


class TestDatasetCreateRequest:
    def test_required_fields(self):
        with pytest.raises(ValidationError):
            DatasetCreateRequest()

    def test_defaults(self):
        r = DatasetCreateRequest(name="ds", target_column="level")
        assert r.covariate_columns == []
        assert r.station_column is None
        assert r.stations == []
        assert r.preprocessing == {}
